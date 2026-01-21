import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TASK_2_SAM.setup import DenseCrossEntropyLoss, configure_model
from utils.common import download_sam_model
from data.dataset import SPairDataset

def check_overfitting():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Overfitting Test on {device} ---")

    # 1. Load Data
    dataset_root = 'dataset/SPair-71k'
    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='trn')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # 2. Load Model
    checkpoint_dir = 'checkpoints'
    ckpt_path = download_sam_model(checkpoint_dir)
    model = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    model.to(device)
    model.train()

    # 3. Initialize YOUR Loss Function
    criterion = DenseCrossEntropyLoss(temperature=0.07).to(device)

    # 4. Prepare Batch
    fixed_batches = []
    iterator = iter(dataloader)
    for _ in range(5):
        b = next(iterator)
        batch_data = {
            'src': b['src_img'].to(device),
            'trg': b['trg_img'].to(device),
            'kps_src': b['src_kps'].to(device),
            'kps_trg': b['trg_kps'].to(device),
            'kps_mask': b['kps_valid'].to(device)
        }
        fixed_batches.append(batch_data)

    # NEW
    # 5 Configure model for fine-tuning
    LAYERS_TO_UNFREEZE = 1
    configure_model(model, unfreeze_last_n_layers=LAYERS_TO_UNFREEZE)

    # 6. Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-3, weight_decay=0.0)
    
    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in fixed_batches:
            optimizer.zero_grad()

            optimizer.zero_grad()
            feat_src = model.image_encoder(batch['src'])
            feat_trg = model.image_encoder(batch['trg'])

            loss = criterion(feat_src, feat_trg,
                                batch['kps_src'],
                                batch['kps_trg'],
                                batch['kps_mask'])

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(fixed_batches)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Loss = {avg_loss:.6f}")
            
            if avg_loss < 0.01: # Soglia vicina a zero
                print("\n✅ SUCCESSO! Il modello ha memorizzato il sottoinsieme.")
                print("Il fine-tuning funziona, i gradienti fluiscono correttamente.")
                break
    if avg_loss > 0.1:
        print("\n❌ FALLIMENTO. Il modello non riesce a fare overfitting.")

if __name__ == "__main__":
    check_overfitting()