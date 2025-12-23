
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from segment_anything import sam_model_registry

from models.setup import DenseCrossEntropyLoss, configure_model
from utils.geometry import extract_features
from utils.common import download_sam_model
from data.dataset import SPairDataset


def train_finetune(model, train_loader, epochs=1, lr=1e-5, accumulation_steps=4):
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    criterion = DenseCrossEntropyLoss(temperature=0.01)
    model.train() 
    scaler = GradScaler()

    print(f"🚀 Inizio Training per {epochs} epoche con Mixed Precision...") 

    for epoch in range(epochs):
        epoch_loss = 0
        steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for i, batch in enumerate(pbar):
            try:
                # Sposta su GPU
                src = batch['src_img'].to(device)
                trg = batch['trg_img'].to(device)
                kps_src = batch['src_kps'].to(device)
                kps_trg = batch['trg_kps'].to(device)
                kps_mask = batch['kps_valid'].to(device)
                
                # --- MIXED PRECISION CONTEXT ---
                # Eseguiamo il forward pass in float16 dove possibile per risparmiare VRAM
                with autocast():
                    # Forward usando extract_features (con gradienti!)
                    feats_src = extract_features(model, src)
                    feats_trg = extract_features(model, trg)

                    loss = criterion(feats_src, feats_trg, kps_src, kps_trg, kps_mask)
                    loss = loss / accumulation_steps

                # --- BACKPROPAGATION SCALATA ---
                if loss.item() > 0:
                    scaler.scale(loss).backward()   

                    if (i + 1) % accumulation_steps == 0:
                        # Unscale gradienti prima di step (opzionale ma consigliato per clipping)
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                # ----------------------------------

                    current_loss = loss.item() * accumulation_steps
                    epoch_loss += current_loss
                    steps += 1
                    pbar.set_postfix({'loss': epoch_loss / max(steps, 1)})
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"| OOM |", end="") # Stampa discreta
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # Libera grafi
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
            
            # Pulizia aggressiva
            del src, trg, feats_src, feats_trg, loss
            
    return model

if __name__ == "__main__":
    # SETUP PATHS
    dataset_root = 'dataset/SPair-71k' # Aggiusta se necessario
    checkpoint_dir = 'checkpoints'

    # CONFIGURA DEVICE
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # DOWNLOAD & LOAD MODEL
    ckpt_path = download_sam_model(checkpoint_dir)
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    sam.to(device)

    # CONFIGURA MODELLO PER FINETUNING
    n_layers = 1  
    configure_model(sam, unfreeze_last_n_layers=n_layers)

    # DATASET E DATALOADER
    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    train_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='trn')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    # AVVIO TRAINING
    print(f"\n\n{'#'*60}")
    print(f"🧪 ESPERIMENTO: Fine-tuning ultimi {n_layers} layer")
    print(f"{'#'*60}")
    sam_tuned = train_finetune(sam, train_dataloader, epochs=1, lr=1e-5, accumulation_steps=4)

    # SALVA IL MODELLO FINALE
    model_name = f'sam_tuned_{n_layers}layer.pth'
    save_path = os.path.join(checkpoint_dir, model_name)
    torch.save(sam_tuned.state_dict(), save_path)
    print(f"💾 Modello salvato con successo in: {save_path}")
