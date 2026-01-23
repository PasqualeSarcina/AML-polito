import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
import copy
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TASK_2_SAM.setup import MyCrossEntropyLoss, configure_model
from utils.common import download_sam_model
from data.dataset import SPairDataset

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    print(f">>> 🔒 SEED FISSATO A {seed} <<<")

def find_lr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    results = {}

    # 1. Load Data
    dataset_root = 'dataset/SPair-71k'
    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='trn')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    checkpoint_dir = 'checkpoints'
    ckpt_path = download_sam_model(checkpoint_dir)
    model = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    model.to(device)

    LAYERS_TO_UNFREEZE = 1
    configure_model(model, unfreeze_last_n_layers=LAYERS_TO_UNFREEZE)

    initial_state = copy.deepcopy(model.state_dict()) # Salva lo stato iniziale per reset

    criterion = MyCrossEntropyLoss(temperature=0.07).to(device)

    for lr in learning_rates:
        print(f"\n--- Testando Learning Rate: {lr} ---")
        
        seed_everything(42)

        # Reset del modello
        model.load_state_dict(initial_state)
        model.train()
        
        # Optimizer con il LR variabile e piccolo weight decay
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr,           
                                weight_decay=1e-4) 
        
        losses = [] # Lista per salvare la storia della loss di questo LR

        for i, batch in enumerate(dataloader):
            if i >= 100:
                break

            src = batch['src_img'].to(device)
            trg = batch['trg_img'].to(device)
            kps_src = batch['src_kps'].to(device)
            kps_trg = batch['trg_kps'].to(device)
            kps_mask = batch['kps_valid'].to(device)

            optimizer.zero_grad()

            feat_src = model.image_encoder(src)
            feat_trg = model.image_encoder(trg)

            loss = criterion(feat_src, feat_trg, kps_src, kps_trg, kps_mask)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if i % 20 == 0:
                print(f"Learning Rate: {lr:.1e} | Batch {i+1} | Loss: {loss.item():.4f}")
        
        results[lr] = losses

    print("\nGenerazione grafico...")
    plt.figure(figsize=(12, 8))
    
    for lr, loss_curve in results.items():
        plt.plot(loss_curve, label=f"LR={lr:.1e}", linewidth=2)

    plt.xlabel("Iterazioni (Batch)", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Confronto Learning Rates per Fine-Tuning", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #plt.yscale('log') #scala logaritmica se le differenze sono enormi
    
    # Salva o mostra
    plt.savefig("lr_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    find_lr()