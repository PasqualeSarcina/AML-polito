import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from segment_anything import sam_model_registry
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TASK_2_SAM.setup import DenseCrossEntropyLoss, configure_model
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

def coarse_grid(dataloader, lr, wd, device, checkpoint_dir):
    seed_everything(42)
    ckpt_path = download_sam_model(checkpoint_dir)
    model = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    model.to(device)

    LAYERS_TO_UNFREEZE = 1
    configure_model(model, unfreeze_last_n_layers=LAYERS_TO_UNFREEZE)

    criterion = DenseCrossEntropyLoss(temperature=0.07).to(device)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr, weight_decay=wd)
    
    scaler = GradScaler()
    epochs = 3
    limit_batches = 1000
    epoch_losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0  
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_batches = 0

        for i, batch in enumerate(pbar):

            if i >= limit_batches:
                break
            
            optimizer.zero_grad()

            src = batch['src_img'].to(device)
            trg = batch['trg_img'].to(device)
            kps_src = batch['src_kps'].to(device)
            kps_trg = batch['trg_kps'].to(device)
            kps_mask = batch['kps_valid'].to(device)
            
            with autocast():
                src_feat = model.image_encoder(src)
                trg_feat = model.image_encoder(trg)
                loss = criterion(src_feat, trg_feat, kps_src, kps_trg, kps_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            total_batches += 1
            pbar.set_postfix({'Loss': loss.item()})
        
        avg_loss = epoch_loss / total_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")

    # Pulizia memoria GPU tra un test e l'altro
    del model
    del optimizer
    torch.cuda.empty_cache()

    return epoch_losses

def plot_grid_results(results):
    """
    Crea un grafico comparativo delle curve di loss.
    results: dizionario {(lr, wd): [loss_ep1, loss_ep2, ...]}
    """
    plt.figure(figsize=(10, 6))
    
    # Ciclo su tutti i risultati per tracciare le linee
    for (lr, wd), losses in results.items():
        # Etichetta per la legenda
        label_str = f"LR={lr}, WD={wd}"
        plt.plot(range(1, len(losses) + 1), losses, marker='o', label=label_str)

    plt.title("Coarse Grid Search Results")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True)
    
    output_filename = "grid_search_chart.png"
    plt.savefig(output_filename)
    print(f"\n[INFO] Grafico salvato come '{output_filename}'. Aprilo per confrontare i modelli!")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = 'dataset/SPair-71k'
    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='small', pck_alpha=0.1, datatype='trn')
    print(f"Dimensione dataset Small: {len(dataset)} coppie.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                             num_workers=4, persistent_workers =True, pin_memory=True)
   
    lr_wd_combinations = [(1e-3, 1e-3),
                          (1e-3, 1e-4),
                          (1e-4, 1e-3),
                          (1e-4, 1e-4),
                          (1e-4, 0)
                        ]
    results = {}

    for lr, wd in lr_wd_combinations:
        print(f"\n--- Coarse Grid Search: LR={lr}, WD={wd} ---")
        loss_history = coarse_grid(dataloader, lr, wd, device, checkpoint_dir='checkpoints')
        results[(lr, wd)] = loss_history

    plot_grid_results(results)

    # Trova il migliore in base all'ultima epoca
    best_combo = min(results, key=lambda k: results[k][-1])
    print("\n==============================")
    print(f"MIGLIORE COMBINAZIONE (Loss finale più bassa):")
    print(f"LR={best_combo[0]}, WD={best_combo[1]}")
    print(f"Loss Finale: {results[best_combo][-1]:.4f}")
    print("==============================")
        