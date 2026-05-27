
import os
import sys
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from segment_anything import sam_model_registry
import random
import numpy as np
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.loss import InfoNCELoss
from utils.configure_layers import configure_model
from utils.common import download_sam_model
from data.dataset_SAM import SPairDataset

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def train_finetune(model, train_loader, val_loader, save_dir, epochs, lr, wd, accumulation_steps, temperature):
    scaler = GradScaler()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = InfoNCELoss(temperature=temperature)

    best_val_loss = float('inf') 
    train_loss_history = []
    val_loss_history = []

    print(f"Inizio Training per {epochs} epoche")
    
    for epoch in range(epochs):
        model.train() 
        train_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")       

        for i, batch in enumerate(pbar):
            src_img = batch['src_img'].to(device)
            trg_img = batch['trg_img'].to(device)
            kps_src = batch['src_kps'].to(device)
            kps_trg = batch['trg_kps'].to(device)
            kps_mask = batch['kps_valid'].to(device)
            
            with autocast():
                feats_src = model.image_encoder(src_img) # B, C, Hf, Wf
                feats_trg = model.image_encoder(trg_img)
                loss = criterion(feats_src, feats_trg, kps_src, kps_trg, kps_mask)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i+1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': train_loss/max(i, 1)})
        
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]"):                
                src_img = batch['src_img'].to(device)
                trg_img = batch['trg_img'].to(device)
                kps_src = batch['src_kps'].to(device)
                kps_trg = batch['trg_kps'].to(device)
                kps_mask = batch['kps_valid'].to(device)

                with autocast():
                    feats_src = model.image_encoder(src_img) # B, C, Hf, Wf
                    feats_trg = model.image_encoder(trg_img)
                    loss = criterion(feats_src, feats_trg, kps_src, kps_trg, kps_mask)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(f"\nEPOCA {epoch+1} COMPLETATA:")
        print(f"Training Loss:   {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Logica di salvataggio
        # 1. Salva il modello corrente come "latest"
        latest_name = f"sam_latest_{epoch}_epochs.pth"
        torch.save(model.state_dict(), os.path.join(save_dir, latest_name))
        
        # 2. Salva SE è il migliore finora
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_name = "sam_best.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, best_name))
            print(f"Salvato: {best_name}")
        
        print("-" * 60)
    return train_loss_history, val_loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SAM")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--w_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers to fine-tune")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    args = parser.parse_args()
    seed_everything(42)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dataset_root = 'dataset/SPair-71k'
    checkpoint_dir = 'checkpoints'
    results_dir = 'results'

    ckpt_path = download_sam_model(checkpoint_dir)
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    sam.to(device)

    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    train_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='trn')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    print(f"Dataset Training caricato: {len(train_dataset)} coppie.")

    val_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"Dataset Validation caricato: {len(val_dataset)} coppie.")

    n_layers = args.n_layers
    n_epochs = args.epochs
    lr = args.lr
    wd = args.w_decay
    accumulation_steps = args.accumulation_steps
    temperature = 0.07  
    configure_model(sam, unfreeze_last_n_layers=n_layers)

    print(f"Fine-tuning ultimi {n_layers} layer")

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
     
    train_hist, val_hist = train_finetune(
        sam, train_dataloader, val_dataloader, checkpoint_dir,
        n_epochs, lr, wd, accumulation_steps, temperature)
    
    # salviamo gli iperparametri e i risultati in un .json nella cartella checkpoints
    history_data = {
        'run_id': run_id,
        'n_layers': n_layers,
        'lr': lr,
        'wd': wd,
        'acc_step': accumulation_steps,
        'temperature': temperature,
        'train_loss': train_hist,
        'val_loss': val_hist,
        'epochs': n_epochs
    }
    json_filename = f"history_{n_layers}layers_{run_id}.json"
    with open(os.path.join(checkpoint_dir, json_filename), 'w') as f:
        json.dump(history_data, f)
    