
import os
from pathlib import Path
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.loss import InfoNCELoss
from utils.utils_download import download
from utils.download_data import download_spair71k
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


def configure_model(model, unfreeze_last_n_layers):
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    
    for param in model.image_encoder.neck.parameters():
        param.requires_grad = True

    blocks_to_train = model.image_encoder.blocks[-unfreeze_last_n_layers :]

    print(f"Scongelamento degli ultimi {len(blocks_to_train)} blocchi.")
    for block in blocks_to_train:
        for param in block.parameters():
            param.requires_grad = True
    

def train_finetune(model, train_loader, val_loader, save_dir, epochs, lr, wd, accumulation_steps, temperature):
    scaler = GradScaler()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = InfoNCELoss(temperature=temperature)

    best_val_loss = float('inf') 


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

        print(f"\nEpoch {epoch+1} completed:")
        print(f"Training Loss:   {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # latest_name = f"sam_latest_{epoch+1}_epochs.pth"
        # torch.save(model.state_dict(), os.path.join(save_dir, latest_name))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_name = "sam_best.pth"
            torch.save(model.state_dict(), save_dir / best_name)
            print("New Best Model Saved")
        
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SAM")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--w_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers to fine-tune")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    args = parser.parse_args()
    seed_everything(42)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dataset_root = 'dataset/SPair-71k'
    if not os.path.exists(dataset_root):
        download_spair71k()

    checkpoint_dir = Path('checkpoints')
    sam_checkpoint = checkpoint_dir / "sam_vit_b_01ec64.pth"
    if not sam_checkpoint.exists():
        download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", sam_checkpoint)

    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device)

    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    train_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', datatype='trn')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    val_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', datatype='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    n_layers = args.n_layers
    n_epochs = args.epochs
    lr = args.lr
    wd = args.w_decay
    accumulation_steps = args.accumulation_steps
    temperature = 0.07  
    configure_model(sam, unfreeze_last_n_layers=n_layers)

    print(f"Fine-tuning ultimi {n_layers} layer")
     
    train_finetune(sam, train_dataloader, val_dataloader, checkpoint_dir,
        n_epochs, lr, wd, accumulation_steps, temperature)
    
    