import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_DINOv3 import SPairDataset
from utils.setup_data_DINOv3 import setup_data
from models.dinov3.model_DINOv3 import load_dinov3_backbone
from Task2_DINOv3.loss import InfoNCELoss

data_root = setup_data() 
if data_root is None: 
    print("Error: Dataset not found. Please run utils/setup_data_DINOv3.py or check data location.")
    sys.exit(1)

base_dir = os.path.join(data_root, 'SPair-71k') 
pair_ann_path = os.path.join(base_dir, 'PairAnnotation')
layout_path = os.path.join(base_dir, 'Layout')
image_path = os.path.join(base_dir, 'JPEGImages')

train_dataset = SPairDataset(
    pair_ann_path, layout_path, image_path, 
    dataset_size='large', pck_alpha=0.5, datatype='trn'
)

val_dataset = SPairDataset(
    pair_ann_path, layout_path, image_path,
    dataset_size="large", pck_alpha=0.5, datatype="val"
)

trn_dataloader = DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=True, 
    num_workers=4,           
    persistent_workers=True, 
    pin_memory=True          
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=4,           
    persistent_workers=True, 
    pin_memory=True          
)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> SEED SET TO {seed} <<<")
    

def fine_tuning(epochs, lr, w_decay, n_layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    dinov3_dir = Path("/content/dinov3") if Path("/content/dinov3").exists() else Path("third_party/dinov3")
    weights_path = Path("checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path.resolve()}")
    model = load_dinov3_backbone(
            dinov3_dir=dinov3_dir,
            weights_path=weights_path,
            device=device,
            sanity_input_size=512,
            verbose=True,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")

    if hasattr(model, 'norm'):
        print("yes 'model.norm' exist.")
        print(model.norm)
    else:
        print("don't exist")
    
    criterion = InfoNCELoss(temperature=0.07).to(device)
    
    for param in model.parameters(): param.requires_grad = False
    for block in model.blocks[-n_layers:]:
        for param in block.parameters(): param.requires_grad = True

    for param in model.norm.parameters(): param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=w_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler('cuda')
    num_epochs = epochs
    best_val_loss=float('inf')
    accumulation_steps = 8  # Simulate batch_size = 8

    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        pbar = tqdm(trn_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}[Training]")
        optimizer.zero_grad()

        model.train()
        for i, batch in enumerate(pbar):

            # 2. Training Logic 
            src_img = batch['src_img'].to(device)
            trg_img = batch['trg_img'].to(device)
            src_kps = batch['src_kps'].to(device)
            trg_kps = batch['trg_kps'].to(device)
            mask    = batch['valid_mask'].to(device)


            with torch.amp.autocast('cuda'):
                output_src = model.forward_features(src_img)
                output_trg = model.forward_features(trg_img)

                feat_src = output_src['x_norm_patchtokens']
                feat_trg = output_trg['x_norm_patchtokens']

                B, N, C = feat_src.shape
                H, W = 32, 32
                feat_src = feat_src.permute(0, 2, 1).reshape(B, C, H, W)
                feat_trg = feat_trg.permute(0, 2, 1).reshape(B, C, H, W)

                loss = criterion(feat_src, feat_trg, src_kps, trg_kps, mask)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trn_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})

        avg_train_loss = total_epoch_loss / len(trn_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Avg Training Loss: {avg_train_loss:.4f}")
        scheduler.step()

        # Print current LR to verify
        current_lr = scheduler.get_last_lr()[0]
        print(f"--> Learning Rate for next epoch: {current_lr:.8f}")

        model.eval()
        val_loss=0
        pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}[Validation]")
        with torch.no_grad():
            for i, batch in enumerate(pbar):

                # 2. Validation Logic
                src_img = batch['src_img'].to(device)
                trg_img = batch['trg_img'].to(device)
                src_kps = batch['src_kps'].to(device)
                trg_kps = batch['trg_kps'].to(device)
                mask    = batch['valid_mask'].to(device)

                with torch.amp.autocast('cuda'):
                    output_src = model.forward_features(src_img)
                    output_trg = model.forward_features(trg_img)

                    feat_src = output_src['x_norm_patchtokens']
                    feat_trg = output_trg['x_norm_patchtokens']

                    B, N, C = feat_src.shape
                    H, W = 32, 32
                    feat_src = feat_src.permute(0, 2, 1).reshape(B, C, H, W)
                    feat_trg = feat_trg.permute(0, 2, 1).reshape(B, C, H, W)

                    loss = criterion(feat_src, feat_trg, src_kps, trg_kps, mask)

                val_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Avg Validation Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss=avg_val_loss
                save_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "dinov3"
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                print(f"--> New Best Model Saved! (Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    EPOCHS = 5
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-2
    N_LAYERS_TO_FINE_TUNE = 1

    fine_tuning( epochs=EPOCHS, lr=LEARNING_RATE, w_decay=WEIGHT_DECAY, n_layers=N_LAYERS_TO_FINE_TUNE)