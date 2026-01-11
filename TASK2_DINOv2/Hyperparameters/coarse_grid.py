import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.task2_DINOv2_dataset import SPairDataset
from utils.setup_data import setup_data
from TASK2_DINOv2.Hyperparameters.loss import InfoNCELoss

def coarse_grid(lr_to_try, w_decay):
    print(f"\n{'='*60}")
    print(f">>> TESTING LR: {lr_to_try} | WD: {w_decay} <<<")
    print(f"{'='*60}")
    
    data_root = setup_data() 
    if data_root is None: return

    base_dir = os.path.join(data_root, 'SPair-71k','Spair-71k') 
    pair_ann_path = os.path.join(base_dir, 'PairAnnotation')
    layout_path = os.path.join(base_dir, 'Layout')
    image_path = os.path.join(base_dir, 'JPEGImages')

    train_dataset = SPairDataset(
        pair_ann_path, layout_path, image_path, 
        dataset_size='large', pck_alpha=0.5, datatype='trn'
    )
    
    trn_dataloader = DataLoader(
        train_dataset, 
        batch_size=5, 
        shuffle=True, 
        num_workers=4,           
        persistent_workers=True, 
        pin_memory=True          
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    criterion = InfoNCELoss(temperature=0.07).to(device)
  
    for param in model.parameters(): param.requires_grad = False
    for block in model.blocks[-2:]:
        for param in block.parameters(): param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=lr_to_try, weight_decay=w_decay)
    
    scaler = torch.amp.GradScaler('cuda')

    num_epochs = 5
    model.train()
    limit_batches = 1000

    for epoch in range(num_epochs):
        total_epoch_loss = 0
        pbar = tqdm(trn_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # --- FIXED LOOP STRUCTURE ---
        for i, batch in enumerate(pbar):
            
            # 1. Check Limit
            if i >= limit_batches:
                break 
            
            # 2. Training Logic (INDENTED INSIDE THE LOOP)
            src_img = batch['src_img'].to(device)
            trg_img = batch['trg_img'].to(device)
            src_kps = batch['src_kps'].to(device)
            trg_kps = batch['trg_kps'].to(device)
            mask    = batch['valid_mask'].to(device)
            
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                output_src = model.forward_features(src_img)
                output_trg = model.forward_features(trg_img)
                
                feat_src = output_src['x_norm_patchtokens']
                feat_trg = output_trg['x_norm_patchtokens']
                
                B, N, C = feat_src.shape
                H, W = 37, 37
                feat_src = feat_src.permute(0, 2, 1).reshape(B, C, H, W)
                feat_trg = feat_trg.permute(0, 2, 1).reshape(B, C, H, W)

                loss = criterion(feat_src, feat_trg, src_kps, trg_kps, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average based on the ACTUAL number of batches run (limit_batches)
        # We use min(i, len) to be safe in case dataset is smaller than limit
        actual_batches = i + 1
        avg_loss = total_epoch_loss / actual_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.4f}")

    print(f"--> Done. Final Avg Loss for LR {lr_to_try}/WD {w_decay}: {avg_loss:.4f}")

if __name__ == '__main__':
   
    hyperparameters=[ # (lr, w_decay)
        (1e-4, 1e-3),
        (1e-4, 1e-4),
        ]
    print("Starting Optimized Grid Search...")
    for x,y in hyperparameters:
        coarse_grid(x, y)