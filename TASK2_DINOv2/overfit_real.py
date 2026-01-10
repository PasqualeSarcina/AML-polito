import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.task2_DINOv2_no_regularization import SPairDataset
from utils.setup_data import setup_data
from TASK2_DINOv2.loss import InfoNCELoss

if __name__ == '__main__':
    print("--- 1. Checking Data Availability ---")
    data_root = setup_data() 
    
    if data_root is None:
        print("CRITICAL ERROR: Data could not be set up. Exiting.")
        exit()

    base_dir = os.path.join(data_root, 'SPair-71k','Spair-71k') 
    pair_ann_path = os.path.join(base_dir, 'PairAnnotation')
    layout_path = os.path.join(base_dir, 'Layout')
    image_path = os.path.join(base_dir, 'JPEGImages')

    if not os.path.exists(pair_ann_path):
        print(f"Error: Paths look wrong. Checked inside: {base_dir}")
        print(f"Contents of data root: {os.listdir(data_root)}")
        exit()

    print("\n--- 2. Loading Test Dataset ---")

    train_dataset = SPairDataset(
        pair_ann_path, 
        layout_path, 
        image_path, 
        dataset_size='large',
        pck_alpha=0.5,
        datatype='trn'
    )
    
    trn_dataloader = DataLoader(train_dataset, batch_size=5, num_workers=0, shuffle=False)
    print(f"Train Set Loaded: {len(train_dataset)} images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 3. Loading DINOv2 on {device} ---")

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    criterion = InfoNCELoss(temperature=0.07).to(device)
  
    # Freeze Backbone
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze Last 2 Blocks
    for block in model.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # B. Setup Optimizer (High LR, No Decay)
    # We filter specifically for parameters that require grad
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=1e-3, weight_decay=0)
    
   
    batch = next(iter(trn_dataloader))

    src_img = batch['src_img'].to(device)
    trg_img = batch['trg_img'].to(device)
    src_kps = batch['src_kps'].to(device)
    trg_kps = batch['trg_kps'].to(device)
    mask    = batch['valid_mask'].to(device)

    epochs=100
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Extract features (Inside loop because params now have requires_grad=True)
        # DINOv2 returns dict: 'x_norm_patchtokens' are the 768-dim embeddings
        output_src = model.forward_features(src_img)
        output_trg = model.forward_features(trg_img)
        
        feat_src = output_src['x_norm_patchtokens'] # [B, 1369, 768]
        feat_trg = output_trg['x_norm_patchtokens']
        
        # Reshape to [B, C, H, W] for the Loss Class
        B, N, C = feat_src.shape
        H, W = 37, 37
        feat_src = feat_src.permute(0, 2, 1).reshape(B, C, H, W)
        feat_trg = feat_trg.permute(0, 2, 1).reshape(B, C, H, W)

        # Calculate Loss
        loss = criterion(feat_src, feat_trg, src_kps, trg_kps, mask)

        # Backpropagate
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    print("\n--- Overfit Check Complete ---")
    if loss.item() < 0.1:
        print("SUCCESS: The model successfully overfitted the sample. The pipeline is correct.")
    else:
        print("WARNING: Loss did not drop significantly. Check coordinate normalization or temperature.")
