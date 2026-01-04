import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from dataset.Task2_DINOv2_dataset import SPairDataset

# --- 1. Define the Loss Function (InfoNCE) ---
def info_nce_loss(feat_A, feat_B, temperature=0.1):
    """
    Calculates the loss assuming the correct match for patch[i] in A 
    is patch[i] in B (The Diagonal).
    NOTE: For SPair-71k (different instances), this assumption is weak,
    but it works for checking if the code pipeline runs without crashing.
    """
    # Normalize features (Crucial for Dot Product Similarity)
    feat_A = F.normalize(feat_A, dim=-1)
    feat_B = F.normalize(feat_B, dim=-1)
    
    # Similarity Matrix: [Batch, Tokens, Tokens]
    sim_matrix = torch.bmm(feat_A, feat_B.transpose(1, 2)) / temperature
    
    # Labels: We assume Identity matching for the check (0->0, 1->1...)
    B, T, _ = sim_matrix.shape
    labels = torch.arange(T).to(feat_A.device).expand(B, T)
    
    # Calculate Cross Entropy
    loss = F.cross_entropy(sim_matrix.flatten(0, 1), labels.flatten())
    return loss

# --- 2. The Sanity Check Execution ---
if __name__ == '__main__':
    # A. CONFIGURATION
    base_dir = r"C:\Users\nicol\Documents\PoliTo\AdvancedML\project\SPair-71k_extracted\SPair-71k\SPair-71k"
    pair_ann_path = os.path.join(base_dir, 'PairAnnotation')
    layout_path = os.path.join(base_dir, 'Layout')
    image_path = os.path.join(base_dir, 'JPEGImages')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Step 1: InfoNCE Sanity Check on {device} ---")

    # B. LOAD DATASET (Using the correct file!)
    if os.path.exists(base_dir):
        print("1. Loading Dataset...")
        # We use batch_size=2 just to check if it runs
        trn_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.05, datatype='trn')
        trn_loader = DataLoader(trn_dataset, batch_size=4, shuffle=True)
    else:
        print(f"❌ Error: Path not found at {base_dir}")
        exit()

    # C. LOAD MODEL
    print("2. Loading DINOv2 Backbone...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    backbone.eval() # Freeze for check
    
    # D. GET ONE BATCH & RUN
    print("3. Running Forward Pass...")
    try:
        batch = next(iter(trn_loader))
        
        src_img = batch['src_img'].to(device)
        trg_img = batch['trg_img'].to(device)
        
        with torch.no_grad():
            # Forward Pass
            dict_A = backbone.forward_features(src_img)
            dict_B = backbone.forward_features(trg_img)
            
            # Extract Patch Tokens
            feat_A = dict_A['x_norm_patchtokens']
            feat_B = dict_B['x_norm_patchtokens']
            
            # Calculate Loss
            loss = info_nce_loss(feat_A, feat_B)
            
        print(f"\n✅ Calculated InfoNCE Loss: {loss.item():.4f}")
        
        # --- INTERPRETATION ---
        print("-" * 30)
        # 1369 patches -> ln(1369) ≈ 7.22
        expected_loss = 7.22 
        
        if 6.0 < loss.item() < 8.5:
            print(f"✅ PASSED. Loss is close to random guess ({expected_loss:.2f}).")
            print("   This is expected because Image A and Image B are different objects.")
        elif loss.item() > 9.0:
            print(f"⚠️ HIGH. Loss is {loss.item():.2f}.")
        else:
            print(f"❌ LOW. Loss is {loss.item():.2f}. (Unexpectedly good match?)")
            
    except Exception as e:
        print(f"❌ CRASHED: {e}")