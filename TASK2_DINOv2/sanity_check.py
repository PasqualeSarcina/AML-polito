import torch
import os
import sys
from torch.utils.data import DataLoader

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.task2_DINOv2_dataset import SPairDataset
from utils.setup_data import setup_data
from TASK2_DINOv2.loss import InfoNCELoss  

def check_initial_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Initial Loss Check using loss.py on {device} ---")
    
    # 1. Load Data
    data_root = setup_data()
    base_dir = os.path.join(data_root, 'SPair-71k', 'Spair-71k')
    dataset = SPairDataset(
        os.path.join(base_dir, 'PairAnnotation'),
        os.path.join(base_dir, 'Layout'),
        os.path.join(base_dir, 'JPEGImages'),
        dataset_size='large', 
        pck_alpha=0.5,
        datatype='trn'
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    # 2. Load Model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    model.eval()

    # 3. Initialize YOUR Loss Function
    criterion = InfoNCELoss(temperature=0.07).to(device)

    # 4. Prepare Batch
    src_img = batch['src_img'].to(device)
    trg_img = batch['trg_img'].to(device)
    src_kps = batch['src_kps'].to(device)
    trg_kps = batch['trg_kps'].to(device)
    mask    = batch['valid_mask'].to(device)

    with torch.no_grad():
        # Loss expects [B, 768, 37, 37]
        feat_src = model.forward_features(src_img)['x_norm_patchtokens'] # [4, 1369, 768]
        feat_trg = model.forward_features(trg_img)['x_norm_patchtokens']
        B, N, C = feat_src.shape
        H, W = 37, 37
        feat_src = feat_src.permute(0, 2, 1).reshape(B, C, H, W)
        feat_trg = feat_trg.permute(0, 2, 1).reshape(B, C, H, W)

        # 5. Calculate Loss using your class
        loss = criterion(feat_src, feat_trg, src_kps, trg_kps, mask)

    print("-" * 30)
    print(f"✅ Initial Loss (from loss.py): {loss.item():.4f}")
    print("-" * 30)

if __name__ == '__main__':
    check_initial_loss()