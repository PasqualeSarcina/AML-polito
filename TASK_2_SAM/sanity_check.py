import torch
import os
import sys
from torch.utils.data import DataLoader
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SPairDataset
from segment_anything import sam_model_registry
from utils.common import download_sam_model
from TASK_2_SAM.setup import MyCrossEntropyLoss

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

def check_initial_loss():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Initial Loss Check using loss.py on {device} ---")
    
    # 1. Load Data
    dataset_root = 'dataset/SPair-71k'
    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='trn')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))

    # 2. Load Model
    checkpoint_dir = 'checkpoints'
    ckpt_path = download_sam_model(checkpoint_dir)
    model = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    model.to(device)
    model.eval()

    # 3. Initialize YOUR Loss Function
    criterion = MyCrossEntropyLoss(temperature=1.0).to(device)

    # 4. Prepare Batch
    src = batch['src_img'].to(device)
    trg = batch['trg_img'].to(device)
    kps_src = batch['src_kps'].to(device)
    kps_trg = batch['trg_kps'].to(device)
    kps_mask = batch['kps_valid'].to(device)

    H_feat, W_feat = 64, 64 
    num_classes = H_feat * W_feat
    expected_loss = np.log(num_classes)

    print(f"Dimensione attesa feature map: {H_feat}x{W_feat}")
    print(f"Numero classi (patches): {num_classes}")
    print(f"Loss Teorica (Target): {expected_loss:.4f}")
    
    with torch.no_grad():
        feat_src = model.image_encoder(src) #output atteso: [1,256,64,64]
        feat_trg = model.image_encoder(trg)

        # 5. Calculate Loss using your class
        loss = criterion(feat_src, feat_trg, kps_src, kps_trg, kps_mask)

    print("-" * 30)
    print(f"✅ Initial Loss: {loss.item():.4f}")
    
    diff = abs(loss.item() - expected_loss)
    print(f"Differenza: {diff:.4f}")

if __name__ == '__main__':
    check_initial_loss()