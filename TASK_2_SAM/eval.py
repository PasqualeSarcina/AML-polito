import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_SAM import SPairDataset
from segment_anything import sam_model_registry
from utils.common import download_sam_model
from TASK_1_SAM.eval import evaluate_pck

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_root = 'dataset/SPair-71k'
    checkpoint_dir = 'checkpoints'
    model_name = 'sam_best.pth' # Il modello da testare

    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    test_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    base_ckpt = download_sam_model(checkpoint_dir)
    sam = sam_model_registry["vit_b"](checkpoint=base_ckpt)

    # Caricamento pesi custom se esistono
    tuned_path = os.path.join(checkpoint_dir, model_name)
    if os.path.exists(tuned_path):
        print(f"Caricamento pesi custom: {tuned_path}")
        sam.image_encoder.load_state_dict(torch.load(tuned_path, map_location=device), strict=True)
    else:
        print(f"ERRORE: Il file {tuned_path} non esiste.")

    sam.to(device)
    
    evaluate_pck(sam, test_dataloader, device)
    
