import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm # Per la barra di caricamento
import gc
from collections import defaultdict
from data.dataset import SPairDataset
from utils.geometry import extract_features, compute_correspondence
from segment_anything import sam_model_registry
from utils.common import download_sam_model
from peft import LoraConfig, get_peft_model

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_root = 'dataset/SPair-71k'
    checkpoint_dir = 'checkpoints'
    model_name = 'sam_tuned_1layer.pth' # Il modello da testare

    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    test_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    #LOAD MODEL
    base_ckpt = download_sam_model(checkpoint_dir)
    sam = sam_model_registry["vit_b"](checkpoint=base_ckpt)

    # Load LoRA model
    lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["qkv"], 
            lora_dropout=0.1, # serve per la config
            bias="none"
        )
    model = get_peft_model(sam, lora_config)

    # Caricamento pesi custom se esistono
    tuned_path = os.path.join(checkpoint_dir, model_name)
    if os.path.exists(tuned_path):
        print(f"Caricamento pesi custom: {tuned_path}")
        sam.load_state_dict(torch.load(tuned_path, map_location=device), strict=True)
    else:
        print(f"❌ ERRORE: Il file {tuned_path} non esiste.")

    sam.to(device)
    evaluate_pck(sam, test_dataloader, device)