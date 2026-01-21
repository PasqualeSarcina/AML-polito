import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from segment_anything import sam_model_registry

from models.setup import configure_model, DenseCrossEntropyLoss
from data.dataset import SPairDataset
from utils.common import download_sam_model

dataset_root = 'dataset/SPair-71k'
pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
layout_path = os.path.join(dataset_root, 'Layout')
image_path = os.path.join(dataset_root, 'JPEGImages')
checkpoint_dir = 'checkpoints'

train_dataset = SPairDataset(
    pair_ann_path, layout_path, image_path,
    dataset_size='large', pck_alpha=0.1, datatype='trn')

train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, 
    num_workers=4, persistent_workers=True, pin_memory=True)

print(f"Dataset Training caricato: {len(train_dataset)} coppie.")

val_dataset = SPairDataset(
    pair_ann_path, layout_path, image_path,
    dataset_size='large', pck_alpha=0.1, datatype='val')

val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=4, persistent_workers=True, pin_memory=True)
    
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

def fine_tuning_loRA(num_epochs, lr, w_decay):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = download_sam_model(checkpoint_dir)
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
   
    
    # Configurazione LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(sam, lora_config) #applica LoRA al modello SAM
    model.to(device)                         #sposta il modello sul device
    model.print_trainable_parameters()       #stampa i parametri addestrabili

    # Funzione di perdita e ottimizzatore
    criterion = DenseCrossEntropyLoss(temperature=0.1).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=w_decay)
    #scheduler
    scaler = torch.cuda.amp.GradScaler('cuda')
    best_val_loss=float('inf')
    accumulation_steps = 8

    model.train()
    
    print(f"🚀 Inizio Fine-Tuning LoRA")
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}[Training]")
        optimizer.zero_grad()
        for i, batch in enumerate(pbar):
            src_img = batch['src_img'].to(device)
            tgt_img = batch['tgt_img'].to(device)
            src_kps = batch['src_kps'].to(device)
            tgt_kps = batch['tgt_kps'].to(device)

            with torch.cuda.amp.autocast():
                src_feats = model.encode_image(src_img)
                tgt_feats = model.encode_image(tgt_img)

                loss = criterion(outputs, tgt_kps)
                loss = loss / accumulation_steps
    
if __name__ == "__main__":
    seed_everything(42)
    fine_tuning_loRA(num_epochs=5, lr=1e-4, w_decay=1e-2)