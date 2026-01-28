import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import sys
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from segment_anything import sam_model_registry

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TASK_2_SAM.loss import DenseCrossEntropyLoss
from data.dataset_SAM import SPairDataset
from utils.common import download_sam_model
    
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

def fine_tuning_loRA(epochs, lr, w_decay):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_root = 'dataset/SPair-71k'
    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    checkpoint_dir = 'checkpoints'

    train_dataset = SPairDataset(
        pair_ann_path, layout_path, image_path,
        dataset_size='small', pck_alpha=0.1, datatype='trn')

    train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, 
        num_workers=4, persistent_workers=True, pin_memory=True)

    val_dataset = SPairDataset(
        pair_ann_path, layout_path, image_path,
        dataset_size='small', pck_alpha=0.1, datatype='val')

    val_dataloader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=4, persistent_workers=True, pin_memory=True)
    
    save_dir = 'checkpoints/LoRA_finetuned'
    ckpt_path = download_sam_model(checkpoint_dir)
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
   
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv"], 
        lora_dropout=0.1,
        bias="none",
    )
    
    model = get_peft_model(sam, lora_config) 
    model.to(device)                         
    model.print_trainable_parameters()      

    criterion = DenseCrossEntropyLoss(temperature=0.1).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=w_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=2e-4, 
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
        pct_start=0.1 
    )
    scaler = GradScaler()
    best_val_loss=float('inf')
    accumulation_steps = 2

    model.train()
    
    print(f"Inizio Fine-Tuning LoRA")
    for epoch in range(epochs):
        train_loss = 0
        steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [TRAIN]")       

        for i, batch in enumerate(pbar):
            src = batch['src_img'].to(device)
            trg = batch['trg_img'].to(device)
            kps_src = batch['src_kps'].to(device)
            kps_trg = batch['trg_kps'].to(device)
            kps_mask = batch['kps_valid'].to(device)
            
            with autocast():
                feats_src = model.image_encoder(src)
                feats_trg = model.image_encoder(trg)
                loss = criterion(feats_src, feats_trg, kps_src, kps_trg, kps_mask)
                loss = loss / accumulation_steps

            if loss.item() > 0:
                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() 
                    scheduler.step()

                current_loss = loss.item() * accumulation_steps
                train_loss += current_loss
                steps += 1
                pbar.set_postfix({'loss': train_loss / max(steps, 1)})
                
        avg_train_loss = train_loss / max(steps, 1)

        model.eval()
        val_loss = 0
        val_steps = 0
        torch.cuda.empty_cache()

        with torch.no_grad(): 
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [VAL]"):                
                src = batch['src_img'].to(device)
                trg = batch['trg_img'].to(device)
                kps_src = batch['src_kps'].to(device)
                kps_trg = batch['trg_kps'].to(device)
                kps_mask = batch['kps_valid'].to(device)

                with autocast():
                    feats_src = model.image_encoder(src)
                    feats_trg = model.image_encoder(trg)
                    loss = criterion(feats_src, feats_trg, kps_src, kps_trg, kps_mask)
                
                if loss.item() > 0:
                    val_loss += loss.item()
                    val_steps += 1
        
        avg_val_loss = val_loss / max(val_steps, 1)

        # --- STAMPA E SALVATAGGIO ---
        print(f"\n✅ EPOCA {epoch+1} COMPLETATA:")
        print(f"   📉 Training Loss:   {avg_train_loss:.4f}")
        print(f"   📊 Validation Loss: {avg_val_loss:.4f}")
        
        # Logica di salvataggio
        # 1. Salva il modello corrente come "latest"
        model.save_pretrained(os.path.join(save_dir, "lora_sam_latest"))        
        # 2. Salva SE è il migliore finora
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, f"lora_sam_best")
            model.save_pretrained(save_path)
            print(f" 🏆 Nuovo record! Adapter salvato in: {save_path}")
        
        print("-" * 60)
   
    
if __name__ == "__main__":
    seed_everything(42)
    fine_tuning_loRA(epochs=2, lr=2e-4, w_decay=1e-2)