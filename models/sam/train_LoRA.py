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

from utils.loss import InfoNCELoss
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

def setup_lora_sam(model, r=16):
    for param in model.parameters():
        param.requires_grad = False

    config = LoraConfig(
        r=r,
        lora_alpha=2*r,
        target_modules=["qkv"], 
        lora_dropout=0.1,
        bias="none"
    )

    model.image_encoder = get_peft_model(model.image_encoder, config)
    model.image_encoder.print_trainable_parameters()
    return model


def lora_training(model, train_loader, val_loader, device, epochs, lr, accumulation_steps):
    lora_path = 'checkpoints/sam_lora'
    os.makedirs(os.path.dirname(lora_path), exist_ok=True)
    scaler = GradScaler()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = InfoNCELoss(temperature=0.07).to(device)
    best_val_loss = float('inf')

    print(f"Inizio Training per {epochs} epoche")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")

        for i, batch in enumerate(pbar):
            src = batch['src_img'].to(device)
            trg = batch['trg_img'].to(device)
            kps_src = batch['src_kps'].to(device)
            kps_trg = batch['trg_kps'].to(device)
            kps_mask = batch['kps_valid'].to(device)

            with autocast():
                feats_src = model.image_encoder(src) # B, C, Hf, Wf
                feats_trg = model.image_encoder(trg)
                loss = criterion(feats_src, feats_trg, kps_src, kps_trg, kps_mask)
                loss = loss / accumulation_steps

            if loss.item() > 0:
                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0 or (i+1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                current_loss = loss.item() * accumulation_steps
                train_loss += current_loss
                steps += 1
                pbar.set_postfix({'loss': train_loss / max(steps, 1)})

        avg_train_loss = train_loss / max(steps, 1)
        scheduler.step()

        model.eval()
        val_loss = 0
        val_steps = 0
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]"):
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

        print(f"\nEPOCA {epoch+1} COMPLETATA:")
        print(f"Training Loss:   {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Logica di salvataggio
        # 1. Salva il modello corrente
        epoch_save_path = os.path.join(lora_path, f'epoch_{epoch+1}')
        model.image_encoder.save_pretrained(epoch_save_path)

        # 2. Salva SE è il migliore finora
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_save_path = os.path.join(lora_path, 'best_model')
            model.image_encoder.save_pretrained(best_save_path)
            print(f"Salvato: {best_save_path}")

        print("-" * 60)
    
if __name__ == "__main__":
    seed_everything(42)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dataset_root = 'dataset/SPair-71k'
    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    checkpoint_dir = 'checkpoints'

    BATCH_SIZE = 1
    NUM_WORKERS = 4
    train_dataset = SPairDataset(
        pair_ann_path, layout_path, image_path,
        dataset_size='large', pck_alpha=0.1, datatype='trn')

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    val_dataset = SPairDataset(
        pair_ann_path, layout_path, image_path,
        dataset_size='large', pck_alpha=0.1, datatype='val')

    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    
    ckpt_path = download_sam_model(checkpoint_dir)
    sam_model_lora = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    sam_model_lora = setup_lora_sam(sam_model_lora)
    sam_model_lora.to(device)

    lora_training(sam_model_lora, train_dataloader, val_dataloader, device,
               epochs=5, lr=1e-5, accumulation_steps=8)