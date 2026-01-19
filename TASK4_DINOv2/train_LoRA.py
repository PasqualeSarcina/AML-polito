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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.task2_DINOv2_dataset import SPairDataset
from utils.setup_data import setup_data
from task2_dinov2.Hyperparameters.loss import InfoNCELoss


data_root = setup_data() 
if data_root is None: 
    print("Error: Dataset not found. Please run utils/setup_data.py or check data location.")
    sys.exit(1)

base_dir = os.path.join(data_root, 'SPair-71k','Spair-71k') 
pair_ann_path = os.path.join(base_dir, 'PairAnnotation')
layout_path = os.path.join(base_dir, 'Layout')
image_path = os.path.join(base_dir, 'JPEGImages')

train_dataset = SPairDataset(
    pair_ann_path, layout_path, image_path, 
    dataset_size='large', pck_alpha=0.5, datatype='trn'
)

val_dataset = SPairDataset(
    pair_ann_path, layout_path, image_path,
    dataset_size="large", pck_alpha=0.5, datatype="val"
)

trn_dataloader = DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=True, 
    num_workers=4,           
    persistent_workers=True, 
    pin_memory=True          
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=4,           
    persistent_workers=True, 
    pin_memory=True          
)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> SEED SET TO {seed} <<<")

def fine_tuning(epochs, lr, w_decay):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    # 2. Configura LoRA
    lora_config = LoraConfig(
        r=16,                   # Rank: 8 o 16 sono i valori standard
        lora_alpha=32,          # Alpha: di solito il doppio del rank
        target_modules=["qkv"], # <--- DINOv2 usa un unico layer 'qkv' per l'attenzione
        lora_dropout=0.1,       # Aiuta a prevenire l'overfitting
        bias="none"
    )

    # 3. Applica LoRA al modello
    model = get_peft_model(model, lora_config)

    # 4. Sposta su GPU
    model = model.to(device)

    # 5. Stampa di controllo
    model.print_trainable_parameters()
    # Ti dirà che stai allenando circa l'1% dei parametri.
    # Ma quell'1% è distribuito in modo intelligente su tutto il modello!
    criterion = InfoNCELoss(temperature=0.07).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=lr, weight_decay=w_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler('cuda')
    num_epochs = epochs
    best_val_loss=float('inf')
    accumulation_steps = 8  # Simulate batch_size = 8
    
    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        pbar = tqdm(trn_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}[Training]")
        
        optimizer.zero_grad()  # Initialize gradients before the loop

        for i, batch in enumerate(pbar):
            
            # 2. Training Logic (INDENTED INSIDE THE LOOP)
            src_img = batch['src_img'].to(device)
            trg_img = batch['trg_img'].to(device)
            src_kps = batch['src_kps'].to(device)
            trg_kps = batch['trg_kps'].to(device)
            mask    = batch['valid_mask'].to(device)
            
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
                loss = loss / accumulation_steps  # Normalize loss for gradient accumulation

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trn_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})
    
        avg_train_loss = total_epoch_loss / len(trn_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Avg Training Loss: {avg_train_loss:.4f}")
        scheduler.step()
        
        # Optional: Print current LR to verify
        current_lr = scheduler.get_last_lr()[0]
        print(f"--> Learning Rate for next epoch: {current_lr:.8f}")
        
        model.eval()
        val_loss=0
        pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}[Validation]")
        with torch.no_grad():
            for i, batch in enumerate(pbar):
            
                # 2. Training Logic (INDENTED INSIDE THE LOOP)
                src_img = batch['src_img'].to(device)
                trg_img = batch['trg_img'].to(device)
                src_kps = batch['src_kps'].to(device)
                trg_kps = batch['trg_kps'].to(device)
                mask    = batch['valid_mask'].to(device)
                
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

                val_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
    
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Avg Validation Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss=avg_val_loss
                os.makedirs("checkpoints", exist_ok=True)
                # Save the weights
                model.save_pretrained("checkpoints/best_model")
                print(f"--> 🏆 New Best Model Saved! (Loss: {best_val_loss:.4f})")

if __name__ == '__main__':
    fine_tuning(epochs=5, lr=1e-4, w_decay=1e-2)
   


    
    
   

   