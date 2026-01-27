import os
import sys
import torch
from torch.optim import AdamW
from tqdm import tqdm
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TASK_2_SAM.loss import SoftSemanticLoss
from data.dataset import SPairDataset
from utils.common import download_sam_model
from TASK_4_SAM.setup_lora_sam import setup_lora_sam

def train_one_epoch(model, dataloader, optimizer, criterion, device, lr_scheduler):
    model.train()
    total_loss = 0
    accumulation_steps = 5
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training LoRA")

    for i, batch in pbar:
        optimizer.zero_grad()
        # Spostiamo tutto su device
        src_img = batch['src_img'].to(device)
        trg_img = batch['trg_img'].to(device)
        src_kps = batch['src_kps'].to(device)
        trg_kps = batch['trg_kps'].to(device)
        kps_mask = batch['kps_valid'].to(device)

        feats_src = model.image_encoder(src_img)
        feats_trg = model.image_encoder(trg_img)

        # Calcolo Loss
        loss = criterion(feats_src, feats_trg, src_kps, trg_kps, kps_mask)

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src_img, trg_img = batch['src_img'].to(device), batch['trg_img'].to(device)
            src_kps, trg_kps = batch['src_kps'].to(device), batch['trg_kps'].to(device)
            kps_mask = batch['kps_valid'].to(device)

            feats_src = model.image_encoder(src_img)
            feats_trg = model.image_encoder(trg_img)
            loss = criterion(feats_src, feats_trg, src_kps, trg_kps, kps_mask)
            total_val_loss += loss.item()
    return total_val_loss / len(dataloader)

def run_training_lora(sam_model, train_loader, val_loader, device, num_epochs=1):
    lora_path = os.path.join('checkpoints', 'lora_adapter_final')
    os.makedirs(os.path.dirname(lora_path), exist_ok=True)
    # 1. Setup
    optimizer = AdamW(sam_model.image_encoder.parameters(), lr=4e-4, weight_decay=0.2)
    criterion = SoftSemanticLoss(temperature=0.1)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / 250)
    )

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        avg_train_loss = train_one_epoch(sam_model, train_loader, optimizer, criterion, device, lr_scheduler)
        val_loss = validate(sam_model, val_loader, criterion, device)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        sam_model.image_encoder.save_pretrained(lora_path)
        print(f"Modello salvato su {lora_path}")

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_root = 'dataset/SPair-71k'
    checkpoint_dir = 'checkpoints'

    ckpt_path = download_sam_model(checkpoint_dir)
    sam_model_lora = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    
    sam_model_lora = setup_lora_sam(sam_model_lora)
    sam_model_lora.to(device)

    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    BATCH_SIZE = 2
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

    run_training_lora(sam_model_lora, train_dataloader, val_dataloader, device, num_epochs=1)
