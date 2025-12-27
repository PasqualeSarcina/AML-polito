
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from segment_anything import sam_model_registry

from models.setup import DenseCrossEntropyLoss, configure_model, get_grouped_params
from utils.geometry import extract_features
from utils.common import download_sam_model
from data.dataset import SPairDataset

# ==========================================
# TRAINING LOOP
# ==========================================
def train_finetune(model, train_loader, val_loader, save_dir, n_layers, epochs=10, lr=1e-5, accumulation_steps=4):
    scaler = GradScaler()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = DenseCrossEntropyLoss(temperature=0.1)
    model.train() 
    
    best_val_loss = float('inf') #TENIAMO TRACCIA DEL MIGLIOR MODELLO

    print(f"🚀 Inizio Training per {epochs} epoche...")
    
    for epoch in range(epochs):
        train_loss = 0
        steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")       

        for i, batch in enumerate(pbar):
            try:
                # Sposta su GPU
                src = batch['src_img'].to(device)
                trg = batch['trg_img'].to(device)
                kps_src = batch['src_kps'].to(device)
                kps_trg = batch['trg_kps'].to(device)
                kps_mask = batch['kps_valid'].to(device)
                
                # Forward usando extract_features (con gradienti!)
                with autocast():
                    feats_src = extract_features(model, src)
                    feats_trg = extract_features(model, trg)
                    loss = criterion(feats_src, feats_trg, kps_src, kps_trg, kps_mask)
                    loss = loss / accumulation_steps

                if loss.item() > 0:
                    scaler.scale(loss).backward() #accumula il gradiente (non azzeriamo ancora)

                    # Facciamo lo step solo ogni N passaggi
                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad() #solo a questo punto azzeriamo i gradienti

                    # Moltiplichiamo per accumulation_steps solo per la stampa a video (per vedere la loss reale)
                    current_loss = loss.item() * accumulation_steps
                    train_loss += current_loss
                    steps += 1
                    pbar.set_postfix({'loss': train_loss / max(steps, 1)})
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ OOM/CUDA Error. Salto batch e svuoto cache.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e # Se è un altro errore, fermati
        
        avg_train_loss = train_loss / max(steps, 1)

        model.eval()
        val_loss = 0
        val_steps = 0
        torch.cuda.empty_cache()

        with torch.no_grad(): # Niente gradienti qui, risparmia memoria
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

        # --- STAMPA E SALVATAGGIO ---
        print(f"\n✅ EPOCA {epoch+1} COMPLETATA:")
        print(f"   📉 Training Loss:   {avg_train_loss:.4f}")
        print(f"   📊 Validation Loss: {avg_val_loss:.4f}")
        
        # Logica di salvataggio
        # 1. Salva il modello corrente (sovrascrive 'latest')
        torch.save(model.state_dict(), os.path.join(save_dir, "sam_latest.pth"))
        
        # 2. Salva SE è il migliore finora
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"{n_layers}layer_sam_BEST_{epoch}epochs.pth"))
            print("   🏆 Trovato nuovo miglior modello! Salvato.")
        
        print("-" * 60)

if __name__ == "__main__":
    # SETUP PATHS
    dataset_root = 'dataset/SPair-71k' # Aggiusta se necessario
    checkpoint_dir = 'checkpoints'

    # CONFIGURA DEVICE
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # DOWNLOAD & LOAD MODEL
    ckpt_path = download_sam_model(checkpoint_dir)
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    sam.to(device)

    # CONFIGURA MODELLO PER FINETUNING
    n_layers = 1
    n_epochs = 10  
    configure_model(sam, unfreeze_last_n_layers=n_layers)

    # DATASET E DATALOADER
    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    train_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='trn')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    print(f"Dataset Training caricato: {len(train_dataset)} coppie.")

    val_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"Dataset Validation caricato: {len(val_dataset)} coppie.")

    # AVVIO TRAINING
    print(f"\n\n{'#'*60}")
    print(f"🧪 ESPERIMENTO: Fine-tuning ultimi {n_layers} layer")
    print(f"{'#'*60}")
    train_finetune(sam, train_dataloader, val_dataloader, checkpoint_dir, n_layers, n_epochs, lr=1e-5, accumulation_steps=4)
