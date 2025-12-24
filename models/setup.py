import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Funzione per configurare i layer da allenare
def configure_model(sam_model, unfreeze_last_n_layers):
    # Congela tutto inizialmente
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
    
    # Scongela solo gli ultimi N blocchi
    blocks = sam_model.image_encoder.blocks
    
    if unfreeze_last_n_layers > 0:
        blocks_to_train = blocks[-unfreeze_last_n_layers :]
        print(f"🔓 Scongelamento degli ultimi {len(blocks_to_train)} blocchi.")
        for block in blocks_to_train:
            for param in block.parameters():
                param.requires_grad = True

        # Scongela anche il neck (utile per l'adattamento finale)
        for param in sam_model.image_encoder.neck.parameters():
            param.requires_grad = True
    
    # Conta parametri
    trainable = sum(p.numel() for p in sam_model.image_encoder.parameters() if p.requires_grad)
    print(f"Parametri addestrabili: {trainable / 1e6:.2f} M")

# 2. La Loss Function 
class DenseCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, feats_src, feats_trg, kps_src, kps_trg, kps_mask):
        """
        feats: (B, 256, 64, 64)
        kps: (B, N, 2) in coordinate immagine originale
        kps_mask: (B, N) booleani (True se visibile)
        """
        device = feats_src.device
        B, C, Hf, Wf = feats_src.shape
        loss_total = torch.tensor(0.0, device=device, requires_grad=True)
        valid_samples = 0
        
        # Normalizziamo le feature map globalmente (importante per dot product)
        feats_src = F.normalize(feats_src, dim=1)
        feats_trg = F.normalize(feats_trg, dim=1)

        for b in range(B):
            # Filtriamo solo i keypoint validi usando la maschera
            mask = kps_mask[b]
            if mask.sum() == 0: continue

            valid_kps_src = kps_src[b][mask]
            valid_kps_trg = kps_trg[b][mask]

            # Flatten target map: (256, 64, 64) -> (256, 4096)
            trg_flat = feats_trg[b].view(C, -1) 

            # Per ogni keypoint valido
            for i in range(len(valid_kps_src)):
                # --- SORGENTE ---
                # Mappiamo coordinate immagine -> coordinate feature (64x64)
                # SAM scala 1024 -> 64, quindi fattore 16
                sx, sy = valid_kps_src[i]
                fx_src = int(sx / 16)
                fy_src = int(sy / 16)
                
                # Clamp per sicurezza (non uscire dalla mappa 64x64)
                fx_src = min(max(fx_src, 0), Wf - 1)
                fy_src = min(max(fy_src, 0), Hf - 1)

                # Estraiamo il vettore sorgente (query)
                query_vec = feats_src[b, :, fy_src, fx_src].unsqueeze(0) # (1, 256)

                # --- TARGET (Ground Truth) ---
                tx, ty = valid_kps_trg[i]
                fx_trg = int(tx / 16)
                fy_trg = int(ty / 16)
                fx_trg = min(max(fx_trg, 0), Wf - 1)
                fy_trg = min(max(fy_trg, 0), Hf - 1)

                # Calcoliamo l'indice piatto (0-4095) che rappresenta la risposta corretta
                target_idx = fy_trg * Wf + fx_trg
                target_tensor = torch.tensor([target_idx], device=device)

                # --- CALCOLO LOSS ---
                # Similarità con TUTTI i 4096 pixel target
                logits = torch.mm(query_vec, trg_flat) / self.temperature # (1, 4096)
                
                loss_total = loss_total + self.criterion(logits, target_tensor)
                valid_samples += 1
                
        if valid_samples > 0:
            loss_total = loss_total / valid_samples
        return loss_total

# 3. Funzione per il Layer-wise Learning Rate Decay (LLRD)
def get_grouped_params(model, base_lr, weight_decay, decay_factor=0.9):
    """
    Applica Layer-wise Learning Rate Decay.
    L'ultimo layer ha base_lr. Il penultimo ha base_lr * decay_factor, e così via.
    """
    param_groups = []
    
    # 1. Parametri del Neck (sempre LR alto, sono l'adattatore finale)
    param_groups.append({
        "params": [p for p in model.image_encoder.neck.parameters() if p.requires_grad],
        "lr": base_lr,
        "weight_decay": weight_decay
    })

    # 2. Blocchi del ViT (Encoder)
    blocks = model.image_encoder.blocks
    # Identifichiamo quali blocchi sono addestrabili
    trainable_block_indices = [i for i, b in enumerate(blocks) 
                               if any(p.requires_grad for p in b.parameters())]
    
    # Iteriamo sui blocchi in ordine inverso (dall'output verso l'input)
    current_lr = base_lr
    
    # Invertiamo per partire dall'ultimo blocco (LR più alto)
    for i in reversed(trainable_block_indices):
        block = blocks[i]
        param_groups.append({
            "params": [p for p in block.parameters() if p.requires_grad],
            "lr": current_lr,
            "weight_decay": weight_decay
        })
        print(f"⚙️ Blocco {i}: LR impostato a {current_lr:.2e}")
        
        # Riduciamo il LR per il blocco precedente (più profondo)
        current_lr *= decay_factor

    return param_groups
