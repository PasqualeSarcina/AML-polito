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
    def __init__(self, temperature=0.01):
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
            