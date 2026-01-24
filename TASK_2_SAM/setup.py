import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Funzione per configurare i layer da allenare
def configure_model(model, unfreeze_last_n_layers):
    # Congela tutto inizialmente
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    
    # Scongela anche il neck (utile per l'adattamento finale)
    for param in model.image_encoder.neck.parameters():
        param.requires_grad = True

    # Scongela solo gli ultimi N blocchi
    blocks_to_train = model.image_encoder.blocks[-unfreeze_last_n_layers :]

    print(f"🔓 Scongelamento degli ultimi {len(blocks_to_train)} blocchi.")
    for block in blocks_to_train:
        for param in block.parameters():
            param.requires_grad = True
    
    # Conta parametri
    trainable_params = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.image_encoder.parameters())
    print(f"--- Configurazione Fine-Tuning ({unfreeze_last_n_layers} blocchi) ---")
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri allenabili: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")

# 2. La Loss Function 
class MyCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.stride = 16
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)

    def forward(self, feat_src, feat_trg, kps_src, kps_trg, valid_mask):
        B, C, H, W = feat_src.shape
        _, K, _ = kps_src.shape

        # 1. Normalizzazione coordinate Sorgente per grid_sample [-1, 1]
        img_h, img_w = H * self.stride, W * self.stride
        
        # Grid sample vuole (x, y) normalizzati
        kps_src_norm = kps_src.clone()
        kps_src_norm[..., 0] = 2 * (kps_src[..., 0] / (img_w - 1)) - 1
        kps_src_norm[..., 1] = 2 * (kps_src[..., 1] / (img_h - 1)) - 1
        kps_src_norm = kps_src_norm.unsqueeze(2) # [B, K, 1, 2]

        # 2. Estrazione Descrittori Sorgente (Vettorizzata)
        # [B, C, H, W] -> sample -> [B, C, K, 1] -> [B, K, C]
        desc_src = F.grid_sample(feat_src, kps_src_norm, align_corners=True, mode='bilinear')
        desc_src = desc_src.squeeze(3).permute(0, 2, 1) 
        desc_src = F.normalize(desc_src, dim=-1) # L2 Norm cruciale

        # 3. Preparazione Target (Flattening)
        feat_trg_flat = feat_trg.flatten(2) # [B, C, H*W]
        feat_trg_flat = F.normalize(feat_trg_flat, dim=1)

        # 4. Calcolo Similarità (Batch Matrix Mult - Veloce!)
        # [B, K, C] @ [B, C, HW] -> [B, K, HW]
        logits = torch.bmm(desc_src, feat_trg_flat) / self.temperature

        # 5. Calcolo Label Target (Indici piatti)
        # Usiamo lo stride corretto passato nell'init
        fx_trg = (kps_trg[..., 0] / self.stride).round().long().clamp(0, W - 1)
        fy_trg = (kps_trg[..., 1] / self.stride).round().long().clamp(0, H - 1)
        target_indices = (fy_trg * W) + fx_trg # [B, K]

        # 6. Calcolo Loss e Mascheramento
        loss = self.criterion(logits.flatten(0, 1), target_indices.flatten()) # [B*K]
        loss = loss.view(B, K)
        
        # Media pesata solo sui keypoint validi
        final_loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        return final_loss