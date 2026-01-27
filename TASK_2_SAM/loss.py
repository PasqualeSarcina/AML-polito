import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        # Usiamo reduction='none' per gestire manualmente la maschera dei keypoint
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feats_src, feats_trg, kps_src, kps_trg, kps_mask):
        B, C, Hf, Wf = feats_src.shape
        N_kps = kps_src.shape[1]
        STRIDE = 16

        # 1. Normalizzazione
        feats_src = F.normalize(feats_src, dim=1)
        feats_trg = F.normalize(feats_trg, dim=1)

        # 2. Prepariamo la griglia per grid_sample (B, N, 1, 2)
        # kps_src ha dimensione (B, 29, 2)
        grid_src = torch.zeros((B, N_kps, 1, 2), device=kps_src.device, dtype=feats_src.dtype)
        grid_src[..., 0] = (2 * (kps_src[..., 0] / (Wf * STRIDE - 1)) - 1).unsqueeze(-1)
        grid_src[..., 1] = (2 * (kps_src[..., 1] / (Hf * STRIDE - 1)) - 1).unsqueeze(-1)

        # 3. Estrazione feature sorgente in un colpo solo
        # query_feats: (B, C, 29, 1) -> (B, 29, C)
        query_feats = F.grid_sample(feats_src, grid_src, align_corners=True, mode='bilinear')
        query_feats = query_feats.squeeze(-1).permute(0, 2, 1)

        # 4. Calcolo Logits (B, 29, Hf*Wf)
        trg_flat = feats_trg.view(B, C, -1) # (B, C, 4096)
        logits = torch.bmm(query_feats, trg_flat) / self.temperature # (B, 29, 4096)

        # 5. Calcolo indici Target
        fx_trg = (kps_trg[..., 0] / STRIDE).round().long().clamp(0, Wf-1)
        fy_trg = (kps_trg[..., 1] / STRIDE).round().long().clamp(0, Hf-1)
        target_indices = fy_trg * Wf + fx_trg # (B, 29)

        # 6. Loss con Maschera
        # Appiattiamo per la CrossEntropy: (B*29, 4096) vs (B*29)
        loss = self.criterion(logits.view(-1, Hf*Wf), target_indices.view(-1))

        # Riapplichiamo la maschera per ignorare i punti paddati
        mask_flat = kps_mask.view(-1)
        masked_loss = loss * mask_flat.float()

        return masked_loss.sum() / mask_flat.sum().clamp(min=1)
    

class SoftSemanticLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feats_src, feats_trg, kps_src, kps_trg, kps_mask):
        B, C, Hf, Wf = feats_src.shape
        N_kps = kps_src.shape[1]
        STRIDE = 16 # 1024 / 64

        # 1. Normalizzazione Feature
        feats_src = F.normalize(feats_src, dim=1)
        feats_trg = F.normalize(feats_trg, dim=1)

        # 2. Estrazione Query Features (Sorgente)
        # Convertiamo kps in coordinate [-1, 1] per grid_sample
        grid_src = (kps_src.unsqueeze(2) / (1024 - 1)) * 2 - 1
        query_feats = F.grid_sample(feats_src, grid_src, align_corners=True, mode='bilinear')
        query_feats = query_feats.squeeze(-1).permute(0, 2, 1) # (B, N_kps, C)

        # 3. Calcolo Similarity Matrix
        trg_flat = feats_trg.view(B, C, -1) # (B, C, 4096)
        logits = torch.bmm(query_feats, trg_flat) / self.temperature # (B, N_kps, 4096)

        # 4. Generazione Soft Targets (Bilineari)
        # Troviamo la posizione continua sulla griglia delle feature (0-63)
        trg_kps_grid = kps_trg / STRIDE

        # Calcoliamo i 4 vicini e i pesi
        x0 = torch.floor(trg_kps_grid[..., 0]).long().clamp(0, Wf-1)
        x1 = (x0 + 1).clamp(0, Wf-1)
        y0 = torch.floor(trg_kps_grid[..., 1]).long().clamp(0, Hf-1)
        y1 = (y0 + 1).clamp(0, Hf-1)

        wa = (x1.float() - trg_kps_grid[..., 0]) * (y1.float() - trg_kps_grid[..., 1])
        wb = (x1.float() - trg_kps_grid[..., 0]) * (trg_kps_grid[..., 1] - y0.float())
        wc = (trg_kps_grid[..., 0] - x0.float()) * (y1.float() - trg_kps_grid[..., 1])
        wd = (trg_kps_grid[..., 0] - x0.float()) * (trg_kps_grid[..., 1] - y0.float())

        soft_targets = torch.zeros_like(logits)
        for b in range(B):
            for k in range(N_kps):
                if kps_mask[b, k]:
                    soft_targets[b, k, y0[b,k] * Wf + x0[b,k]] += wa[b,k]
                    soft_targets[b, k, y1[b,k] * Wf + x0[b,k]] += wb[b,k]
                    soft_targets[b, k, y0[b,k] * Wf + x1[b,k]] += wc[b,k]
                    soft_targets[b, k, y1[b,k] * Wf + x1[b,k]] += wd[b,k]

        # 5. Cross Entropy tra distribuzioni
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1) # (B, N_kps)

        return (loss * kps_mask.float()).sum() / kps_mask.sum().clamp(min=1)