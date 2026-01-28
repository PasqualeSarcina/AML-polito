import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feats_src, feats_trg, kps_src, kps_trg, kps_mask):
        B, C, Hf, Wf = feats_src.shape
        N_kps = kps_src.shape[1]
        STRIDE = 16

        feats_src = F.normalize(feats_src, dim=1)
        feats_trg = F.normalize(feats_trg, dim=1)

        grid_src = torch.zeros((B, N_kps, 1, 2), device=kps_src.device, dtype=feats_src.dtype)
        grid_src[..., 0] = (2 * (kps_src[..., 0] / (Wf * STRIDE - 1)) - 1).unsqueeze(-1)
        grid_src[..., 1] = (2 * (kps_src[..., 1] / (Hf * STRIDE - 1)) - 1).unsqueeze(-1)

        query_feats = F.grid_sample(feats_src, grid_src, align_corners=True, mode='bilinear')
        query_feats = query_feats.squeeze(-1).permute(0, 2, 1)

        trg_flat = feats_trg.view(B, C, -1) # (B, C, 4096)
        logits = torch.bmm(query_feats, trg_flat) / self.temperature # (B, 29, 4096)

        fx_trg = (kps_trg[..., 0] / STRIDE).round().long().clamp(0, Wf-1)
        fy_trg = (kps_trg[..., 1] / STRIDE).round().long().clamp(0, Hf-1)
        target_indices = fy_trg * Wf + fx_trg # (B, 29)

        loss = self.criterion(logits.view(-1, Hf*Wf), target_indices.view(-1))

        mask_flat = kps_mask.view(-1)
        masked_loss = loss * mask_flat.float()

        return masked_loss.sum() / mask_flat.sum().clamp(min=1)
