import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feat_src, feat_trg, kps_src, kps_trg, valid_mask):
        """
        feat_src, feat_trg: [B, C, H, W] (Feature Maps, e.g. 32x32)
        kps_src, kps_trg:   [B, K, 2]    (Keypoint coordinates in pixels)
        valid_mask:         [B, K]       (1=Valid, 0=Invalid)
        """
        B, C, H, W = feat_src.shape
        _, K, _ = kps_src.shape
        patch_size = 16


        # --- 1. Extract Source Descriptors ---
        src_norm = self._normalize_coords(kps_src, H, W, patch_size) # [B, K, 1, 2]

        desc_src = F.grid_sample(feat_src, src_norm, align_corners=True, mode='bilinear')
        desc_src = desc_src.squeeze(-1).permute(0, 2, 1) # [B, K, C]

        # Normalize vectors (Important for Cosine Sim!)
        desc_src = F.normalize(desc_src, dim=-1)
        feat_trg_flat = F.normalize(feat_trg.flatten(2), dim=1) # [B, C, N_patches]

        # --- 2. Calculate Similarity Heatmap ---
        logits = torch.bmm(desc_src, feat_trg_flat) / self.temperature

        # --- 3. Create Ground Truth Labels ---
        trg_x_grid = (kps_trg[:, :, 0].floor() / patch_size).long()   # oppure kps_trg.long() // patch_size
        trg_y_grid = (kps_trg[:, :, 1].floor() / patch_size).long()

        # Clamp to avoid crashing if point is slightly outside
        trg_x_grid = trg_x_grid.clamp(0, W - 1)
        trg_y_grid = trg_y_grid.clamp(0, H - 1)

        target_classes = (trg_y_grid * W) + trg_x_grid # [B, K]

        # --- 4. Calculate Loss ---
        # Reshape to treat every keypoint as a classification example
        loss = self.ce_loss(logits.flatten(0, 1), target_classes.flatten())

        # Reshape back to [B, K] to apply the mask
        loss = loss.view(B, K)

        # Mask out invalid keypoints (padding or invisible)
        # We only learn from points that exist in both images
        final_loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        return final_loss

    def _normalize_coords(self, kps, H, W, patch_size):
    # kps: [B,K,2] in pixel (0..511)
    # convert pixel -> patch coords (float)
        x = kps[:, :, 0] / patch_size  # [0..W)
        y = kps[:, :, 1] / patch_size  # [0..H)

        # normalize to [-1, 1] for grid_sample over feature map size W,H
        x = (x / (W - 1)) * 2 - 1
        y = (y / (H - 1)) * 2 - 1

        grid = torch.stack([x, y], dim=-1)  # [B,K,2] with order (x,y)
        return grid.unsqueeze(2)            # [B,K,1,2]
        