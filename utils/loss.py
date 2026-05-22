import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for dense semantic correspondence.

    Works with both:
        DINOv2 ViT-B/14:
            input size 518x518
            patch size 14
            feature grid 37x37

        DINOv3 ViT-B/16:
            input size 512x512
            patch size 16
            feature grid 32x32

    Expected inputs:
        feat_src:   [B, C, H, W]
        feat_trg:   [B, C, H, W]
        kps_src:    [B, K, 2] in resized image coordinates
        kps_trg:    [B, K, 2] in resized image coordinates
        valid_mask: [B, K], 1 for valid keypoints, 0 for ignored keypoints
    """
    def __init__(self, temperature=0.07, patch_size=16):
        super().__init__()
        self.temperature = temperature
        self.patch_size = patch_size
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feat_src, feat_trg, kps_src, kps_trg, valid_mask):
    
        B, C, H, W = feat_src.shape
        _, K, _ = kps_src.shape

        src_norm = self._normalize_coords(kps_src, H=H, W=W, patch_size=self.patch_size) 

        desc_src = F.grid_sample(feat_src, src_norm, align_corners=True, mode='bilinear')

        desc_src = desc_src.squeeze(-1).permute(0, 2, 1) 
        desc_src = F.normalize(desc_src, dim=-1)

        feat_trg_flat = F.normalize(feat_trg.flatten(2), dim=1)

        logits = torch.bmm(desc_src, feat_trg_flat) / self.temperature
    
        trg_x_grid = torch.round((kps_trg[:, :, 0] + 0.5) / self.patch_size - 0.5).long()
        trg_y_grid = torch.round((kps_trg[:, :, 1] + 0.5) / self.patch_size - 0.5).long()

        trg_x_grid = trg_x_grid.clamp(0, W - 1)
        trg_y_grid = trg_y_grid.clamp(0, H - 1)

        target_classes = (trg_y_grid * W) + trg_x_grid

        loss = self.ce_loss(logits.flatten(0, 1), target_classes.flatten())

        loss = loss.view(B, K)

        final_loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        return final_loss

    def _normalize_coords(self, kps, H, W, patch_size):
        """
        Convert resized image-space coordinates to grid_sample coordinates
        on the ViT patch-token grid.

        kps: [B, K, 2] in resized image coordinates.
        feature grid: H x W, e.g. 32 x 32 for DINOv3 ViT-B/16.
        """

        x_grid = (kps[:, :, 0] + 0.5) / patch_size - 0.5
        y_grid = (kps[:, :, 1] + 0.5) / patch_size - 0.5

        x_grid = x_grid.clamp(0, W - 1)
        y_grid = y_grid.clamp(0, H - 1)

        x_norm = 2.0 * x_grid / (W - 1) - 1.0
        y_norm = 2.0 * y_grid / (H - 1) - 1.0

        grid = torch.stack([x_norm, y_norm], dim=-1)

        return grid.unsqueeze(2)