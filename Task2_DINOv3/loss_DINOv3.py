import torch
import torch.nn as nn
import torch.nn.functional as F

def _pad_kps_list(kps_list, pad_value=-1.0):
    """
    kps_list: list of tensors [Ki,2] (variable Ki across batch)
    Returns:
      kps_pad: [B,Kmax,2]
      is_real: [B,Kmax] bool
    """
    B = len(kps_list)
    if B == 0:
        raise ValueError("Empty batch for keypoints.")
    Kmax = max(k.shape[0] for k in kps_list)
    device = kps_list[0].device
    dtype  = kps_list[0].dtype

    kps_pad = torch.full((B, Kmax, 2), pad_value, device=device, dtype=dtype)
    is_real = torch.zeros((B, Kmax), device=device, dtype=torch.bool)
    for i, kps in enumerate(kps_list):
        Ki = kps.shape[0]
        kps_pad[i, :Ki] = kps
        is_real[i, :Ki] = True
    return kps_pad, is_real


class InfoNCEPatchClassifyLoss(nn.Module):
    """
    For each source keypoint, classify which target feature cell (patch) is correct.

    Expected by your train loop:
      loss, nkps = loss_fn(src_feats, trg_feats, src_kps_list, trg_kps_list,
                           src_meta_list, trg_meta_list, return_kps=True)
    """
    def __init__(
        self,
        *,
        out_size=512,
        patch_size=16,
        temperature=0.07,
        align_corners=True,
        kp_to_cell="round",   # "round" or "floor"
    ):
        super().__init__()
        self.out_size = int(out_size)
        self.patch_size = int(patch_size)
        self.temperature = float(temperature)
        self.align_corners = bool(align_corners)
        assert kp_to_cell in ("round", "floor")
        self.kp_to_cell = kp_to_cell
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feat_src, feat_trg, src_kps_list, trg_kps_list, src_meta_list=None, trg_meta_list=None, *, return_kps=False):
        B, C, Hf, Wf = feat_src.shape
        device = feat_src.device
        dtype  = feat_src.dtype
        src_kps_list = [k.to(device=device, dtype=dtype) for k in src_kps_list]
        trg_kps_list = [k.to(device=device, dtype=dtype) for k in trg_kps_list]
        if feat_trg.shape != (B, C, Hf, Wf):
            raise ValueError(f"feat_trg shape {feat_trg.shape} != feat_src shape {feat_src.shape}")

        # pad variable-length keypoints
        src_kps, src_real = _pad_kps_list(src_kps_list, pad_value=-1.0)  # [B,K,2]
        trg_kps, trg_real = _pad_kps_list(trg_kps_list, pad_value=-1.0)
        Kmax = src_kps.shape[1]

        if Kmax == 0:
            loss = feat_src.sum() * 0.0
            return (loss, 0) if return_kps else loss

        # validity: exists + nonnegative + in bounds
        def in_bounds(kps):
            x = kps[..., 0]
            y = kps[..., 1]
            return (x >= 0) & (y >= 0) & (x <= self.out_size - 1) & (y <= self.out_size - 1)

        valid = src_real & trg_real & in_bounds(src_kps) & in_bounds(trg_kps)  # [B,K] bool
        nkps = int(valid.sum().item())
        if nkps == 0:
            loss = feat_src.sum() * 0.0
            return (loss, 0) if return_kps else loss

        # mapping padded pixels -> feature grid coords (assumes linear mapping)
        sx = (Wf - 1) / (self.out_size - 1)
        sy = (Hf - 1) / (self.out_size - 1)

        # ---- sample source descriptors with grid_sample
        src_xf = src_kps[..., 0] * sx
        src_yf = src_kps[..., 1] * sy

        if self.align_corners:
            gx = (src_xf / (Wf - 1)) * 2 - 1
            gy = (src_yf / (Hf - 1)) * 2 - 1
        else:
            gx = (src_xf + 0.5) / Wf * 2 - 1
            gy = (src_yf + 0.5) / Hf * 2 - 1

        grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [B,K,1,2]

        desc_src = F.grid_sample(
            feat_src, grid, mode="bilinear", align_corners=self.align_corners
        )  # [B,C,K,1]
        desc_src = desc_src.squeeze(-1).permute(0, 2, 1)  # [B,K,C]
        desc_src = F.normalize(desc_src, dim=-1)

        # ---- flatten target features (classes)
        feat_trg_flat = F.normalize(feat_trg.flatten(2), dim=1)  # [B,C,N], N=Hf*Wf

        # logits: [B,K,N]
        logits = torch.bmm(desc_src, feat_trg_flat) / self.temperature

        # ---- hard labels for each kp (target grid index)
        trg_xf = trg_kps[..., 0] * sx
        trg_yf = trg_kps[..., 1] * sy
        if self.kp_to_cell == "round":
            trg_xi = trg_xf.round().long()
            trg_yi = trg_yf.round().long()
        else:
            trg_xi = trg_xf.floor().long()
            trg_yi = trg_yf.floor().long()

        trg_xi = trg_xi.clamp(0, Wf - 1)
        trg_yi = trg_yi.clamp(0, Hf - 1)
        target_cls = (trg_yi * Wf + trg_xi)  # [B,K] in [0..Hf*Wf-1]

        # ---- CE, masked
        loss_per = self.ce(logits.reshape(B * Kmax, -1), target_cls.reshape(B * Kmax))
        loss_per = loss_per.view(B, Kmax)

        vf = valid.to(loss_per.dtype)
        loss = (loss_per * vf).sum() / (vf.sum() + 1e-6)

        return (loss, nkps) if return_kps else loss