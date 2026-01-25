import torch
import torch.nn.functional as F
from utils.matching_DINOv3 import grid_valid_mask_from_meta

def compute_gaussian_ce_loss_from_feats(
    src_feats: torch.Tensor,      # [B,C,Hf,Wf]
    trg_feats: torch.Tensor,      # [B,C,Hf,Wf]
    src_kps_list,                 # list len B of [K,2] pixel coords
    trg_kps_list,                 # list len B of [K,2] pixel coords
    src_meta_list,                # list len B of dict
    trg_meta_list,                # list len B of dict
    *,
    out_size: int,
    patch_size: int,
    temperature: float = 0.5,
    sigma: float = 2.0,           # in PATCH units
    eps: float = 1e-8,
) -> torch.Tensor:
    assert src_feats.ndim == 4 and trg_feats.ndim == 4, "Expected [B,C,Hf,Wf]"
    B, C, Hf, Wf = trg_feats.shape
    device = trg_feats.device

    srcn = F.normalize(src_feats, dim=1)
    trgn = F.normalize(trg_feats, dim=1)

    N = Hf * Wf
    trgn_flat = trgn.view(B, C, N).transpose(1, 2).contiguous()  # [B,N,C]

    yy, xx = torch.meshgrid(
        torch.arange(Hf, device=device, dtype=torch.float32),
        torch.arange(Wf, device=device, dtype=torch.float32),
        indexing="ij",
    )

    total_loss = torch.zeros((), device=device)
    total_kps = 0

    temp = max(float(temperature), eps)
    sig2 = 2.0 * float(sigma) * float(sigma)

    # a safe "very negative" for masking logits pre-softmax
    neg_large = -1e9
    if trg_feats.dtype in (torch.float16, torch.bfloat16):
        neg_large = -1e4  # safer for half precision to avoid inf/NaN in some kernels

    for b in range(B):
        src_kps = src_kps_list[b]
        trg_kps = trg_kps_list[b]
        src_meta = src_meta_list[b]
        trg_meta = trg_meta_list[b]

        if not isinstance(src_kps, torch.Tensor):
            src_kps = torch.tensor(src_kps, device=device, dtype=torch.float32)
        else:
            src_kps = src_kps.to(device=device, dtype=torch.float32)

        if not isinstance(trg_kps, torch.Tensor):
            trg_kps = torch.tensor(trg_kps, device=device, dtype=torch.float32)
        else:
            trg_kps = trg_kps.to(device=device, dtype=torch.float32)

        if src_kps.numel() == 0:
            continue

        valid = (
            (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) &
            (src_kps[:, 0] < out_size) & (src_kps[:, 1] < out_size) &
            (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0) &
            (trg_kps[:, 0] < out_size) & (trg_kps[:, 1] < out_size)
        )
        if valid.sum().item() == 0:
            continue

        src_kps_v = src_kps[valid]
        trg_kps_v = trg_kps[valid]

        # valid (non-padding) patch masks
        src_grid_valid = grid_valid_mask_from_meta(
            src_meta, out_size=out_size, patch=patch_size, device=device
        )  # [Hf,Wf]
        trg_grid_valid = grid_valid_mask_from_meta(
            trg_meta, out_size=out_size, patch=patch_size, device=device
        )  # [Hf,Wf]
        trg_valid_flat = trg_grid_valid.view(-1)  # [N] bool

        # source patch index = nearest patch CENTER
        sx = torch.clamp(((src_kps_v[:, 0] / patch_size) - 0.5).round().long(), 0, Wf - 1)
        sy = torch.clamp(((src_kps_v[:, 1] / patch_size) - 0.5).round().long(), 0, Hf - 1)

        # drop src keypoints that fall on padded src patches
        src_on_valid = src_grid_valid[sy, sx]
        if src_on_valid.sum().item() == 0:
            continue

        sx = sx[src_on_valid]
        sy = sy[src_on_valid]
        trg_kps_v = trg_kps_v[src_on_valid]
        Kv = int(trg_kps_v.shape[0])

        # [Kv,C]
        src_desc = srcn[b, :, sy, sx].transpose(0, 1).contiguous()

        # logits over target patches: [Kv,N]
        logits = (trgn_flat[b] @ src_desc.T).transpose(0, 1).contiguous()
        logits = logits / temp

        # -------------------------------------------------
        # MASK CORRETTA (AMP-safe)
        # -------------------------------------------------
        mask = ~trg_valid_flat.unsqueeze(0)  # [1, N] bool

        # fai la log_softmax in float32 per sicurezza numerica
        logits_fp32 = logits.float()
        logits_fp32 = logits_fp32.masked_fill(mask, -1e9)
        log_probs = F.log_softmax(logits_fp32, dim=-1)


        # gaussian centers in patch coords (center-consistent)
        cx = (trg_kps_v[:, 0] / patch_size) - 0.5
        cy = (trg_kps_v[:, 1] / patch_size) - 0.5

        for k in range(Kv):
            dx = xx - cx[k]
            dy = yy - cy[k]
            g = torch.exp(-(dx * dx + dy * dy) / (sig2 + eps))  # [Hf,Wf]
            g_flat = g.reshape(-1)

            # zero out padded target patches + renorm
            g_flat = g_flat * trg_valid_flat.float()
            denom = g_flat.sum()
            if denom.item() <= 0:
                # no valid mass (shouldn't happen unless metadata is broken)
                continue
            g_flat = g_flat / (denom + eps)

            # --- FIX #2: no 0 * (-inf) anymore, since log_probs is finite on valid support ---
            total_loss = total_loss - torch.sum(g_flat * log_probs[k])

        total_kps += Kv

    if total_kps == 0:
        return torch.zeros((), device=device)

    return total_loss / total_kps
