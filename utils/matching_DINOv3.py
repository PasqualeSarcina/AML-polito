import torch
import torch.nn.functional as F


def grid_valid_mask_from_meta(meta: dict, out_size: int, patch: int, device=None) -> torch.Tensor:
    """
    Boolean mask [Hf,Wf] where True = valid (non-padding) patch.
    A patch is valid if its CENTER (in pixel coords) lies inside the non-padded region.
    """
    Hf = out_size // patch
    Wf = out_size // patch

    pad_x = int(meta["pad_x"])
    pad_y = int(meta["pad_y"])
    H_rs  = int(meta["H_rs"])
    W_rs  = int(meta["W_rs"])

    # Non-padding region in pixel coords (half-open intervals)
    x0, x1 = pad_x, pad_x + W_rs
    y0, y1 = pad_y, pad_y + H_rs

    # Patch centers in pixel coords
    xs = (torch.arange(Wf, device=device, dtype=torch.float32) + 0.5) * patch
    ys = (torch.arange(Hf, device=device, dtype=torch.float32) + 0.5) * patch

    valid_x = (xs >= x0) & (xs < x1)
    valid_y = (ys >= y0) & (ys < y1)

    return valid_y[:, None] & valid_x[None, :]


def _kps_valid_mask(kps_xy: torch.Tensor, out_size: int) -> torch.Tensor:
    x = kps_xy[:, 0]
    y = kps_xy[:, 1]
    return (x >= 0) & (y >= 0) & (x < out_size) & (y < out_size)


@torch.no_grad()
def match_argmax_nearest_patch_masked(
    src_feat: torch.Tensor,        # [1,C,Hf,Wf]
    trg_feat: torch.Tensor,        # [1,C,Hf,Wf]
    src_kps_xy: torch.Tensor,      # [K,2] pixel coords in [0,out_size)
    out_size: int,
    patch: int,
    src_meta: dict,
    trg_meta: dict,
):
    device = src_feat.device
    src_kps_xy = src_kps_xy.to(device)

    B, C, Hf, Wf = src_feat.shape
    assert B == 1, "This function expects features for a single pair (B=1)."

    # masks of valid (non-padding) patches
    src_grid_valid = grid_valid_mask_from_meta(src_meta, out_size=out_size, patch=patch, device=device)  # [Hf,Wf]
    trg_grid_valid = grid_valid_mask_from_meta(trg_meta, out_size=out_size, patch=patch, device=device)  # [Hf,Wf]

    K = src_kps_xy.shape[0]
    pred = torch.full((K, 2), -1.0, device=device)

    # keypoints valid in image bounds
    valid = _kps_valid_mask(src_kps_xy, out_size)
    if valid.sum().item() == 0:
        return pred, valid

    xs = src_kps_xy[valid, 0]
    ys = src_kps_xy[valid, 1]

    # Assign keypoint to the NEAREST PATCH CENTER
    # Patch center at index i is (i + 0.5) * patch => i ~= xs/patch - 0.5
    ix = torch.clamp(((xs / patch) - 0.5).round().long(), 0, Wf - 1)
    iy = torch.clamp(((ys / patch) - 0.5).round().long(), 0, Hf - 1)

    # Invalidate source keypoints that fall on padded patches
    src_kp_on_valid_patch = src_grid_valid[iy, ix]  # [Kv]
    if src_kp_on_valid_patch.sum().item() == 0:
        valid2 = valid.clone()
        valid_idx_all = valid.nonzero(as_tuple=False).squeeze(1)
        valid2[valid_idx_all] = False
        return pred, valid2

    keep = src_kp_on_valid_patch
    ix2 = ix[keep]
    iy2 = iy[keep]

    # Gather source descriptors: [Kkeep,C]
    src_map = src_feat.squeeze(0)  # [C,Hf,Wf]
    desc = src_map[:, iy2, ix2].transpose(0, 1).contiguous()
    desc = F.normalize(desc, p=2, dim=1)

    # Flatten target descriptors: [C,HW]
    trg_flat = trg_feat.squeeze(0).view(C, Hf * Wf)
    trg_flat = F.normalize(trg_flat, p=2, dim=0)

    sim = desc @ trg_flat  # [Kkeep, HW]

    # Exclude padded target patches
    trg_valid_flat = trg_grid_valid.view(-1)  # [HW]
    sim[:, ~trg_valid_flat] = float("-inf")

    idx = sim.argmax(dim=1)  # [Kkeep]
    iy_t = torch.div(idx, Wf, rounding_mode="floor")
    ix_t = idx - iy_t * Wf

    # Predicted pixel coords = target patch CENTER
    px = (ix_t.float() + 0.5) * patch
    py = (iy_t.float() + 0.5) * patch
    pred_keep = torch.stack([px, py], dim=1)

    # Write back predictions into full K array
    valid_idx_all = valid.nonzero(as_tuple=False).squeeze(1)  # indices of "valid in bounds" kps
    keep_idx = valid_idx_all[keep]                            # subset indices (also valid on src non-padding)
    pred[keep_idx] = pred_keep

    # Final validity: in-bounds AND on non-padding src patch
    valid_out = valid.clone()
    valid_out[valid_idx_all[~keep]] = False
    return pred, valid_out
