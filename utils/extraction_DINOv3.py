import torch
import torch.nn.functional as F


def extract_dense_features(
    model,
    img: torch.Tensor,          # CHW or BCHW, normalized
    n_layers: int = 1,          # last n layers
    return_grid: bool = False,  # IMPORTANT: default False (avoids tuple bugs)
):
    """
    Returns:
      feat: [B,C,Hf,Wf] L2-normalized over channel
      (Hf, Wf) if return_grid=True
    Notes:
      - Works with CHW and BCHW.
      - Uses last N_spatial tokens to be robust to CLS/register tokens.
    """
    assert img.ndim in (3, 4), f"Expected CHW or BCHW, got {tuple(img.shape)}"
    if img.ndim == 3:
        assert img.shape[0] == 3
        x = img.unsqueeze(0)
    else:
        assert img.shape[1] == 3
        x = img

    dev = next(model.parameters()).device
    x = x.to(dev)

    outs = model.get_intermediate_layers(x, n=n_layers)  # list length n_layers
    B, Nt, C = outs[-1].shape

    patch = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if patch is None:
        patch = getattr(model, "patch_size", 16)
    patch = int(patch[0]) if isinstance(patch, (tuple, list)) else int(patch)

    H, W = x.shape[-2], x.shape[-1]
    Hf, Wf = H // patch, W // patch
    N_spatial = Hf * Wf

    maps = []
    for tok in outs:
        spatial = tok[:, -N_spatial:, :]  # [B, Hf*Wf, C]
        fm = spatial.transpose(1, 2).contiguous().view(B, C, Hf, Wf)
        maps.append(fm)

    feat = torch.stack(maps, dim=0).mean(dim=0)  # [B,C,Hf,Wf]
    feat = F.normalize(feat, p=2, dim=1)

    if return_grid:
        return feat, (Hf, Wf)
    return feat