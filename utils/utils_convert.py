from typing import Tuple, Optional

import torch

def pixel_to_patch_idx(
    xy: torch.Tensor,
    stride: int,
    grid_hw: Tuple[int, int],
    img_hw: Tuple[int, int]
) -> Tuple[int, int]:
    """
    (x,y) pixel -> (x_idx,y_idx) patch.

    Args:
        xy: (x,y) in pixel nello spazio resized.
        stride: patch/stride effettivo (16 per DIFT/SAM, 14 per DINO).
        grid_hw: (H_grid, W_grid) numero di patch validi.
        img_hw: (H_img, W_img) image size (same as model input and ti which poit have been scaled).

    Returns:
        xy_idx, y_idx) clampati in griglia valida.
    """
    # leggi (x,y)
    x = float(xy[0].item())
    y = float(xy[1].item())

    # clamp pixel
    H_img, W_img = int(img_hw[0]), int(img_hw[1])
    x = max(0.0, min(W_img - 1.0, x))
    y = max(0.0, min(H_img - 1.0, y))

    # pixel -> patch
    x_idx = int(x // stride)
    y_idx = int(y // stride)

    # clamp su griglia valida
    H_grid, W_grid = int(grid_hw[0]), int(grid_hw[1])
    x_idx = max(0, min(W_grid - 1, x_idx))
    y_idx = max(0, min(H_grid - 1, y_idx))

    return x_idx, y_idx

def patch_idx_to_pixel(xy_idx: tuple[float, float], stride: int):
    x = (xy_idx[0] + 0.5) * stride - 0.5
    y = (xy_idx[1] + 0.5) * stride - 0.5
    return x, y