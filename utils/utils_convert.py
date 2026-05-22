from typing import Tuple, Optional

import torch

def pixel_to_patch_idx(
    xy: torch.Tensor,
    stride: int,
    grid_hw: Tuple[int, int],
    img_hw: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Convert pixel coordinates to patch grid indices with boundary clamping.

    Args:
        xy: A tensor containing (x, y) coordinates in pixel space of the
            resized image.
        stride: The patch size or stride in pixels. Determines how many pixels form one patch.
        grid_hw: A tuple (H_grid, W_grid) specifying the valid patch grid
            dimensions (number of patches in height and width).
        img_hw: A tuple (H_img, W_img) specifying the image dimensions in pixels.
            Should match the resolution to which the input image was resized before feature extraction.

    Returns:
        A tuple (x_idx, y_idx) containing the patch indices, clamped to valid
            grid bounds
    """

    x = float(xy[0].item())
    y = float(xy[1].item())

    # clamp pixel on valid image boundaries
    H_img, W_img = int(img_hw[0]), int(img_hw[1])
    x = max(0.0, min(W_img - 1.0, x))
    y = max(0.0, min(H_img - 1.0, y))


    # Convert pixel coordinates to patch indices using integer division
    # Dividing by stride (patch size) gives the patch grid position
    x_idx = int(x // stride)
    y_idx = int(y // stride)

    # Clamp patch indices to valid grid bounds
    H_grid, W_grid = int(grid_hw[0]), int(grid_hw[1])
    x_idx = max(0, min(W_grid - 1, x_idx))
    y_idx = max(0, min(H_grid - 1, y_idx))

    return x_idx, y_idx

def patch_idx_to_pixel(xy_idx: tuple[float, float], stride: int):
    """
    Convert patch grid indices to pixel coordinates in the image space.

    Args:
        xy_idx: A tuple (x_idx, y_idx) containing patch grid indices.
        stride: The patch size or stride in pixels.

    Returns:
        A tuple (x, y) containing the converted pixel coordinates in the resized image space.
    """

    # Calculate pixel coordinates from patch indexes
    # The +0.5 centers the patch, because the patch index corresponds to the top-left corner of the patch,
    #   and we want the center of the patch in pixel space.
    #
    # * stride scales to pixel space
    #
    #  The -0.5 accounts for pixel centers being indexed at integer coordinates.
    x = (xy_idx[0] + 0.5) * stride - 0.5
    y = (xy_idx[1] + 0.5) * stride - 0.5

    return x, y