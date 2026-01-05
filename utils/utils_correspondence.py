from typing import Tuple
import torch.nn.functional as F
import torch


def hard_argmax(sim2d: torch.Tensor) -> Tuple[int, int]:
    """
    2D hard argmax.
    Args:
        sim2d: Tensor (Ht, Wt) — similarity map

    Returns:
        (x_idx, y_idx): coordinates of the estimated keypoint in the target image
    """

    if sim2d.dim() != 2:
        raise ValueError("sim2d must have shape (Ht, Wt)")

    sim_h, sim_w = sim2d.shape

    flat_idx = torch.argmax(sim2d) # flat index of the max value
    y_idx = (flat_idx // sim_w).long()
    x_idx = (flat_idx %  sim_w).long()

    return int(x_idx.item()), int(y_idx.item())


def argmax(
    sim2d: torch.Tensor,
    window_size: int = 5,
    beta: float = 50.0
) -> Tuple[float, float]:
    """
    2D argmax with optional windowed soft-argmax refinement.
    Args:
        sim2d: Tensor (Ht, Wt) — similarity map
        window_size:
            - 1  -> hard argmax
            - >1 -> window soft-argmax
        beta: softmax temperature

    Returns:
        (x_idx, y_idx): coordinates of the estimated keypoint in the target image
    """

    max_x, max_y = hard_argmax(sim2d)

    # hard argmax
    if window_size <= 1:
        return max_x, max_y

    sim_h, sim_w = sim2d.shape
    window_radius = (window_size - 1) // 2

    # window soft-argmax
    y1 = max(max_y - window_radius, 0)
    y2 = min(max_y + window_radius + 1, sim_h)
    x1 = max(max_x - window_radius, 0)
    x2 = min(max_x + window_radius + 1, sim_w)

    # Build window
    window = sim2d[y1:y2, x1:x2]

    prob = F.softmax((window * beta).reshape(-1), dim=0).view_as(window)

    column_indices = torch.arange(x1, x2, device=sim2d.device, dtype=sim2d.dtype)
    row_indices = torch.arange(y1, y2, device=sim2d.device, dtype=sim2d.dtype)

    x_patch = (prob.sum(dim=0) * column_indices).sum()
    y_patch = (prob.sum(dim=1) * row_indices).sum()

    return float(x_patch.item()), float(y_patch.item())