import torch


def soft_argmax_window(sim_map_2d, window_radius=3, temperature=20):
    """
    Args:
        sim_map_2d: Tensor shape (H, W) containing similarity scores.
        window_radius: How many neighbors to look at (e.g., 3).
        temperature: Sharpening factor. Higher = closer to hard argmax.
    Returns:
        y_soft, x_soft: Float coordinates on the grid.
    """
    H, W = sim_map_2d.shape

    # 1. Find the Hard Peak (Integer)
    flattened = sim_map_2d.view(-1)
    idx = torch.argmax(flattened)
    y_hard = idx // W
    x_hard = idx % W

    if window_radius == 1:
        return y_hard.float(), x_hard.float()

    # 2. Define the Window around the peak
    y_min = max(0, y_hard - window_radius)
    y_max = min(H, y_hard + window_radius + 1)
    x_min = max(0, x_hard - window_radius)
    x_max = min(W, x_hard + window_radius + 1)

    # 3. Crop the window
    window = sim_map_2d[y_min:y_max, x_min:x_max]

    # 4. Convert Scores to Probabilities (Softmax)
    # We subtract max for numerical stability, then multiply by temperature
    # Cosine similarity is usually -1 to 1. We scale it up so Softmax isn't too flat.
    # 1. Flatten the window to 1D so Softmax considers ALL pixels together
    flat_input = ((window - window.max()) * temperature).view(-1)

    # 2. Apply Softmax on the flat array (dim=0)
    flat_weights = torch.nn.functional.softmax(flat_input, dim=0)

    # 3. Reshape back to the original 2D square shape
    weights = flat_weights.view(window.shape)
    # 5. Calculate Center of Mass (Weighted Sum)
    # Create a grid of coordinates for the window
    device = sim_map_2d.device
    local_y, local_x = torch.meshgrid(
        torch.arange(y_min, y_max, device=device).float(),
        torch.arange(x_min, x_max, device=device).float(),
        indexing='ij'
    )

    y_soft = torch.sum(weights * local_y)
    x_soft = torch.sum(weights * local_x)

    return float(y_soft.item()), float(x_soft.item())