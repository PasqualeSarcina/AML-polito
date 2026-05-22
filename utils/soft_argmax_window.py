import torch


def soft_argmax_window(sim_map_2d, window_radius=3, temperature=0.05):
    """
    Args:
        sim_map_2d: 2D similarity map of shape (H, W) containing scalar similarity scores.
        window_radius: Radius of the local window centered at the hard argmax peak. A window radius of 3 results in a
            7x7 window. If 0, no window is applied and the hard argmax coordinates are returned.
        temperature: Sharpening factor. Lower = closer to hard argmax.
    Returns:
        y_soft, x_soft: Float coordinates on the grid.
    Notes:
        - If window_radius=0, the function returns the hard peak coordinates as floats.
    """
    H, W = sim_map_2d.shape

    # Find the "hard" peak with argmax
    flattened = sim_map_2d.view(-1)
    idx = torch.argmax(flattened)
    y_hard = idx // W
    x_hard = idx % W

    # Return the peak patch coordinates as floats if no window is applied
    if window_radius == 0:
        return y_hard.float(), x_hard.float()

    # Define the Window around the peak
    y_min = max(0, y_hard - window_radius)
    y_max = min(H, y_hard + window_radius + 1)
    x_min = max(0, x_hard - window_radius)
    x_max = min(W, x_hard + window_radius + 1)

    # Crop the similarity map to the window around the hard peak
    window = sim_map_2d[y_min:y_max, x_min:x_max]

    # Convert Scores to Probabilities with Softmax
    # We subtract the maximum score for numerical stability before applying softmax:
    # this keeps the largest value equal to 0 and avoids very large exponentials, without changing the final softmax probabilities.
    # Flatten the window to 1D so Softmax considers ALL pixels together
    flat_input = ((window - window.max()) / temperature).view(-1)
    # Apply Softmax on the flat array (dim=0)
    flat_weights = torch.nn.functional.softmax(flat_input, dim=0)

    # Reshape back to the original 2D square shape
    weights = flat_weights.view(window.shape)

    # Compute the center of mass of the similarity map via a weighted sum.
    # Create a 2D grid of absolute coordinates for the selected window.
    # local_y stores the y-coordinate for each cell in the window.
    # local_x stores the x-coordinate for each cell in the window.
    device = sim_map_2d.device
    local_y, local_x = torch.meshgrid(
        torch.arange(y_min, y_max, device=device).float(),
        torch.arange(x_min, x_max, device=device).float(),
        indexing='ij'
    )

    # Weighted average of coordinates
    y_soft = torch.sum(weights * local_y)
    x_soft = torch.sum(weights * local_x)

    return float(y_soft.item()), float(x_soft.item())