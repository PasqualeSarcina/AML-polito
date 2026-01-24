import torch
import torch.nn.functional as F 

def window_softmax(pred_y, pred_x, similarity_map_up, device, H_img, W_img):
    temperature=10
    window_size=13
    radius = window_size // 2
    # Conversione a int per sicurezza nello slicing
    cy = pred_y.item() if isinstance(pred_y, torch.Tensor) else pred_y
    cx = pred_x.item() if isinstance(pred_x, torch.Tensor) else pred_x
    
    # Clamp per non uscire dai bordi dell'immagine
    top = max(cy - radius, 0)
    bottom = min(cy + radius + 1, H_img)
    left = max(cx - radius, 0)
    right = min(cx + radius + 1, W_img)

    # Estrarre FINESTRA LOCALE dalla heatmap --> prendiamo [0, 0, y, x]
    window_map = similarity_map_up[0, 0, top:bottom, left:right]

    # Applicare Softmax spaziale sulla finestra
    # Moltiplichiamo per temperature per affinare la distribuzione se necessario
    flat_input = ((window_map - window_map.max()) * temperature).view(-1)
    weights = F.softmax(flat_input, dim=0).view_as(window_map)

    # Generare griglia di coordinate
    grid_y, grid_x = torch.meshgrid(
        torch.arange(top, bottom, device=device).float(),
        torch.arange(left, right, device=device).float(),
        indexing='ij')

    # Calcolo del valore atteso (Soft-Argmax locale)
    soft_dx = torch.sum(grid_x * weights)
    soft_dy = torch.sum(grid_y * weights)

    return soft_dx, soft_dy