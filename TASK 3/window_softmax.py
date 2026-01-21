import torch
import torch.nn.functional as F 

def window_softmax(pred_y, pred_x, similarity_map_up, device, H_img, W_img):
    temperature=20, window_size=15
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
    # 15x15
    window_map = similarity_map_up[0, 0, top:bottom, left:right]

    # Applicare Softmax spaziale sulla finestra
    # Moltiplichiamo per temperature per affinare la distribuzione se necessario
    weights = F.softmax(window_map.flatten() * temperature, dim=0).view_as(window_map)

    # Generare griglia di coordinate relative alla finestra
    win_h, win_w = window_map.shape #15, 15
    # meshgrid crea due matrici 15x15
    #grid_x: in ogni riga ha [0, 1, 2, ..., 14] --> "questo pixel è alla colonna X"
    #grid_y: in ogni colonna ha [0, 1, 2, ..., 14] --> "questo pixel è alla riga Y"
    grid_y, grid_x = torch.meshgrid(torch.arange(win_h, device=device), #[0, 1, ..., 14]
                                    torch.arange(win_w, device=device), 
                                    indexing='ij')

    # Calcolo del valore atteso (Soft-Argmax locale)
    soft_dx = torch.sum(grid_x * weights)
    soft_dy = torch.sum(grid_y * weights)

    # Somma offset globale (top, left): posizione della finestra nell'immagine
    final_x = left + soft_dx
    final_y = top + soft_dy

    return final_x, final_y