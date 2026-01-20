import torch
import torch.nn.functional as F

def extract_features(model, img_tensor, model_type='sam'):
    """
    Estrae le feature dense dal backbone.
    Gestisce la differenza tra SAM e altri modelli.
    """
    if model_type == 'sam':
        features = model.image_encoder(img_tensor)
    else:
        features = model.forward_features(img_tensor)['x_norm_patchtokens']
        pass

    return features

def compute_correspondence(src_feats, trg_feats, src_kps, img_size, temperature=20, window_size=15):
    """
    Calcola la similarità del coseno e trova il punto di match (punti 2 e 3 del PDF).
    """
    B, C, H_feat, W_feat = src_feats.shape
    H_img, W_img = img_size
    device = src_feats.device

    patch_size_h = H_img / H_feat
    patch_size_w = W_img / W_feat

    predictions = []

    src_feats = F.normalize(src_feats, p=2, dim=1)
    trg_feats = F.normalize(trg_feats, p=2, dim=1)
    trg_feats_flat = trg_feats.contiguous().view(C, -1)

    for i in range(len(src_kps)):
        kp = src_kps[i] # (x, y)

        feat_x = int(kp[0] / patch_size_w)
        feat_y = int(kp[1] / patch_size_h)
        feat_x = min(max(feat_x, 0), W_feat - 1)
        feat_y = min(max(feat_y, 0), H_feat - 1)

        src_vec = src_feats[0, :, feat_y, feat_x].unsqueeze(0)

        # Similarità map (1, H*W)
        similarity = torch.mm(src_vec, trg_feats_flat)

        # Rimodella in mappa 2D (1, 1, H_feat, W_feat)
        similarity_map = similarity.view(1, 1, H_feat, W_feat)
        similarity_map_up = F.interpolate(similarity_map, size=(H_img, W_img), mode='bilinear', align_corners=False)
        max_idx = torch.argmax(similarity_map_up)

        # Converti indice lineare in (y, x)
        pred_y = max_idx // W_img
        pred_x = max_idx % W_img

        pred_x, pred_y = window_softmax(pred_y, pred_x, similarity_map_up, device,
                                        H_img, W_img, temperature=20, window_size=15)

        predictions.append([pred_x.item(), pred_y.item()])

    return torch.tensor(predictions)

def window_softmax(pred_y, pred_x, similarity_map_up, device, H_img, W_img, temperature=20, window_size=15):
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