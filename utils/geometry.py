import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TASK_3_SAM.window_softmax import window_softmax

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

def compute_correspondence(src_feats, trg_feats, src_kps, img_size, softmax_flag=True):
    
    B, C, H_feat, W_feat = src_feats.shape
    H_img, W_img = img_size
    device = src_feats.device
    
    # 1. Normalizzazione L2 delle feature
    src_feats = F.normalize(src_feats, p=2, dim=1)
    trg_feats = F.normalize(trg_feats, p=2, dim=1)

    # 2. Estrazione vettorializzata delle feature sorgente nei punti chiave (Interpolazione Bilineare)
    # Normalizza coordinate kps in [-1, 1]
    # src_kps è (N, 2), aggiungiamo dimensioni per grid_sample -> (1, N, 1, 2)
    # Nota: grid_sample si aspetta (x, y)
    
    kps_norm = src_kps.clone()
    kps_norm[:, 0] = 2 * (kps_norm[:, 0] / W_img) - 1
    kps_norm[:, 1] = 2 * (kps_norm[:, 1] / H_img) - 1
    kps_norm = kps_norm.view(1, -1, 1, 2).to(device)

    # Campiona le feature sorgente: Output (1, C, N, 1)
    src_vecs = F.grid_sample(src_feats, kps_norm, mode='bilinear', align_corners=False)
    src_vecs = src_vecs.view(C, -1).permute(1, 0) # (N, C)

    # 3. Calcolo Similarità (Tutti i keypoints insieme)
    # trg_feats appiattito: (C, H_feat * W_feat)
    trg_feats_flat = trg_feats.view(C, -1)
    
    # Matrice di similarità: (N, H_feat * W_feat)
    similarity_matrix = torch.mm(src_vecs, trg_feats_flat)
    
    # 4. Reshape e Upsample delle mappe di similarità
    N = len(src_kps)
    similarity_maps = similarity_matrix.view(N, 1, H_feat, W_feat)
    
    # Upsample a risoluzione immagine (N, 1, H_img, W_img)
    similarity_maps_up = F.interpolate(similarity_maps, size=(H_img, W_img), mode='bilinear', align_corners=False)
    
    # 5. Trova coordinate (Argmax)
    # Appiattisci spazialmente
    similarity_maps_up_flat = similarity_maps_up.view(N, -1)
    max_indices = torch.argmax(similarity_maps_up_flat, dim=1)
    
    pred_y = max_indices // W_img
    pred_x = max_indices % W_img
    
    if softmax_flag:
        refined_point = []
        for i in range(N):
            px, py = window_softmax(pred_y[i], pred_x[i], similarity_maps_up[i:i+1], device, H_img, W_img)
            refined_point.append(torch.stack([px, py]))

        predictions = torch.stack(refined_point)
    else:
        predictions = torch.stack([pred_x, pred_y], dim=1).float()
    
    return predictions