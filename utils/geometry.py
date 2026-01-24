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

        if softmax_flag:
            pred_x, pred_y = window_softmax(pred_y, pred_x, similarity_map_up, device,
                                            H_img, W_img)

        predictions.append([pred_x.item(), pred_y.item()])

    return torch.tensor(predictions)

def compute_correspondence_NEW(src_feats, trg_feats, src_kps, img_size, softmax_flag=True):
    
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
    
    # Stack per output (N, 2)
    predictions = torch.stack([pred_x, pred_y], dim=1).float()
    
    return predictions