import torch
import torch.nn.functional as F

def extract_features(model, img_tensor, model_type='sam'):
    """
    Estrae le feature dense dal backbone.
    Gestisce la differenza tra SAM e altri modelli.
    """
    if model_type == 'sam':
        if img_tensor.shape[-2:] != (1024, 1024):
            input_sam = F.interpolate(img_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
        else:
            input_sam = img_tensor

        features = model.image_encoder(input_sam)
    else:
        features = model.forward_features(img_tensor)['x_norm_patchtokens']
        pass

    return features

def compute_correspondence(src_feats, trg_feats, src_kps, img_size):
    """
    Calcola la similarità del coseno e trova il punto di match (punti 2 e 3 del PDF).
    """
    B, C, H_feat, W_feat = src_feats.shape
    H_img, W_img = img_size

    patch_size_h = H_img / H_feat
    patch_size_w = W_img / W_feat

    predictions = []

    src_feats = F.normalize(src_feats, p=2, dim=1)
    trg_feats = F.normalize(trg_feats, p=2, dim=1)

    for i in range(len(src_kps)):
        kp = src_kps[i] # (x, y)

        feat_x = int(kp[0] / patch_size_w)
        feat_y = int(kp[1] / patch_size_h)

        feat_x = min(max(feat_x, 0), W_feat - 1)
        feat_y = min(max(feat_y, 0), H_feat - 1)

        src_vec = src_feats[0, :, feat_y, feat_x].unsqueeze(0)

        trg_feats_flat = trg_feats.contiguous().view(C, -1)

        # Similarità map (1, H*W)
        similarity = torch.mm(src_vec, trg_feats_flat)

        # Rimodella in mappa 2D (1, 1, H_feat, W_feat)
        similarity_map = similarity.view(1, 1, H_feat, W_feat)

        similarity_map_up = F.interpolate(similarity_map, size=(H_img, W_img), mode='bilinear', align_corners=False)

        max_idx = torch.argmax(similarity_map_up)

        # Converti indice lineare in (y, x)
        pred_y = max_idx // W_img
        pred_x = max_idx % W_img

        predictions.append([pred_x.item(), pred_y.item()])

    return torch.tensor(predictions)
