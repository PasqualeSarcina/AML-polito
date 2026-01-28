import torch
from collections import defaultdict
from tqdm import tqdm
from utils.extraction_DINOv3 import extract_dense_features
from utils.matching_DINOv3 import match_argmax_nearest_patch_masked

@torch.no_grad()
def evaluate_model_batched(
    name: str,
    model,
    loader,
    max_pairs: int | None = None,
    n_layers: int = 1,
    patch_size: int | None = None,
    thresholds=(0.05, 0.10, 0.20),
):
    model.eval()
    dev = next(model.parameters()).device

    if patch_size is None:
        patch = getattr(getattr(model, "patch_embed", None), "patch_size", None)
        if patch is None:
            patch = getattr(model, "patch_size", 16)
        patch_size = int(patch[0]) if isinstance(patch, (tuple, list)) else int(patch)

    # accumulators
    num_pairs = 0
    num_valid_kps = 0

    micro_correct = {t: 0 for t in thresholds}
    micro_total   = {t: 0 for t in thresholds}

    per_img_sum = {t: 0.0 for t in thresholds}
    per_img_n   = {t: 0 for t in thresholds}

    per_cat = defaultdict(lambda: {
        "num_images": 0,
        "num_valid_keypoints": 0,
        "micro_correct": {t: 0 for t in thresholds},
        "micro_total":   {t: 0 for t in thresholds},
        "img_sum":       {t: 0.0 for t in thresholds},
        "img_n":         {t: 0 for t in thresholds},
    })

    pbar = tqdm(loader, desc=name, leave=True)

    for batch in pbar:
        # stop condition on PAIRS (images), not on loader batches
        if max_pairs is not None and num_pairs >= max_pairs:
            break

        src_img = batch["src_img"].to(dev, non_blocking=True)     # [B,3,H,W]
        trg_img = batch["trg_img"].to(dev, non_blocking=True)
        trg_bbox = batch["trg_bbox"].to(dev, non_blocking=True)   # [B,4]

        cats = batch["category"]               # list len B
        src_kps_list = batch["src_kps"]        # list of [Ki,2]
        trg_kps_list = batch["trg_kps"]
        src_meta_list = batch["src_meta"]      # list of dict
        trg_meta_list = batch["trg_meta"]

        B = src_img.shape[0]
        out_size = int(src_img.shape[-1])

        # batched feats (IMPORTANT: return_grid=False)
        src_feats = extract_dense_features(model, src_img, n_layers=n_layers, return_grid=False)  # [B,C,Hf,Wf]
        trg_feats = extract_dense_features(model, trg_img, n_layers=n_layers, return_grid=False)

        for b in range(B):
            if max_pairs is not None and num_pairs >= max_pairs:
                break

            src_kps = src_kps_list[b].to(dev)
            trg_kps = trg_kps_list[b].to(dev)
            cat = cats[b]
            src_meta = src_meta_list[b]
            trg_meta = trg_meta_list[b]

            # valid in-bounds in preprocess coords (for both GTs)
            src_valid = (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) & (src_kps[:, 0] < out_size) & (src_kps[:, 1] < out_size)
            trg_valid = (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0) & (trg_kps[:, 0] < out_size) & (trg_kps[:, 1] < out_size)
            valid = src_valid & trg_valid
            if valid.sum().item() == 0:
                continue

            # padding-aware matching (single-pair call: B=1)
            pred_kps, valid_src_mask = match_argmax_nearest_patch_masked(
                src_feats[b:b+1], trg_feats[b:b+1], src_kps,
                out_size=out_size, patch=patch_size,
                src_meta=src_meta, trg_meta=trg_meta
            )
            valid = valid & valid_src_mask
            if valid.sum().item() == 0:
                continue

            num_pairs += 1
            Kv = int(valid.sum().item())
            num_valid_kps += Kv

            # PCK norm: max(w,h) from target bbox
            bb = trg_bbox[b]
            w = (bb[2] - bb[0]).clamp(min=1.0)
            h = (bb[3] - bb[1]).clamp(min=1.0)
            norm = torch.max(w, h)

            d = torch.norm(pred_kps[valid] - trg_kps[valid], dim=1)

            # per-image acc
            for t in thresholds:
                thr = t * norm
                correct = (d <= thr).float()
                c = int(correct.sum().item())
                n = int(correct.numel())

                micro_correct[t] += c
                micro_total[t] += n

                per_img_sum[t] += float(correct.mean().item())
                per_img_n[t] += 1

                per_cat[cat]["micro_correct"][t] += c
                per_cat[cat]["micro_total"][t] += n
                per_cat[cat]["img_sum"][t] += float(correct.mean().item())
                per_cat[cat]["img_n"][t] += 1

            per_cat[cat]["num_images"] += 1
            per_cat[cat]["num_valid_keypoints"] += Kv

        pbar.set_postfix(pairs=num_pairs, valid_kps=num_valid_kps)

    # build report
    report = {
        "name": name,
        "n_layers": n_layers,
        "num_images": num_pairs,
        "num_valid_keypoints": num_valid_kps,
        "pck_per_keypoint": {},
        "pck_per_image_mean": {},
        "per_category": {}
    }

    for t in thresholds:
        report["pck_per_keypoint"][t] = (micro_correct[t] / micro_total[t]) if micro_total[t] > 0 else 0.0
        report["pck_per_image_mean"][t] = (per_img_sum[t] / per_img_n[t]) if per_img_n[t] > 0 else 0.0

    for cat, stats in per_cat.items():
        entry = {
            "num_images": stats["num_images"],
            "num_valid_keypoints": stats["num_valid_keypoints"],
            "pck_per_keypoint": {},
            "pck_per_image_mean": {}
        }
        for t in thresholds:
            tot = stats["micro_total"][t]
            entry["pck_per_keypoint"][t] = (stats["micro_correct"][t] / tot) if tot > 0 else 0.0
            nimg = stats["img_n"][t]
            entry["pck_per_image_mean"][t] = (stats["img_sum"][t] / nimg) if nimg > 0 else 0.0
        report["per_category"][cat] = entry

    return report