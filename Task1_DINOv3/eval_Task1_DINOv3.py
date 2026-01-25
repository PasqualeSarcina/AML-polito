import torch
from pathlib import Path
import os

import time
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.dinov3.model_DINOv3 import load_dinov3_backbone
from dataset.dataset_DINOv3 import SPairDataset, collate_single
from utils.extraction_DINOv3 import extract_dense_features
from utils.matching_DINOv3 import grid_valid_mask_from_meta, _kps_valid_mask, match_argmax_nearest_patch_masked
from utils.printing_helpers_DINOv3 import print_report, print_per_category
from utils.setup_data_DINOv3 import setup_data

# ----------------------------
# Evaluation (global + per-category, per-kp and per-img)
# ----------------------------
@torch.no_grad()
def evaluate_model(
    name: str,
    model,
    loader,
    device,
    n_layers: int = 1,
    thresholds=(0.05, 0.10, 0.20),
    max_pairs: int | None = None,
    verbose_every: int = 0,   # 0 = no periodic prints
):
    model.eval()

    # Robust patch size retrieval
    patch = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if patch is None:
        patch = getattr(model, "patch_size", 16)
    patch = int(patch[0]) if isinstance(patch, (tuple, list)) else int(patch)

    # Accumulators
    total_valid_kps = 0
    micro_correct = {t: 0 for t in thresholds}  # correct keypoints (global)
    micro_total   = {t: 0 for t in thresholds}  # valid keypoints (global)

    macro_sum = {t: 0.0 for t in thresholds}    # sum of per-pair accuracies
    macro_n   = {t: 0 for t in thresholds}      # number of evaluated pairs

    # Per-category accumulators (micro + macro)
    cat_micro_correct = defaultdict(lambda: {t: 0 for t in thresholds})
    cat_micro_total   = defaultdict(lambda: {t: 0 for t in thresholds})
    cat_macro_sum     = defaultdict(lambda: {t: 0.0 for t in thresholds})
    cat_macro_n       = defaultdict(lambda: {t: 0 for t in thresholds})

    t0 = time.time()
    pairs_seen = 0

    pbar = tqdm(
        loader,
        total=(max_pairs if max_pairs is not None else len(loader)),
        desc=f"{name}",
        leave=True
    )

    for i, sample in enumerate(pbar):
        if max_pairs is not None and pairs_seen >= max_pairs:
            break

        cat = sample.get("category", "unknown")
        pair_id = sample.get("pair_id", f"pair_{i}")

        src_img = sample["src_img"].to(device)      # [C,H,W] or [1,C,H,W]
        trg_img = sample["trg_img"].to(device)
        src_kps = sample["src_kps"].to(device)      # [K,2]
        trg_kps = sample["trg_kps"].to(device)      # [K,2]
        trg_bbox = sample["trg_bbox"].to(device)    # [4] xyxy in preprocess coords

        # IMPORTANT: meta is on CPU (python dict). Keep it as-is.
        src_meta = sample["src_meta"]
        trg_meta = sample["trg_meta"]

        # Ensure CHW for extractor
        src_chw = src_img.squeeze(0) if src_img.ndim == 4 else src_img
        trg_chw = trg_img.squeeze(0) if trg_img.ndim == 4 else trg_img

        out_size = int(src_chw.shape[-1])  # square
        img_w = out_size
        img_h = out_size

        # Valid mask: keypoints valid in BOTH images (in preprocess coords)
        src_valid = (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) & (src_kps[:, 0] < img_w) & (src_kps[:, 1] < img_h)
        trg_valid = (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0) & (trg_kps[:, 0] < img_w) & (trg_kps[:, 1] < img_h)
        valid = src_valid & trg_valid

        if valid.sum().item() == 0:
            continue

        # Extract dense features
        src_feat = extract_dense_features(model, src_chw, n_layers=n_layers, return_grid=False)  # [1,C,Hf,Wf]
        trg_feat = extract_dense_features(model, trg_chw, n_layers=n_layers, return_grid=False)

        # Predict target keypoints (argmax baseline, but masked to ignore padded patches)
        pred_kps, valid_src_mask = match_argmax_nearest_patch_masked(
            src_feat=src_feat,
            trg_feat=trg_feat,
            src_kps_xy=src_kps,
            out_size=out_size,
            patch=patch,
            src_meta=src_meta,
            trg_meta=trg_meta,
        )

        # Score only keypoints that are valid in both GT and matchable by the matcher
        valid = valid & valid_src_mask
        if valid.sum().item() == 0:
            continue

        pairs_seen += 1
        total_valid_kps += int(valid.sum().item())

        pbar.set_postfix({
            "pairs": pairs_seen,
            "valid_kps": total_valid_kps
        })

        # PCK normalization using target bbox (xyxy)
        w = (trg_bbox[2] - trg_bbox[0]).clamp(min=1.0)
        h = (trg_bbox[3] - trg_bbox[1]).clamp(min=1.0)
        norm = torch.max(w, h)

        # Distances for valid keypoints
        d = torch.norm(pred_kps[valid] - trg_kps[valid], dim=1)  # [Kv]

        # Per-pair PCK for macro (mean over keypoints in this pair)
        for t in thresholds:
            thr = t * norm
            correct = (d <= thr).float()
            pair_acc = float(correct.mean().item())

            c = int(correct.sum().item())
            n = int(correct.numel())

            # Global micro
            micro_correct[t] += c
            micro_total[t] += n

            # Global macro
            macro_sum[t] += pair_acc
            macro_n[t] += 1

            # Per-category micro
            cat_micro_correct[cat][t] += c
            cat_micro_total[cat][t] += n

            # Per-category macro
            cat_macro_sum[cat][t] += pair_acc
            cat_macro_n[cat][t] += 1

        if verbose_every and (pairs_seen % verbose_every == 0):
            elapsed = time.time() - t0
            print(f"[{name}] pairs={pairs_seen} valid_kps={total_valid_kps} elapsed={elapsed:.1f}s last_pair={pair_id}")

    # Build report dict
    report = {
        "name": name,
        "n_layers": n_layers,
        "pairs_evaluated": pairs_seen,
        "valid_keypoints": total_valid_kps,
        "patch_size": patch,
        "thresholds": list(thresholds),
        "global_pck_micro": {},
        "global_pck_macro": {},
        "per_category": {},
    }

    for t in thresholds:
        report["global_pck_micro"][t] = (micro_correct[t] / micro_total[t]) if micro_total[t] > 0 else 0.0
        report["global_pck_macro"][t] = (macro_sum[t] / macro_n[t]) if macro_n[t] > 0 else 0.0

    # Per-category report
    cats = sorted(set(list(cat_micro_total.keys()) + list(cat_macro_n.keys())))
    for cat in cats:
        entry = {
            "pck_micro": {},
            "pck_macro": {},
            "pairs": int(max(cat_macro_n[cat].values()) if len(cat_macro_n[cat]) else 0),
            "valid_kps": int(max(cat_micro_total[cat].values()) if len(cat_micro_total[cat]) else 0),
        }
        for t in thresholds:
            tot = cat_micro_total[cat][t]
            entry["pck_micro"][t] = (cat_micro_correct[cat][t] / tot) if tot > 0 else 0.0
            n = cat_macro_n[cat][t]
            entry["pck_macro"][t] = (cat_macro_sum[cat][t] / n) if n > 0 else 0.0
        report["per_category"][cat] = entry

    return report

if __name__ == "__main__":
    print("--- 1. Checking Data Availability ---")
    data_root = setup_data()

    if data_root is None:
        print("Dataset not found. Please follow the instructions in README.md to download and prepare the SPair-71k dataset.")
        exit()
    
    base_dir = os.path.join(data_root, 'SPair-71k')
    pair_ann_path = os.path.join(base_dir, 'PairAnnotation')
    layout_path = os.path.join(base_dir, 'Layout')
    image_path = os.path.join(base_dir, 'JPEGImages')

    if not os.path.exists(pair_ann_path) or not os.path.exists(layout_path) or not os.path.exists(image_path):
        print("Dataset structure is incorrect. Please ensure that the SPair-71k dataset is organized as specified in README.md.")
        exit()

    print("\n--- 2. Loading Test Dataset ---")

    test_dataset = SPairDataset(
        spair_root=Path(base_dir),
        split="test",
        layout_size="large",
        out_size=512, 
        pad_mode="center",
        max_pairs=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_single,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Test dataset loaded: {len(test_dataset)} pairs.")

    print("\n--- 3. Loading DINOv3 Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dinov3_dir = Path("/content/dinov3") if Path("/content/dinov3").exists() else Path("third_party/dinov3")
    weights_path = Path("checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    model = load_dinov3_backbone(
        dinov3_dir=dinov3_dir,
        weights_path=weights_path,
        device=device,
        sanity_input_size=512,
        verbose=True,
    )

    print(f"Model loaded on device: {device}")

    print("\n--- 4. Evaluating Model ---")
    report_LastLayer = evaluate_model(
        "DINOv3 Evaluation Task1 (n_layers=1)",
        model,
        test_loader,
        device,
        max_pairs=None,
        n_layers=1,
    )

    print_report(report_LastLayer, 1)
    print_per_category(report_LastLayer)

    reportLast2Layers = evaluate_model(
        "DINOv3 Evaluation Task1 (n_layers=2)",
        model,
        test_loader,
        device,
        max_pairs=None,
        n_layers=2,
    )

    print_report(reportLast2Layers, 1)
    print_per_category(reportLast2Layers)

    reportLast4Layers = evaluate_model(
        "DINOv3 Evaluation Task1 (n_layers=4)",
        model,
        test_loader,
        device,
        max_pairs=None,
        n_layers=4,
    )

    print_report(reportLast4Layers, 1)
    print_per_category(reportLast4Layers)

    print("\n--- Evaluation Complete ---")
