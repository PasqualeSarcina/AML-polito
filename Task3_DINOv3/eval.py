import os
import time
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dinov3.model_DINOv3 import load_dinov3_backbone
from dataset.dataset_DINOv3 import SPairDataset, collate_single  
from utils.setup_data_DINOv3 import setup_data

from utils.extraction_DINOv3 import extract_dense_features
from utils.matching_DINOv3 import match_wsa_nearest_patch_masked, match_argmax_nearest_patch_masked


from utils.printing_helpers_DINOv3 import print_report, print_per_category

@torch.no_grad()
def evaluate_model(
    name: str,
    model,
    loader,
    device,
    *,
    n_layers: int = 1,
    thresholds=(0.05, 0.10, 0.20),
    max_pairs: int | None = None,
    verbose_every: int = 0,   
    use_wsa: bool = False,
    wsa_window: int = 5,
    wsa_temp: float = 0.1,
):

    model.eval()

    # Robust patch size retrieval
    patch = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if patch is None:
        patch = getattr(model, "patch_size", 16)
    patch = int(patch[0]) if isinstance(patch, (tuple, list)) else int(patch)

    # Accumulators
    total_valid_kps = 0
    micro_correct = {t: 0 for t in thresholds}
    micro_total   = {t: 0 for t in thresholds}

    macro_sum = {t: 0.0 for t in thresholds}
    macro_n   = {t: 0 for t in thresholds}

    # Per-category accumulators
    cat_micro_correct = defaultdict(lambda: {t: 0 for t in thresholds})
    cat_micro_total   = defaultdict(lambda: {t: 0 for t in thresholds})
    cat_macro_sum     = defaultdict(lambda: {t: 0.0 for t in thresholds})
    cat_macro_n       = defaultdict(lambda: {t: 0 for t in thresholds})

    t0 = time.time()
    pairs_seen = 0

    pbar_total = max_pairs if max_pairs is not None else None
    pbar = tqdm(loader, total=pbar_total, desc=f"{name}", leave=True)

    for i, sample in enumerate(pbar):
        if max_pairs is not None and pairs_seen >= max_pairs:
            break

        cat = sample.get("category", "unknown")
        pair_id = sample.get("pair_id", f"pair_{i}")

        src_img  = sample["src_img"].to(device)      # [C,H,W] or [1,C,H,W]
        trg_img  = sample["trg_img"].to(device)
        src_kps  = sample["src_kps"].to(device)      # [K,2]
        trg_kps  = sample["trg_kps"].to(device)      # [K,2]
        trg_bbox = sample["trg_bbox"].to(device)     # [4] xyxy in preprocess coords

        # meta dicts stay on CPU
        src_meta = sample["src_meta"]
        trg_meta = sample["trg_meta"]

        # Ensure CHW for extractor
        src_chw = src_img.squeeze(0) if src_img.ndim == 4 else src_img
        trg_chw = trg_img.squeeze(0) if trg_img.ndim == 4 else trg_img

        out_size = int(src_chw.shape[-1])  # square
        img_w = out_size
        img_h = out_size

        # Keypoints valid in both images (in preprocess coords)
        src_valid = (
            (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) &
            (src_kps[:, 0] < img_w) & (src_kps[:, 1] < img_h)
        )
        trg_valid = (
            (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0) &
            (trg_kps[:, 0] < img_w) & (trg_kps[:, 1] < img_h)
        )
        valid = src_valid & trg_valid
        if int(valid.sum().item()) == 0:
            continue

        # Extract dense features
        src_feat = extract_dense_features(model, src_chw, n_layers=n_layers, return_grid=False)  # [1,C,Hf,Wf]
        trg_feat = extract_dense_features(model, trg_chw, n_layers=n_layers, return_grid=False)

        # Predict target keypoints
        if use_wsa:
            pred_kps, valid_src_mask = match_wsa_nearest_patch_masked(
                src_feat=src_feat,
                trg_feat=trg_feat,
                src_kps_xy=src_kps,
                out_size=out_size,
                patch=patch,
                src_meta=src_meta,
                trg_meta=trg_meta,
                wsa_window=wsa_window,
                wsa_temp=wsa_temp,
            )
        else:
            pred_kps, valid_src_mask = match_argmax_nearest_patch_masked(
                src_feat=src_feat,
                trg_feat=trg_feat,
                src_kps_xy=src_kps,
                out_size=out_size,
                patch=patch,
                src_meta=src_meta,
                trg_meta=trg_meta,
            )

        # Score only keypoints valid in GT and matchable (e.g., not in padded area)
        valid = valid & valid_src_mask
        if int(valid.sum().item()) == 0:
            continue

        pairs_seen += 1
        total_valid_kps += int(valid.sum().item())
        pbar.set_postfix({"pairs": pairs_seen, "valid_kps": total_valid_kps})

        # PCK normalization (target bbox max side)
        w = (trg_bbox[2] - trg_bbox[0]).clamp(min=1.0)
        h = (trg_bbox[3] - trg_bbox[1]).clamp(min=1.0)
        norm = torch.max(w, h)

        # Distances for valid keypoints
        d = torch.norm(pred_kps[valid] - trg_kps[valid], dim=1)  # [Kv]

        for t in thresholds:
            thr = float(t) * norm
            correct = (d <= thr).float()

            c = int(correct.sum().item())
            n = int(correct.numel())
            pair_acc = float(correct.mean().item())

            micro_correct[t] += c
            micro_total[t] += n

            macro_sum[t] += pair_acc
            macro_n[t] += 1

            cat_micro_correct[cat][t] += c
            cat_micro_total[cat][t] += n
            cat_macro_sum[cat][t] += pair_acc
            cat_macro_n[cat][t] += 1

        if verbose_every and (pairs_seen % verbose_every == 0):
            elapsed = time.time() - t0
            print(
                f"[{name}] pairs={pairs_seen} valid_kps={total_valid_kps} "
                f"elapsed={elapsed:.1f}s last_pair={pair_id} "
                f"mode={'WSA' if use_wsa else 'ARGMAX'}"
            )

    # Build report
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
        "inference_mode": "WSA" if use_wsa else "ARGMAX",
    }
    if use_wsa:
        report["wsa_window"] = int(wsa_window)
        report["wsa_temp"] = float(wsa_temp)

    for t in thresholds:
        report["global_pck_micro"][t] = (micro_correct[t] / micro_total[t]) if micro_total[t] > 0 else 0.0
        report["global_pck_macro"][t] = (macro_sum[t] / macro_n[t]) if macro_n[t] > 0 else 0.0

    cats = sorted(set(list(cat_micro_total.keys()) + list(cat_macro_n.keys())))
    for cat in cats:
        entry = {
            "pck_micro": {},
            "pck_macro": {},
            # these are essentially constant across thresholds; take max for safety
            "pairs": int(max(cat_macro_n[cat].values()) if len(cat_macro_n[cat]) else 0),
            "valid_kps": int(max(cat_micro_total[cat].values()) if len(cat_micro_total[cat]) else 0),
        }
        for t in thresholds:
            tot = cat_micro_total[cat][t]
            entry["pck_micro"][t] = (cat_micro_correct[cat][t] / tot) if tot > 0 else 0.0
            nn = cat_macro_n[cat][t]
            entry["pck_macro"][t] = (cat_macro_sum[cat][t] / nn) if nn > 0 else 0.0
        report["per_category"][cat] = entry

    return report



# -----------------------------
# Main
# -----------------------------
def main():
    print("--- 1. Checking Data Availability ---")
    data_root = setup_data()
    if data_root is None:
        print("Dataset not found. Please follow the instructions in README.md to download and prepare SPair-71k.")
        return

    base_dir = Path(data_root) / "SPair-71k"
    pair_ann_path = base_dir / "PairAnnotation"
    layout_path   = base_dir / "Layout"
    image_path    = base_dir / "JPEGImages"
    if not (pair_ann_path.exists() and layout_path.exists() and image_path.exists()):
        print("Dataset structure is incorrect. Expected PairAnnotation/Layout/JPEGImages under SPair-71k.")
        return

    print("\n--- 2. Loading Test Dataset ---")
    test_dataset = SPairDataset(
        spair_root=base_dir,
        split="test",
        layout_size="large",
        out_size=512,
        pad_mode="center",
        max_pairs=None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_single,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
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

    wsa_window = 5
    wsa_temp = 0.1

    
    model.eval()

    r_base_argmax = evaluate_model(
        name="Task3_Base_Argmax",
        model=model,
        loader=test_loader,
        device=device,
        n_layers=1,
        thresholds=(0.05, 0.10, 0.20),
        max_pairs=None,
        use_wsa=False,
    )
    print_report(r_base_argmax, task=3)
    print_per_category(r_base_argmax)

    r_base_wsa = evaluate_model(
        name=f"Task3_Base_WSA_w{wsa_window}_t{wsa_temp}",
        model=model,
        loader=test_loader,
        device=device,
        n_layers=1,
        thresholds=(0.05, 0.10, 0.20),
        max_pairs=None,
        use_wsa=True,
        wsa_window=wsa_window,
        wsa_temp=wsa_temp,
    )
    print_report(r_base_wsa, task=3)
    print_per_category(r_base_wsa)


if __name__ == "__main__":
    main()
