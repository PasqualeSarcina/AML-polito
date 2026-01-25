import os
import time
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model_DINOv3 import load_dinov3_backbone
from dataset.dataset_DINOv3 import SPairDataset, collate_spair  
from utils.setup_data_DINOv3 import setup_data

from utils.extraction_DINOv3 import extract_dense_features
from utils.matching_DINOv3 import match_wsa_nearest_patch_masked


from utils.printing_helpers_DINOv3 import print_report, print_per_category


# ----------------------------
# Evaluation (global + per-category, micro + macro)
# ----------------------------
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
    wsa_window: int = 5,
    wsa_temp: float = 0.1,
    verbose_every: int = 0,   # 0 = no periodic prints
):
    model.eval()

    # Robust patch size retrieval
    patch = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if patch is None:
        patch = getattr(model, "patch_size", 16)
    patch = int(patch[0]) if isinstance(patch, (tuple, list)) else int(patch)

    # Global accumulators
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

    # tqdm "total" should match what we actually iterate
    total_bar = max_pairs if max_pairs is not None else len(loader)

    pbar = tqdm(loader, total=total_bar, desc=f"{name}", leave=True)

    for i, batch in enumerate(pbar):
        if max_pairs is not None and pairs_seen >= max_pairs:
            break

        # collate_spair returns a batch dict; with batch_size=1 many fields are length-1 lists/tensors.
        # We normalize to "single sample" by indexing [0] where appropriate.
        # Images: [B,3,H,W]
        src_img = batch["src_img"].to(device)
        trg_img = batch["trg_img"].to(device)

        # Keypoints: usually [B,K,2]
        src_kps = batch["src_kps"].to(device)
        trg_kps = batch["trg_kps"].to(device)

        # BBox: [B,4]
        trg_bbox = batch["trg_bbox"].to(device)

        # Meta: list[dict] length B
        src_meta = batch["src_meta"][0] if isinstance(batch["src_meta"], (list, tuple)) else batch["src_meta"]
        trg_meta = batch["trg_meta"][0] if isinstance(batch["trg_meta"], (list, tuple)) else batch["trg_meta"]

        # Category: list[str] length B
        cat = batch.get("category", "unknown")
        if isinstance(cat, (list, tuple)):
            cat = cat[0]
        elif torch.is_tensor(cat):
            # very rare; keep safe
            cat = str(cat.item())

        pair_id = batch.get("pair_id", f"pair_{i}")
        if isinstance(pair_id, (list, tuple)):
            pair_id = pair_id[0]

        # Squeeze batch dimension (we expect B=1)
        src_img = src_img.squeeze(0)  # [3,H,W]
        trg_img = trg_img.squeeze(0)
        src_kps = src_kps.squeeze(0)  # [K,2]
        trg_kps = trg_kps.squeeze(0)
        trg_bbox = trg_bbox.squeeze(0)  # [4]

        out_size = int(src_img.shape[-1])  # square by construction

        # --- Extract dense features (returns [1,C,Hf,Wf]) ---
        src_feat = extract_dense_features(model, src_img, n_layers=n_layers, return_grid=False)
        trg_feat = extract_dense_features(model, trg_img, n_layers=n_layers, return_grid=False)

        # --- Predict with masked WSA matcher ---
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

        # --- Validity: GT in-bounds (both) + matcher-valid ---
        img_w = out_size
        img_h = out_size
        src_valid = (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) & (src_kps[:, 0] < img_w) & (src_kps[:, 1] < img_h)
        trg_valid = (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0) & (trg_kps[:, 0] < img_w) & (trg_kps[:, 1] < img_h)
        valid = src_valid & trg_valid & valid_src_mask

        if valid.sum().item() == 0:
            continue

        pairs_seen += 1
        kv = int(valid.sum().item())
        total_valid_kps += kv

        pbar.set_postfix({"pairs": pairs_seen, "valid_kps": total_valid_kps})

        # --- PCK normalization (target bbox, xyxy) ---
        w = (trg_bbox[2] - trg_bbox[0]).clamp(min=1.0)
        h = (trg_bbox[3] - trg_bbox[1]).clamp(min=1.0)
        norm = float(torch.max(w, h).item())

        # Distances for valid keypoints
        d = torch.norm(pred_kps[valid] - trg_kps[valid], dim=1)  # [Kv]

        # Per-pair accuracies (macro = mean over pairs)
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

    # --- Build report dict (PCK values as fractions in [0,1]) ---
    report = {
        "name": name,
        "n_layers": n_layers,
        "pairs_evaluated": pairs_seen,
        "valid_keypoints": total_valid_kps,
        "patch_size": patch,
        "thresholds": list(thresholds),
        "wsa_window": int(wsa_window),
        "wsa_temp": float(wsa_temp),
        "global_pck_micro": {},
        "global_pck_macro": {},
        "per_category": {},
    }

    for t in thresholds:
        report["global_pck_micro"][t] = (micro_correct[t] / micro_total[t]) if micro_total[t] > 0 else 0.0
        report["global_pck_macro"][t] = (macro_sum[t] / macro_n[t]) if macro_n[t] > 0 else 0.0

    cats = sorted(set(list(cat_micro_total.keys()) + list(cat_macro_n.keys())))
    for cat in cats:
        entry = {
            "pck_micro": {},
            "pck_macro": {},
            "pairs": int(cat_macro_n[cat][thresholds[0]]) if thresholds else 0,
            "valid_kps": int(cat_micro_total[cat][thresholds[0]]) if thresholds else 0,
        }
        for t in thresholds:
            tot = cat_micro_total[cat][t]
            entry["pck_micro"][t] = (cat_micro_correct[cat][t] / tot) if tot > 0 else 0.0
            n = cat_macro_n[cat][t]
            entry["pck_macro"][t] = (cat_macro_sum[cat][t] / n) if n > 0 else 0.0
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
        collate_fn=collate_spair,
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

    # You can change these to match exactly the settings you want to report
    wsa_window = 5
    wsa_temp = 0.1

    for n_layers in (1, 2, 4):
        report = evaluate_model(
            name=f"DINOv3 Base Eval (n_layers={n_layers})",
            model=model,
            loader=test_loader,
            device=device,
            n_layers=n_layers,
            max_pairs=None,
            wsa_window=wsa_window,
            wsa_temp=wsa_temp,
        )
        print_report(report)
        print_per_category(report)

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main()
