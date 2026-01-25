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
# Utils
# ----------------------------
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

def load_checkpoint_into_model(model, ckpt_path: Path, device: torch.device, *, strict: bool = True):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt  # assume raw state_dict

    # common safety if trained with DataParallel
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    msg = model.load_state_dict(state, strict=strict)
    return msg


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
):
    model.eval()

    # patch size
    patch = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if patch is None:
        patch = getattr(model, "patch_size", 16)
    patch = int(patch[0]) if isinstance(patch, (tuple, list)) else int(patch)

    total_valid_kps = 0
    micro_correct = {t: 0 for t in thresholds}
    micro_total   = {t: 0 for t in thresholds}
    macro_sum = {t: 0.0 for t in thresholds}
    macro_n   = {t: 0 for t in thresholds}

    cat_micro_correct = defaultdict(lambda: {t: 0 for t in thresholds})
    cat_micro_total   = defaultdict(lambda: {t: 0 for t in thresholds})
    cat_macro_sum     = defaultdict(lambda: {t: 0.0 for t in thresholds})
    cat_macro_n       = defaultdict(lambda: {t: 0 for t in thresholds})

    pairs_seen = 0
    total_bar = max_pairs if max_pairs is not None else len(loader)
    pbar = tqdm(loader, total=total_bar, desc=name, leave=True)

    for i, batch in enumerate(pbar):
        if max_pairs is not None and pairs_seen >= max_pairs:
            break

        # batch_size=1
        src_img = batch["src_img"].to(device).squeeze(0)   # [3,H,W]
        trg_img = batch["trg_img"].to(device).squeeze(0)
        src_kps = batch["src_kps"].to(device).squeeze(0)   # [K,2]
        trg_kps = batch["trg_kps"].to(device).squeeze(0)
        trg_bbox = batch["trg_bbox"].to(device).squeeze(0) # [4]

        src_meta = batch["src_meta"][0] if isinstance(batch["src_meta"], (list, tuple)) else batch["src_meta"]
        trg_meta = batch["trg_meta"][0] if isinstance(batch["trg_meta"], (list, tuple)) else batch["trg_meta"]

        cat = batch.get("category", "unknown")
        if isinstance(cat, (list, tuple)):
            cat = cat[0]

        out_size = int(src_img.shape[-1])

        # features
        src_feat = extract_dense_features(model, src_img, n_layers=n_layers, return_grid=False)
        trg_feat = extract_dense_features(model, trg_img, n_layers=n_layers, return_grid=False)

        # predictions (masked WSA)
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

        # GT in-bounds (both) + matcher-valid
        img_w = out_size
        img_h = out_size
        src_valid = (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) & (src_kps[:, 0] < img_w) & (src_kps[:, 1] < img_h)
        trg_valid = (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0) & (trg_kps[:, 0] < img_w) & (trg_kps[:, 1] < img_h)
        valid = src_valid & trg_valid & valid_src_mask

        if valid.sum().item() == 0:
            continue

        pairs_seen += 1
        total_valid_kps += int(valid.sum().item())
        pbar.set_postfix({"pairs": pairs_seen, "valid_kps": total_valid_kps})

        # PCK norm (target bbox)
        w = (trg_bbox[2] - trg_bbox[0]).clamp(min=1.0)
        h = (trg_bbox[3] - trg_bbox[1]).clamp(min=1.0)
        norm = float(torch.max(w, h).item())

        d = torch.norm(pred_kps[valid] - trg_kps[valid], dim=1)

        for t in thresholds:
            thr = t * norm
            correct = (d <= thr).float()
            pair_acc = float(correct.mean().item())

            c = int(correct.sum().item())
            n = int(correct.numel())

            micro_correct[t] += c
            micro_total[t] += n
            macro_sum[t] += pair_acc
            macro_n[t] += 1

            cat_micro_correct[cat][t] += c
            cat_micro_total[cat][t] += n
            cat_macro_sum[cat][t] += pair_acc
            cat_macro_n[cat][t] += 1

    report = {
        "name": name,
        "n_layers": n_layers,
        "pairs_evaluated": pairs_seen,
        "valid_keypoints": total_valid_kps,
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
        entry = {"pck_micro": {}, "pck_macro": {}}
        for t in thresholds:
            tot = cat_micro_total[cat][t]
            entry["pck_micro"][t] = (cat_micro_correct[cat][t] / tot) if tot > 0 else 0.0
            n = cat_macro_n[cat][t]
            entry["pck_macro"][t] = (cat_macro_sum[cat][t] / n) if n > 0 else 0.0
        report["per_category"][cat] = entry

    return report


# ----------------------------
# Main
# ----------------------------
def main():
    data_root = setup_data()
    if data_root is None:
        print("Dataset not found. Please prepare SPair-71k.")
        return

    base_dir = Path(data_root) / "SPair-71k"

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dinov3_dir = Path("/content/dinov3") if Path("/content/dinov3").exists() else Path("third_party/dinov3")
    weights_path = Path("checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")

    # Build the backbone once (same architecture), then load different checkpoints into it
    model = load_dinov3_backbone(
        dinov3_dir=dinov3_dir,
        weights_path=weights_path,
        device=device,
        sanity_input_size=512,
        verbose=True,
    )
    freeze_all(model)

    # Evaluate these checkpoints:
    layers_to_eval = ["Layer_1", "Layer_2", "Layer_4"]
    ckpt_dir = Path("checkpoints/dinov3")

    # Matching settings (keep consistent with your Task3 base eval)
    wsa_window = 5
    wsa_temp = 0.1

    for layer_name in layers_to_eval:
        ckpt_path = ckpt_dir / f"best_model_{layer_name}.pth"
        if not ckpt_path.exists():
            print(f"[SKIP] Missing checkpoint: {ckpt_path}")
            continue

        # Load weights for this layer checkpoint
        msg = load_checkpoint_into_model(model, ckpt_path, device, strict=True)
        print(f"\nLoaded {ckpt_path.name}: {msg}")

        # Evaluate (same metric as base eval)
        report = evaluate_model(
            name=f"DINOv3 FINETUNED Eval ({layer_name}, n_layers={1})",
            model=model,
            loader=test_loader,
            device=device,
            n_layers=1,          # keep consistent with how you extract feats (last 1 layer); change if you want 2/4
            max_pairs=None,
            wsa_window=wsa_window,
            wsa_temp=wsa_temp,
        )

        print_report(report)
        print_per_category(report)

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
