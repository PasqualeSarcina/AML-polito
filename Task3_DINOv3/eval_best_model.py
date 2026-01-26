import os
import time
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dinov3.model_DINOv3 import load_dinov3_backbone
from dataset.dataset_DINOv3 import SPairDataset, collate_single
from utils.setup_data_DINOv3 import setup_data

from utils.printing_helpers_DINOv3 import print_report, print_per_category

from eval import evaluate_model


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
        state = ckpt  

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    msg = model.load_state_dict(state, strict=strict)
    return msg

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
        collate_fn=collate_single,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

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
    freeze_all(model)

    layers_to_eval = "Layer_1"
    ckpt_dir = Path("checkpoints/dinov3")

    wsa_window = 5
    wsa_temp = 0.1

    finetuned_model = deepcopy(model)
    freeze_all(finetuned_model)


    ckpt_path = ckpt_dir / f"best_model_{layers_to_eval}.pth"
    if not ckpt_path.exists():
        print(f"[SKIP] Missing checkpoint: {ckpt_path}")
        return

    msg = load_checkpoint_into_model( finetuned_model, ckpt_path, device, strict=True)
    print(f"\nLoaded {ckpt_path.name}: {msg}")

    r_best_argmax = evaluate_model(
        name="Task3_Best_Argmax",
        model=finetuned_model,
        loader=test_loader,
        device=device,
        n_layers=1,
        thresholds=(0.05, 0.10, 0.20),
        max_pairs=None,
        use_wsa=False,
    )
    print_report(r_best_argmax, task=3)
    print_per_category(r_best_argmax)

    r_best_wsa = evaluate_model(
        name=f"Task3_Best_WSA_w{wsa_window}_t{wsa_temp}",
        model=finetuned_model,
        loader=test_loader,
        device=device,
        n_layers=1,
        thresholds=(0.05, 0.10, 0.20),
        max_pairs=None,
        use_wsa=True,
        wsa_window=wsa_window,
        wsa_temp=wsa_temp,
    )
    print_report(r_best_wsa, task=3)
    print_per_category(r_best_wsa)

    print("\n--- Done ---")


if __name__ == "__main__":
    main()