import os
import random
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.dataset_DINOv3 import SPairDataset, collate_single
from models.dinov3.model_DINOv3 import load_dinov3_backbone

from utils.setup_data_DINOv3 import setup_data
from Task4_DINOv3.utils_LoRA import make_peft_lora_model
from Task1_DINOv3.eval_Task1_DINOv3 import evaluate_model
from Task2_DINOv3.prepare_train import PATH_CHECKPOINTS
from utils.printing_helpers_DINOv3 import print_report, print_per_category

def load_lora_checkpoint(
    base_model,
    cfg: dict,
    ckpt_path: str | Path,
    device: torch.device,
):
    # Important: costruisci LoRA sul clone CPU, come nel training
    base_cpu = deepcopy(base_model).cpu().eval()

    lora_model = make_peft_lora_model(
        base_cpu,
        last_n_blocks=cfg["last_n_blocks"],
        r=cfg["r"],
        alpha=cfg["alpha"],
        dropout=cfg["dropout"],
        suffixes=cfg["suffixes"],
        verbose=False,
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    lora_model.load_state_dict(ckpt["model_state_dict"], strict=True)

    lora_model = lora_model.to(device).eval()
    return lora_model, ckpt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset root from your setup helper
    data_root = setup_data()
    if data_root is None:
        print("Dataset not found. Please follow README.md.")
        return

    base_dir = os.path.join(data_root, "SPair-71k")

    ds_test = SPairDataset(base_dir, split="test", layout_size="large", out_size=512, pad_mode="center", max_pairs=None)
    
    loader_test = DataLoader( ds_test, batch_size=1,shuffle=False, num_workers=4, collate_fn=collate_single, pin_memory=True, persistent_workers=True, drop_last=False,)

    # Load backbone once
    # (make these paths relative to repo in your actual project)
    dinov3_dir = Path("/content/dinov3") if Path("/content/dinov3").exists() else Path("third_party/dinov3")
    weights_path = Path("checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    model = load_dinov3_backbone(
        dinov3_dir=dinov3_dir,
        weights_path=weights_path,
        device=device,
        sanity_input_size=512,
        verbose=True,
    )
    n_blocks = len(model.blocks)
    print("DINOv3 blocks:", n_blocks)

    # LoRA config used during training
    LORA_CONFIGS = [
        dict(tag="lora_qkv_last1_r8", last_n_blocks=min(1,n_blocks), r=8, alpha=16, dropout=0.1, suffixes=("attn.qkv",)),
        dict(tag="lora_proj_last1_r8", last_n_blocks=min(1,n_blocks), r=8, alpha=16, dropout=0.1, suffixes=("attn.proj",)),
        dict(tag="lora_qkv_last4_r8", last_n_blocks=min(4,n_blocks), r=8, alpha=16, dropout=0.1, suffixes=("attn.qkv",)),
        dict(tag="lora_proj_last4_r8", last_n_blocks=min(4,n_blocks), r=8, alpha=16, dropout=0.1, suffixes=("attn.proj",)),
    ]

    for cfg in LORA_CONFIGS:
        tag = cfg["tag"]
        ckpt_path = Path(PATH_CHECKPOINTS) / f"best_model_LoRA_{tag}.pth"
        if not ckpt_path.exists():
            print(f"Checkpoint for LoRA config '{tag}' not found at {ckpt_path}. Skipping.")
            continue

        print(f"\n--- Evaluating LoRA model: {tag} ---")
        lora_model, ckpt = load_lora_checkpoint(model, cfg, ckpt_path, device)

        report = evaluate_model(
            name=f"Task4_LoRA_{tag}_TEST",
            model=lora_model,
            loader=loader_test,
            device=device,
            n_layers=1,
            thresholds=(0.05, 0.10, 0.20),
            max_pairs=None,
            verbose_every=0,
        )
        print_report(report, tag)
        print_per_category(report)

if __name__ == "__main__":
    main()  