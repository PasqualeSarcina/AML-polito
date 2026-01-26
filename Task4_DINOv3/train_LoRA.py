import os
import random
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.dataset_DINOv3 import SPairDataset, collate_spair
from models.dinov3.model_DINOv3 import load_dinov3_backbone
from Task2_DINOv3.prepare_train import (
    train_one_epoch,
    validate_one_epoch,
)
from utils.setup_data_DINOv3 import setup_data
from Task2_DINOv3.loss_DINOv3 import GaussianCrossEntropyLoss
from Task4_DINOv3.utils_LoRA import make_peft_lora_model, summarize_trainables, save_checkpoint


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> SEED SET TO {seed} <<<")


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

EPOCHS = 2

def finetune_LoRA(model, pretrained_state, loader_train, loader_val, device):
    n_blocks = len(model.blocks)

    LORA_CONFIGS = [
        dict(tag="lora_qkv_last1_r8", last_n_blocks=min(1,n_blocks), r=8, alpha=16, dropout=0.1, suffixes=("attn.qkv",)),
        dict(tag="lora_proj_last1_r8", last_n_blocks=min(1,n_blocks), r=8, alpha=16, dropout=0.1, suffixes=("attn.proj",)),
        dict(tag="lora_qkv_last4_r8", last_n_blocks=min(4,n_blocks), r=8, alpha=16, dropout=0.1, suffixes=("attn.qkv",)),
        dict(tag="lora_proj_last4_r8", last_n_blocks=min(4,n_blocks), r=8, alpha=16, dropout=0.1, suffixes=("attn.proj",)),
    ]

    all_hist = {}
    best_by_tag = {}

    loss_fn = GaussianCrossEntropyLoss(
            out_size=512,
            patch_size=16,
            temperature=0.2,
            sigma=1.0,
            window=7,
            use_windowed=True,
            enable_l2_norm=True,
        )
    
    for cfg in LORA_CONFIGS:
        tag = cfg["tag"]
        print("\n" + "="*60)
        print(f">>> TASK4 LoRA SCREENING: {tag}")
        print("="*60)

        model.load_state_dict(pretrained_state, strict=True)
        freeze_all(model)
        base_cpu_model = deepcopy(model).cpu().eval()


        lora_model = make_peft_lora_model(
            base_cpu_model,
            last_n_blocks=cfg["last_n_blocks"],
            r=cfg["r"],
            alpha=cfg["alpha"],
            dropout=cfg["dropout"],
            suffixes=cfg["suffixes"],)
        
        _ = summarize_trainables(lora_model)

        optimizer = torch.optim.AdamW(
            (p for p in lora_model.parameters() if p.requires_grad),
            lr=1e-3,
            weight_decay=1e-2,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        best_val_loss = float("inf")
        best_path = None
        history = []

        lora_model=lora_model.to(device)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=2, eta_min=1e-6)

        for epoch in range(1, EPOCHS + 1):
            lora_model.train()
            train_loss = train_one_epoch(
                lora_model,
                loader_train,
                optimizer,
                device,
                scaler,
                loss_fn,
                max_train_batches=200,
                n_layers_feats=1,
                grad_clip=1.0,
            )   

            lora_model.eval()

            val_res = validate_one_epoch(
                lora_model,
                loader_val,
                device,
                loss_fn,
                max_val_batches=300,
                n_layers_feats=1,
            )

            lr_current = optimizer.param_groups[0]['lr']
            val_loss=float(val_res["val_loss"])

            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": lr_current,
            })
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr_current:.6f}")    
            
            scheduler.step()

            # Checkpointing
            is_best = float(val_loss) < best_val_loss
            if is_best:
                best_val_loss = float(val_loss)
                saved_path = save_checkpoint(
                    lora_model,
                    optimizer,
                    epoch,
                    tag,
                    is_best=is_best,
                    val_loss=best_val_loss,
                    scaler=scaler,
                    hparams=cfg,
                )

                if saved_path is not None:
                    best_path = saved_path

        all_hist[tag] = history
        best_by_tag[tag] = best_path
    print("\n=== TRAINING COMPLETE ===")
    print("Best checkpoints by LoRA config:")
    for tag, path in best_by_tag.items():
        print(f"  {tag}: {path}")
    return all_hist, best_by_tag

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset root from your setup helper
    data_root = setup_data()
    if data_root is None:
        print("Dataset not found. Please follow README.md.")
        return

    base_dir = os.path.join(data_root, "SPair-71k")

    ds_train = SPairDataset(base_dir, split="trn", layout_size="large", out_size=512, pad_mode="center", max_pairs=None)
    ds_val   = SPairDataset(base_dir, split="val", layout_size="large", out_size=512, pad_mode="center", max_pairs=None)

    loader_train = DataLoader(ds_train, batch_size=8, shuffle=True,  num_workers=0, drop_last=True,  collate_fn=collate_spair)
    loader_val   = DataLoader(ds_val,   batch_size=1,   shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_spair)

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
    pretrained_state = deepcopy(model.state_dict())
    finetune_LoRA(model, pretrained_state, loader_train, loader_val, device)
if __name__ == "__main__":
    main()  