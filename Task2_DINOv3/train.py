import os
import random
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.dataset_DINOv3 import SPairDataset, collate_spair
from models.dinov3.model_DINOv3 import load_dinov3_backbone
from Task2_DINOv3.prepare_train import (
    prepare_model_for_fine_tuning,
    train_one_epoch,
    validate_one_epoch,
    save_checkpoint,
    PATH_CHECKPOINTS,  # IMPORTANT: single source of truth
)
from utils.setup_data_DINOv3 import setup_data
from Task2_DINOv3.loss_DINOv3 import InfoNCEPatchClassifyLoss


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> SEED SET TO {seed} <<<")

@dataclass
class FinetuneScreeningResult:
    layer_name: str
    best_epoch: int
    best_val_loss: float
    history: dict
    best_ckpt_path: str | None
    
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

def finetune_screening(
    model,
    pretrained_state,
    loader_train,
    loader_val,
    device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    n_layer: int,
) -> FinetuneScreeningResult:

    layer_name = f"Layer_{n_layer}"
    os.makedirs(PATH_CHECKPOINTS, exist_ok=True)
    best_ckpt_path = os.path.join(PATH_CHECKPOINTS, f"best_model_{layer_name}.pth")

    model.load_state_dict(pretrained_state, strict=True)
    model.to(device)
    freeze_all(model)

    loss_fn = InfoNCEPatchClassifyLoss(out_size=512, patch_size=16, temperature=0.07)
    
    prepare_model_for_fine_tuning(
        model,
        num_layers_to_unfreeze=n_layer,
        unfreeze_final_norm=True
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )


    scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss, best_epoch = float("inf"), -1

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loader_train,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            loss_fn=loss_fn,
            max_train_batches=None,
            n_layers_feats=1,
            grad_clip=1.0,
        )

        val_res = validate_one_epoch(
            model=model,
            loader=loader_val,
            device=device,
            loss_fn=loss_fn,
            max_val_batches=None,
            n_layers_feats=1,     
        )

        current_lr = optimizer.param_groups[0]['lr']  
        history["lr"].append(current_lr)
        scheduler.step()
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_res["val_loss"]))


        print(
            f"Epoch {epoch}: "
            f"Train Loss {float(train_loss):.4f} | "
            f"Val Loss {float(val_res['val_loss']):.4f} | "
        )

        if float(val_res['val_loss']) < best_val_loss:
            best_val_loss = float(val_res["val_loss"])
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=best_epoch,
                layer_name=layer_name,
                is_best=True,
                scaler=scaler,
                hparams={
                    "epochs": epochs,
                    "lr": current_lr,
                    "sigma": 1.0,
                    "temperature": 0.7,
                    "weight_decay": weight_decay,
                    "n_layer": n_layer,
                    "out_size": 512,
                    "patch_size": 16,
                    "n_layers_feats": 1,
                    "best_val_loss": best_val_loss,
                },
            )

    if not os.path.exists(best_ckpt_path):
        best_ckpt_path = None

    return FinetuneScreeningResult(layer_name, best_epoch, best_val_loss, history, best_ckpt_path)
# -----------------------------
# Main
# -----------------------------
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

    # snapshot pretrained once
    pretrained_state = deepcopy(model.state_dict())

    best_val_loss = float("inf")
    best_epoch = -1
    best_tag = ""

    r = finetune_screening(
            model,
            pretrained_state,
            loader_train,
            loader_val,
            device,
            epochs=2,
            lr=1e-3,
            weight_decay=1e-2,
            n_layer=1,
        )

    if r.best_val_loss < best_val_loss:
        best_val_loss = r.best_val_loss
        best_epoch = r.best_epoch
        best_tag = r.layer_name

    print(f"\n>>> BEST CONFIG: {best_tag} | Val loss={best_val_loss:.2f} - epoch {best_epoch} <<<\n")

if __name__ == "__main__":
    main()
