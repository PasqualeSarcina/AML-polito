from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dinov3.PreProcess import PreProcess
from models.dinov3.model_DINOv3 import load_dinov3_backbone
from models.dinov3.lora import (
    apply_lora_to_dinov3,
    count_parameters,
    get_lora_state_dict,
    merge_lora_weights,
)
from utils.loss import InfoNCELoss
from utils.utils_init_dataloader import init_dataloader
from utils.utils_train import collate_batch, seed_everything


# =============================================================================
# Argument parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of DINOv3 for semantic correspondence."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="spair-71k",
        choices=["spair-71k", "pf-pascal", "pf-willow", "ap-10k"],
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-target-modules", type=str, default="qkv")
    parser.add_argument("--train-norm", action="store_true")
    parser.add_argument("--save-merged-state-dict", action="store_true")

    return parser


# =============================================================================
# Data
# =============================================================================

def setup_data(args, base_dir: Path):
    transform = PreProcess(out_dim=(args.input_size, args.input_size))

    train_dataset, _ = init_dataloader(
        dataset_name=args.dataset,
        base_dir=base_dir,
        datatype="train",
        transform=transform,
        num_workers=args.num_workers,
    )

    val_dataset, _ = init_dataloader(
        dataset_name=args.dataset,
        base_dir=base_dir,
        datatype="val",
        transform=transform,
        num_workers=args.num_workers,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Training pairs: {len(train_dataset)}")
    print(f"Validation pairs: {len(val_dataset)}")

    return train_loader, val_loader


# =============================================================================
# Model
# =============================================================================

def setup_model(args, base_dir: Path, device: torch.device):
    dinov3_dir = base_dir / "third_party" / "dinov3"

    weights_path = (
        base_dir
        / "checkpoints"
        / "dinov3"
        / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )

    if not weights_path.exists():
        weights_path = (
            base_dir
            / "checkpoints"
            / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"DINOv3 pretrained checkpoint not found: {weights_path}"
        )

    model = load_dinov3_backbone(
        dinov3_dir=dinov3_dir,
        weights_path=weights_path,
        device=device,
        sanity_input_size=args.input_size,
        verbose=True,
    )

    lora_target_modules = tuple(
        x.strip()
        for x in args.lora_target_modules.split(",")
        if x.strip()
    )

    lora_summary = apply_lora_to_dinov3(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        train_norm=args.train_norm,
    )

    model.to(device)

    print(f"Model loaded on device: {device}")
    print(f"LoRA targets: {lora_summary.target_modules}")
    print(f"LoRA replaced modules: {len(lora_summary.replaced_modules)}")
    print(
        "Trainable parameters: "
        f"{lora_summary.trainable_parameters:,} / {lora_summary.total_parameters:,}"
    )

    return model, lora_summary


# =============================================================================
# Optimization
# =============================================================================

def setup_optimization(args, model, device: torch.device):
    criterion = InfoNCELoss(
        temperature=args.temperature,
        patch_size=args.patch_size,
    ).to(device)

    trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found after LoRA setup.")

    optimizer = optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    return criterion, optimizer, scheduler, scaler


# =============================================================================
# Feature utilities
# =============================================================================

def tokens_to_featuremap(
    tokens: torch.Tensor,
    img_tensor: torch.Tensor,
    input_size: int,
    patch_size: int,
) -> torch.Tensor:
    """
    Convert DINOv3 patch tokens from [B, N, C] to [B, C, H_grid, W_grid].
    """

    B, N, C = tokens.shape

    H_img = img_tensor.shape[-2]
    W_img = img_tensor.shape[-1]

    if H_img != input_size or W_img != input_size:
        raise RuntimeError(
            f"DINOv3 LoRA training expects input size "
            f"{input_size}x{input_size}, but got {H_img}x{W_img}."
        )

    H_grid = H_img // patch_size
    W_grid = W_img // patch_size

    expected_tokens = H_grid * W_grid

    if N < expected_tokens:
        raise RuntimeError(
            f"Not enough patch tokens: got {N}, expected {expected_tokens}."
        )

    patch_tokens = tokens[:, -expected_tokens:, :]

    feature_map = patch_tokens.permute(0, 2, 1).reshape(
        B,
        C,
        H_grid,
        W_grid,
    )

    return feature_map


def forward_features(
    model,
    src_img: torch.Tensor,
    trg_img: torch.Tensor,
    input_size: int,
    patch_size: int,
):
    """
    Forward source and target images together.

    src_img: [B, 3, H, W]
    trg_img: [B, 3, H, W]
    """

    batch_size = src_img.shape[0]

    imgs = torch.cat([src_img, trg_img], dim=0)

    output = model.forward_features(imgs)

    if "x_norm_patchtokens" not in output:
        raise RuntimeError(
            "DINOv3 forward_features output does not contain 'x_norm_patchtokens'."
        )

    tokens = output["x_norm_patchtokens"]

    feat_all = tokens_to_featuremap(
        tokens=tokens,
        img_tensor=imgs,
        input_size=input_size,
        patch_size=patch_size,
    )

    feat_src = feat_all[:batch_size]
    feat_trg = feat_all[batch_size:]

    return feat_src, feat_trg


# =============================================================================
# Batch utilities
# =============================================================================

def prepare_batch(batch: dict, device: torch.device):
    src_img = batch["src_img"].to(device, non_blocking=True)
    trg_img = batch["trg_img"].to(device, non_blocking=True)

    src_kps = batch["src_kps"].to(device, non_blocking=True)
    trg_kps = batch["trg_kps"].to(device, non_blocking=True)

    valid_mask = batch["valid_mask"].to(
        device,
        non_blocking=True,
        dtype=torch.float32,
    )

    return src_img, trg_img, src_kps, trg_kps, valid_mask


# =============================================================================
# Train / validation
# =============================================================================

def train_one_epoch(
    *,
    epoch: int,
    args,
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{args.epochs} [LoRA Training]",
    )

    for step, batch in enumerate(pbar, start=1):
        src_img, trg_img, src_kps, trg_kps, valid_mask = prepare_batch(
            batch,
            device,
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            feat_src, feat_trg = forward_features(
                model=model,
                src_img=src_img,
                trg_img=trg_img,
                input_size=args.input_size,
                patch_size=args.patch_size,
            )

            loss = criterion(
                feat_src,
                feat_trg,
                src_kps,
                trg_kps,
                valid_mask,
            )

            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_value = loss.item() * args.accumulation_steps
        total_loss += loss_value

        pbar.set_postfix({"loss": f"{loss_value:.4f}"})

    return total_loss / len(train_loader)


def validate_one_epoch(
    *,
    epoch: int,
    args,
    model,
    val_loader,
    criterion,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch}/{args.epochs} [LoRA Validation]",
    )

    with torch.no_grad():
        for batch in pbar:
            src_img, trg_img, src_kps, trg_kps, valid_mask = prepare_batch(
                batch,
                device,
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                feat_src, feat_trg = forward_features(
                    model=model,
                    src_img=src_img,
                    trg_img=trg_img,
                    input_size=args.input_size,
                    patch_size=args.patch_size,
                )

                loss = criterion(
                    feat_src,
                    feat_trg,
                    src_kps,
                    trg_kps,
                    valid_mask,
                )

            loss_value = loss.item()
            total_loss += loss_value

            pbar.set_postfix({"loss": f"{loss_value:.4f}"})

    return total_loss / len(val_loader)


# =============================================================================
# Checkpoint
# =============================================================================

def build_checkpoint_paths(args, base_dir: Path):
    lr_str = f"{args.lr:.0e}".replace("+", "")
    wd_str = f"{args.weight_decay:.0e}".replace("+", "")

    dropout_str = str(args.lora_dropout).replace(".", "p")
    targets_str = args.lora_target_modules.replace(",", "-").replace(".", "_")

    norm_str = "norm" if args.train_norm else "nonorm"

    checkpoint_name = (
        f"task4_lora_"
        f"bs{args.batch_size}_"
        f"acc{args.accumulation_steps}_"
        f"r{args.lora_r}_"
        f"alpha{args.lora_alpha}_"
        f"drop{dropout_str}_"
        f"targets{targets_str}_"
        f"{norm_str}_"
        f"lr{lr_str}_"
        f"wd{wd_str}_"
        f"best_model.pth"
    )

    output_dir = base_dir / "checkpoints" / "dinov3"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / checkpoint_name

    adapter_path = (
        checkpoint_path.parent
        / checkpoint_path.name.replace(".pth", "_adapter.pth")
    )

    merged_state_dict_path = (
        checkpoint_path.parent
        / checkpoint_path.name.replace(".pth", "_merged_state_dict.pth")
    )

    return checkpoint_path, adapter_path, merged_state_dict_path


def save_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    checkpoint_path: Path,
    adapter_path: Path,
    merged_state_dict_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    args,
) -> None:
    trainable, total = count_parameters(model)

    adapter_checkpoint = {
        "epoch": epoch,
        "lora_state_dict": get_lora_state_dict(model),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "config": {
            "dataset": args.dataset,
            "input_size": args.input_size,
            "patch_size": args.patch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "accumulation_steps": args.accumulation_steps,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
            "train_norm": args.train_norm,
            "trainable_parameters": trainable,
            "total_parameters": total,
        },
    }

    full_checkpoint = {
        **adapter_checkpoint,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    torch.save(full_checkpoint, checkpoint_path)
    torch.save(adapter_checkpoint, adapter_path)

    print(f"Full LoRA checkpoint saved: {checkpoint_path}")
    print(f"LoRA adapter checkpoint saved: {adapter_path}")

    if args.save_merged_state_dict:
        model_for_save = copy.deepcopy(model).cpu().eval()
        merge_lora_weights(model_for_save)
        torch.save(model_for_save.state_dict(), merged_state_dict_path)
        del model_for_save
        print(f"Merged DINOv3 state_dict saved: {merged_state_dict_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()

    using_colab = os.getenv("COLAB_RELEASE_TAG")

    if using_colab:
        base_dir = Path(os.path.abspath(os.path.curdir)) / "AML-polito"
    else:
        base_dir = Path(os.path.abspath(os.path.curdir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting DINOv3 LoRA training...")
    print("Using Google Colab:", using_colab)
    print("Base directory set to:", base_dir)
    print("Device:", device)

    seed_everything(args.seed)

    checkpoint_path, adapter_path, merged_state_dict_path = build_checkpoint_paths(
        args,
        base_dir,
    )

    print(f"Output checkpoint: {checkpoint_path}")
    print(f"Output adapter: {adapter_path}")

    if args.save_merged_state_dict:
        print(f"Output merged state_dict: {merged_state_dict_path}")

    train_loader, val_loader = setup_data(args, base_dir)

    model, _ = setup_model(
        args=args,
        base_dir=base_dir,
        device=device,
    )

    criterion, optimizer, scheduler, scaler = setup_optimization(
        args=args,
        model=model,
        device=device,
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            epoch=epoch,
            args=args,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        val_loss = validate_one_epoch(
            epoch=epoch,
            args=args,
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"lr: {current_lr:.8f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                adapter_path=adapter_path,
                merged_state_dict_path=merged_state_dict_path,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                args=args,
            )

            print(f"New best LoRA model saved: {checkpoint_path}")


if __name__ == "__main__":
    main()