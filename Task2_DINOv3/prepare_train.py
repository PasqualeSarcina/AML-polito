import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.extraction_DINOv3 import extract_dense_features
from .loss_DINOv3 import compute_gaussian_ce_loss_from_feats
from .eval_val_DINOv3 import evaluate_model_batched

PATH_CHECKPOINTS = ".checkpoints/dinov3/"


def prepare_model_for_fine_tuning(model, num_layers_to_unfreeze, *, unfreeze_final_norm=True):
    # 1) Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # 2) Safety: how many blocks?
    if not hasattr(model, "blocks"):
        raise AttributeError("Il modello non ha l'attributo `blocks`. Controlla l'architettura DINOv3 caricata.")

    total_blocks = len(model.blocks)
    num_layers_to_unfreeze = int(min(max(num_layers_to_unfreeze, 0), total_blocks))

    # 3) Unfreeze last N blocks (tutti i parametri, incluse le Norm)
    if num_layers_to_unfreeze > 0:
        for block in model.blocks[-num_layers_to_unfreeze:]:
            for p in block.parameters():
                p.requires_grad_(True)

            # In più, per essere espliciti: assicurati che ogni norm nel blocco sia trainable
            for m in block.modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                  nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                    for p in m.parameters(recurse=False):
                        p.requires_grad_(True)

    # 4) Unfreeze final norm (molto utile in ViT fine-tuning)
    if unfreeze_final_norm and hasattr(model, "norm") and model.norm is not None:
        for p in model.norm.parameters():
            p.requires_grad_(True)

    # 5) Report
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modello pronto: sbloccati ultimi {num_layers_to_unfreeze} blocchi (Norm incluse)"
          f"{' + norm finale' if unfreeze_final_norm and hasattr(model,'norm') and model.norm is not None else ''}"
          f" | trainable={trainable_params:,}")

    return model


def save_checkpoint(
    model,
    optimizer,
    epoch,
    pck,
    layer_name,
    *,
    is_best=False,
    scaler=None,
    hparams: dict | None = None
):
    os.makedirs(PATH_CHECKPOINTS, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "pck_10": pck,
        "layer": layer_name,
    }

    if hparams is not None:
        checkpoint["hparams"] = hparams

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if is_best:
        path = os.path.join(PATH_CHECKPOINTS, f"best_model_{layer_name}.pth")
        torch.save(checkpoint, path)

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    *,
    out_size: int,
    patch_size: int,
    max_train_batches=None,
    n_layers_feats: int = 1,
    grad_clip: float = 1.0,
):
    model.train()
    total_loss = 0.0
    steps = 0
    use_amp = (device.type == "cuda")

    pbar = tqdm(loader, desc="  Training", leave=False)
    for b_idx, batch in enumerate(pbar):
        if (max_train_batches is not None) and (b_idx >= max_train_batches):
            break

        src_img = batch["src_img"].to(device, non_blocking=True)  # [B,3,H,W]
        trg_img = batch["trg_img"].to(device, non_blocking=True)

        src_kps_list = batch["src_kps"]      # list of [Ki,2]
        trg_kps_list = batch["trg_kps"]
        src_meta_list = batch["src_meta"]    # list of dict
        trg_meta_list = batch["trg_meta"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            src_feats = extract_dense_features(
                model, src_img, n_layers=n_layers_feats, return_grid=False
            )  # [B,C,Hf,Wf]
            trg_feats = extract_dense_features(
                model, trg_img, n_layers=n_layers_feats, return_grid=False
            )

            loss = compute_gaussian_ce_loss_from_feats(
                src_feats,
                trg_feats,
                src_kps_list,
                trg_kps_list,
                src_meta_list,
                trg_meta_list,
                out_size=out_size,
                patch_size=patch_size,
            )

        # backward
        scaler.scale(loss).backward()

        # clip grad on trainable params
        scaler.unscale_(optimizer)
        trainable_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if trainable_params and (grad_clip is not None) and (grad_clip > 0):
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(grad_clip))

        scaler.step(optimizer)
        scaler.update()

        cur = float(loss.item())
        total_loss += cur
        steps += 1
        pbar.set_postfix(loss=f"{cur:.4f}")

    return total_loss / max(steps, 1)

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    *,
    out_size: int,
    patch_size: int,
    max_val_batches=None,
    max_pck_pairs_val=None,
    n_layers_feats: int = 1,
):
    """
    Coerente con Task2:
      - val loss = Gaussian CE padding-aware
      - val PCK = evaluate_model_batched (padding-aware matcher)
    """
    model.eval()
    total_loss = 0.0
    steps = 0
    use_amp = (device.type == "cuda")

    for b_idx, batch in enumerate(loader):
        if (max_val_batches is not None) and (b_idx >= max_val_batches):
            break

        src_img = batch["src_img"].to(device, non_blocking=True)
        trg_img = batch["trg_img"].to(device, non_blocking=True)

        src_kps_list = batch["src_kps"]
        trg_kps_list = batch["trg_kps"]
        src_meta_list = batch["src_meta"]
        trg_meta_list = batch["trg_meta"]

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            src_feats = extract_dense_features(
                model, src_img, n_layers=n_layers_feats, return_grid=False
            )
            trg_feats = extract_dense_features(
                model, trg_img, n_layers=n_layers_feats, return_grid=False
            )

            loss = compute_gaussian_ce_loss_from_feats(
                src_feats,
                trg_feats,
                src_kps_list,
                trg_kps_list,
                src_meta_list,
                trg_meta_list,
                out_size=out_size,
                patch_size=patch_size,
            )

        total_loss += float(loss.item())
        steps += 1

    val_loss = total_loss / max(steps, 1)

    # PCK on VAL
    pck_res = evaluate_model_batched(
        name="VAL",
        model=model,
        loader=loader,
        device=device,
        max_pairs=max_pck_pairs_val,
        n_layers=n_layers_feats,
        patch_size=patch_size,
        thresholds=(0.05, 0.10, 0.20),
    )

    return {
        "val_loss": val_loss,
        "pck_05": pck_res["pck_per_keypoint"][0.05] * 100.0,
        "pck_10": pck_res["pck_per_keypoint"][0.10] * 100.0,
        "pck_20": pck_res["pck_per_keypoint"][0.20] * 100.0,
        "per_category": pck_res.get("per_category", None),
    }