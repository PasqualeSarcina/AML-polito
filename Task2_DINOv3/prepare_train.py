import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.extraction_DINOv3 import extract_dense_features
from .loss_DINOv3 import GaussianCrossEntropyLoss
from .eval_val_DINOv3 import evaluate_model_batched

PATH_CHECKPOINTS = ".checkpoints/dinov3/"


def prepare_model_for_fine_tuning(model, num_layers_to_unfreeze, *, unfreeze_final_norm=True):
    for p in model.parameters():
        p.requires_grad_(False)

    if not hasattr(model, "blocks"):
        raise AttributeError("Il modello non ha l'attributo `blocks`. Controlla l'architettura DINOv3 caricata.")

    total_blocks = len(model.blocks)
    num_layers_to_unfreeze = int(min(max(num_layers_to_unfreeze, 0), total_blocks))

    if num_layers_to_unfreeze > 0:
        for block in model.blocks[-num_layers_to_unfreeze:]:
            for p in block.parameters():
                p.requires_grad_(True)

            for m in block.modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                  nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                    for p in m.parameters(recurse=False):
                        p.requires_grad_(True)

    if unfreeze_final_norm and hasattr(model, "norm") and model.norm is not None:
        for p in model.norm.parameters():
            p.requires_grad_(True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modello pronto: sbloccati ultimi {num_layers_to_unfreeze} blocchi (Norm incluse)"
          f"{' + norm finale' if unfreeze_final_norm and hasattr(model,'norm') and model.norm is not None else ''}"
          f" | trainable={trainable_params:,}")

    return model

def save_checkpoint(
    model,
    optimizer,
    epoch,
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
    loss_fn,
    *,
    max_train_batches=None,
    n_layers_feats: int = 1,
    grad_clip: float = 1.0,
):
    model.train()
    total_loss = 0.0        
    total_kps  = 0          
    zero_batches = 0        
    steps = 0              
    use_amp = (device.type == "cuda")

    pbar = tqdm(loader, desc="  Training", leave=False)
    for b_idx, batch in enumerate(pbar):
        if (max_train_batches is not None) and (b_idx >= max_train_batches):
            break
        
        steps += 1

        src_img = batch["src_img"].to(device, non_blocking=True)
        trg_img = batch["trg_img"].to(device, non_blocking=True)

        src_kps_list  = batch["src_kps"]
        trg_kps_list  = batch["trg_kps"]
        src_meta_list = batch["src_meta"]
        trg_meta_list = batch["trg_meta"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            src_feats = extract_dense_features(
                model, src_img, n_layers=n_layers_feats, return_grid=False
            )
            trg_feats = extract_dense_features(
                model, trg_img, n_layers=n_layers_feats, return_grid=False
            )

            loss, nkps = loss_fn(
                src_feats,
                trg_feats,
                src_kps_list,
                trg_kps_list,
                src_meta_list,
                trg_meta_list,
                return_kps=True,
            )
        nkps = int(nkps)
        if nkps == 0:
            zero_batches += 1
            avg = total_loss / max(total_kps, 1)
            pbar.set_postfix(loss_batch="0.0000", loss_avg=f"{avg:.4f}", nkps=0)
            continue
        
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        trainable_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if trainable_params and (grad_clip is not None) and (grad_clip > 0):
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(grad_clip))

        scaler.step(optimizer)
        scaler.update()

        cur = float(loss.item())  # loss media per-kp (del batch)
        total_loss += cur * int(nkps)
        total_kps  += int(nkps)

        avg = total_loss / max(total_kps, 1)
        pbar.set_postfix(loss_batch=f"{cur:.4f}", loss_avg=f"{avg:.4f}", nkps=int(nkps))

    return total_loss / max(total_kps, 1)


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    loss_fn,
    *,
    max_val_batches=None,
    n_layers_feats: int = 1,
):
    model.eval()
    total_loss = 0.0
    total_kps  = 0
    zero_batches = 0
    steps = 0
    use_amp = (device.type == "cuda")

    pbar = tqdm(loader, desc="  Val", leave=False)
    for b_idx, batch in enumerate(pbar):
        if (max_val_batches is not None) and (b_idx >= max_val_batches):
            break

        src_img = batch["src_img"].to(device, non_blocking=True)
        trg_img = batch["trg_img"].to(device, non_blocking=True)

        src_kps_list  = batch["src_kps"]
        trg_kps_list  = batch["trg_kps"]
        src_meta_list = batch["src_meta"]
        trg_meta_list = batch["trg_meta"]

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            src_feats = extract_dense_features(
                model, src_img, n_layers=n_layers_feats, return_grid=False
            )
            trg_feats = extract_dense_features(
                model, trg_img, n_layers=n_layers_feats, return_grid=False
            )

            loss, nkps = loss_fn(
                src_feats,
                trg_feats,
                src_kps_list,
                trg_kps_list,
                src_meta_list,
                trg_meta_list,
                return_kps=True,
            )

        steps += 1
        zero_batches += int(nkps == 0)

        cur = float(loss.item())
        total_loss += cur * int(nkps)
        total_kps  += int(nkps)

        avg = total_loss / max(total_kps, 1)
        pbar.set_postfix(val_loss_batch=f"{cur:.4f}", val_loss_avg=f"{avg:.4f}", nkps=int(nkps))

    val_loss = total_loss / max(total_kps, 1)

    return {"val_loss": float(val_loss)}
