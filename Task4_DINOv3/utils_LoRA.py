import torch
import torch.nn.functional as F
from copy import deepcopy
from peft import get_peft_model, LoraConfig
import os
from Task2_DINOv3.prepare_train import PATH_CHECKPOINTS


def peft_target_modules_last_blocks(model, last_n_blocks: int, suffixes):
    n_blocks = len(model.blocks)
    start = max(0, n_blocks - last_n_blocks)
    return [f"blocks.{i}.{s}" for i in range(start, n_blocks) for s in suffixes]

def make_peft_lora_model(base_model, *, last_n_blocks, r, alpha, dropout, suffixes, verbose=True):

    model = deepcopy(base_model)

    for p in model.parameters():
        p.requires_grad = False

    targets = peft_target_modules_last_blocks(model, last_n_blocks, suffixes)

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
    )

    lora_model = get_peft_model(model, cfg)
    lora_model.train()

    trainable = [n for n,p in lora_model.named_parameters() if p.requires_grad]
    assert len(trainable) > 0, "PEFT non ha reso nulla trainabile: target_modules non matchano."
    bad = [n for n in trainable if "lora" not in n.lower()]
    assert len(bad) == 0, f"Parametri trainabili non-LoRA trovati: {bad[:10]}"

    if verbose:
        ntr = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        ntot = sum(p.numel() for p in lora_model.parameters())
        print(f"PEFT-LoRA ready | trainable={ntr:,} / total={ntot:,} | targets={len(targets)}")

    return lora_model

def summarize_trainables(model):
    trainable = [(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trn   = sum(n for _,n in trainable)
    print(f"Trainable params: {trn:,} / {total:,} ({100*trn/total:.4f}%)")
    # stampa i primi 30 nomi
    for n,_ in trainable[:30]:
        print("  ", n)
    # sanity: nessun parametro non-LoRA deve essere trainabile
    bad = [n for n,_ in trainable if "lora" not in n.lower()]
    print("Non-LoRA trainables:", bad[:10])
    return trainable

def save_checkpoint(
    model,
    optimizer,
    epoch,
    tag,
    *,
    is_best: bool = False,
    val_loss: float | None = None,
    scaler=None,
    hparams: dict | None = None,
):
    os.makedirs(PATH_CHECKPOINTS, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tag": tag,
    }

    if val_loss is not None:
        checkpoint["val_loss"] = float(val_loss)

    if hparams is not None:
        checkpoint["hparams"] = hparams

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if is_best:
        path = os.path.join(PATH_CHECKPOINTS, f"best_model_LoRA_{tag}.pth")
        torch.save(checkpoint, path)
        return path

    return None

