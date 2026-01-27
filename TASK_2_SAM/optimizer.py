import torch
import torch.nn as nn
from torch.optim import AdamW

# 1 layer
def setup_optimizer(model):
    base_lr = 8e-4
    wd = 0.1
    ld = 0.6 

    layers = model.image_encoder.blocks
    num_layers = len(layers)

    params = []
    for i, layer in enumerate(layers):
        lr_i = base_lr * (ld ** (num_layers - i))
        params.append({"params": layer.parameters(), "lr": lr_i, "weight_decay": wd})

    params.append({"params": model.image_encoder.neck.parameters(), "lr": base_lr, "weight_decay": wd})

    optimizer = AdamW(params, betas=(0.9, 0.999))
    return optimizer

# 2 layers
def setup_optimizer_v2(model):
    base_lr = 2e-4
    wd = 0.2
    ld = 0.6

    layers = model.image_encoder.blocks
    num_layers = len(layers)

    params = []
    for i, layer in enumerate(layers):
        trainable_layer_params = [p for p in layer.parameters() if p.requires_grad]

        if len(trainable_layer_params) > 0:
            lr_i = base_lr * (ld ** (num_layers - i))
            params.append({
                "params": trainable_layer_params,
                "lr": lr_i,
                "weight_decay": wd
            })

    neck_params = [p for p in model.image_encoder.neck.parameters() if p.requires_grad]
    if neck_params:
        params.append({
            "params": neck_params,
            "lr": base_lr,
            "weight_decay": wd
        })
        
    optimizer = AdamW(params, betas=(0.9, 0.999))
    return optimizer