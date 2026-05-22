from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRASetupSummary:
    target_modules: tuple[str, ...]
    replaced_modules: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear.

    The pretrained linear layer is kept frozen and the trainable update is:
        W_eff = W_0 + (alpha / r) * B @ A

    During the forward pass:
        y = linear(x, W_0, b) + scaling * linear(linear(dropout(x), A), B)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can wrap only torch.nn.Linear modules.")
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}.")

        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.in_features = in_features
        self.out_features = out_features

        self.lora_A = nn.Parameter(torch.empty(self.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))

        self.reset_parameters()

        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_output = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_output + self.scaling * lora_output

    def merged_linear(self) -> nn.Linear:
        """Return a plain nn.Linear equivalent to base + LoRA update."""
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )

        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data.copy_(self.base_layer.weight.data + delta_w.to(self.base_layer.weight.dtype))

        if self.base_layer.bias is not None:
            merged.bias.data.copy_(self.base_layer.bias.data)

        return merged


def _module_matches(name: str, target_modules: Iterable[str]) -> bool:
    leaf_name = name.split(".")[-1]
    return any(leaf_name == target or name.endswith(target) for target in target_modules)


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def apply_lora_to_dinov3(
    model: nn.Module,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Iterable[str] = ("qkv",),
    train_norm: bool = False,
) -> LoRASetupSummary:
    """
    Inject LoRA modules into DINOv3 attention projections.

    Default target_modules=("qkv",) follows the DINOv3 ViT implementation where each
    attention block usually has a fused query/key/value projection named attn.qkv.
    The pretrained backbone is frozen; only LoRA matrices are trainable by default.
    """

    target_modules = tuple(target_modules)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    replacements: list[str] = []

    # list(...) is important because we mutate the module tree while iterating.
    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not _module_matches(module_name, target_modules):
            continue

        parent, child_name = _get_parent_module(model, module_name)
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replacements.append(module_name)

    if not replacements:
        available_linear_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        raise RuntimeError(
            "No nn.Linear modules matched target_modules="
            f"{target_modules}. Available linear modules include: {available_linear_names[:30]}"
        )

    if train_norm:
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    trainable, total = count_parameters(model)

    return LoRASetupSummary(
        target_modules=target_modules,
        replaced_modules=tuple(replacements),
        trainable_parameters=trainable,
        total_parameters=total,
    )


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA trainable weights, useful for a compact adapter checkpoint."""
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Replace every LoRALinear module with a plain nn.Linear containing merged weights.

    This is useful to save a standard DINOv3 state_dict that can be loaded by the
    existing evaluation code as custom_weights, without rebuilding LoRA modules.
    """

    for module_name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent, child_name = _get_parent_module(model, module_name)
            setattr(parent, child_name, module.merged_linear())

    return model
