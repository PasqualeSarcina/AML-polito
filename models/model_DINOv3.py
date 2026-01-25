from __future__ import annotations

from pathlib import Path
import torch
import copy


DINOV3_REPO_URL = "https://github.com/facebookresearch/dinov3.git"
DEFAULT_MODEL_NAME = "dinov3_vitb16"


def load_dinov3_backbone(
    *,
    dinov3_dir: str | Path,
    weights_path: str | Path,
    device: torch.device | str = "cpu",
    sanity_input_size: int = 512,
    verbose: bool = True,
) -> torch.nn.Module:
    """
    Load DINOv3 ViT-B/16 backbone for Task 1 (training-free).

    Requirements:
      - Official DINOv3 repository cloned locally
      - Pretrained checkpoint downloaded separately (licensed)
    """

    dinov3_dir = Path(dinov3_dir).expanduser().resolve()
    weights_path = Path(weights_path).expanduser().resolve()
    device = torch.device(device)

    # ------------------------------------------------------------------
    # 1) Sanity checks: repo + checkpoint
    # ------------------------------------------------------------------
    if not dinov3_dir.exists():
        raise FileNotFoundError(
            f"[DINOv3] Repository not found at: {dinov3_dir}\n"
            f"Clone it first with:\n  git clone {DINOV3_REPO_URL} {dinov3_dir}"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"[DINOv3] Checkpoint not found: {weights_path}\n"
            "Download it after requesting access and place it in the checkpoints directory."
        )

    # ------------------------------------------------------------------
    # 2) Load model from local repo via torch.hub (same behavior as notebook)
    # ------------------------------------------------------------------
    model = torch.hub.load(
        str(dinov3_dir),
        DEFAULT_MODEL_NAME,
        source="local",
        weights=str(weights_path),
    )

    model.to(device).eval()

    # ------------------------------------------------------------------
    # 3) Freeze backbone (Task 1 compliant)
    # ------------------------------------------------------------------
    for p in model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 4) Sanity checks: architecture + patch size
    # ------------------------------------------------------------------
    if not hasattr(model, "blocks"):
        raise RuntimeError("[DINOv3] Loaded model has no attribute 'blocks' (API mismatch?)")

    if not hasattr(model, "patch_embed") or not hasattr(model.patch_embed, "patch_size"):
        raise RuntimeError("[DINOv3] patch_embed.patch_size not found (API mismatch?)")

    n_blocks = len(model.blocks)
    patch = model.patch_embed.patch_size
    patch_int = patch[0] if isinstance(patch, (tuple, list)) else int(patch)

    if patch_int != 16:
        raise RuntimeError(f"[DINOv3] Expected patch size 16 for ViT-B/16, got {patch}")

    # ------------------------------------------------------------------
    # 5) Token-grid sanity check (important for correspondence)
    # ------------------------------------------------------------------
    if sanity_input_size is not None:
        x = torch.randn(1, 3, sanity_input_size, sanity_input_size, device=device)
        with torch.no_grad():
            feats = model.get_intermediate_layers(x, n=1)[0]  # [B, N, C]

        expected_n = (sanity_input_size // patch_int) ** 2
        if feats.shape[1] != expected_n:
            raise RuntimeError(
                f"[DINOv3] Unexpected token count: {feats.shape[1]} vs expected {expected_n}. "
                "Check input size / patch size / token handling."
            )

    if verbose:
        print(
            f"[DINOv3] loaded ViT-B/16 | blocks={n_blocks} | patch={patch_int} | "
            f"checkpoint={weights_path.name}"
        )

    return model



