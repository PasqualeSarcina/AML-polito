from collections.abc import Sequence
from typing import List

import torch
import torch.nn.functional as F


def _compute_pca(
    sd_src_featmap,
    sd_trg_featmap,
    featmap_size: tuple[int, int],
    pca_dims: Sequence[int],
):

    dims_used: List[int] = []

    # Lists used to collect the PCA-reduced feature maps for each layer.
    src_red_list = []
    trg_red_list = []

    src_layers = [sd_src_featmap[0], sd_src_featmap[1], sd_src_featmap[2]]
    trg_layers = [sd_trg_featmap[0], sd_trg_featmap[1], sd_trg_featmap[2]]

    for i, out_dim in enumerate(pca_dims):
        fs = src_layers[i]
        ft = trg_layers[i]

        # Ensure that feature maps have a batch dimension. Necessary for interpolate to work
        if fs.ndim == 3:
            fs = fs.unsqueeze(0)
        if ft.ndim == 3:
            ft = ft.unsqueeze(0)

        # Resize source and target feature maps to the same spatial resolution.
        if fs.shape[-2:] != featmap_size:
            fs = F.interpolate(fs, size=featmap_size, mode="bilinear", align_corners=False)
        if ft.shape[-2:] != featmap_size:
            ft = F.interpolate(ft, size=featmap_size, mode="bilinear", align_corners=False)

        _, channels, height, width = fs.shape   #[1, C, H, W]
        num_patches = height * width

        # Use at most as many PCA components as available feature channels.
        q = min(out_dim, channels)
        dims_used.append(q)

        # Convert feature maps from [1, C, H, W] to a patch-wise representation [H*W, C].
        # PCA works with [samples, features], so we treat each spatial location as a sample and the feature channels as features.
        fs_tok = fs.permute(0, 2, 3, 1).reshape(num_patches, channels)
        ft_tok = ft.permute(0, 2, 3, 1).reshape(num_patches, channels)

        # Concatenate source and target descriptors so that PCA is fitted on both images jointly.
        x = torch.cat([fs_tok, ft_tok], dim=0)
        # Center the descriptors
        mean = x.mean(dim=0, keepdim=True)
        xc = (x - mean).float()

        # Compute a low-rank PCA basis and project the centered descriptors onto it.
        # v contains the q principal directions
        _, _, v = torch.pca_lowrank(xc, q=q)
        # Project the centered descriptors onto the first q PCA components,
        # reducing their dimensionality from C channels to q components.
        z = xc @ v[:, :q]

        # Split the projected descriptors back into source and target parts.
        zs = z[:num_patches, :]
        zt = z[num_patches:, :]

        # Reshape PCA-reduced patch descriptors back into feature-map format [1, q, H, W].
        fs_red = zs.reshape(1, height, width, q).permute(0, 3, 1, 2).contiguous().to(fs.dtype)
        ft_red = zt.reshape(1, height, width, q).permute(0, 3, 1, 2).contiguous().to(ft.dtype)

        src_red_list.append(fs_red)
        trg_red_list.append(ft_red)

    # Concatenate PCA-reduced descriptors from all selected layers along the channel dimension.
    sd_src_proc = torch.cat(src_red_list, dim=1)
    sd_trg_proc = torch.cat(trg_red_list, dim=1)

    height, width = featmap_size
    num_patches = height * width

    # Convert feature maps into descriptor tensors [1, 1, num_patches, C_total]
    sd_src_desc = sd_src_proc.reshape(1, -1, num_patches).permute(0, 2, 1).unsqueeze(1).contiguous()
    sd_trg_desc = sd_trg_proc.reshape(1, -1, num_patches).permute(0, 2, 1).unsqueeze(1).contiguous()

    return sd_src_desc, sd_trg_desc, dims_used
