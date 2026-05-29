from collections.abc import Sequence
from typing import List

import torch
import torch.nn.functional as F


def _compute_pca(
    sd_src_featmap,
    sd_trg_featmap,
    featmap_size: tuple[int, int],
    pca_dims: Sequence[int],
    weights: Sequence[float],
):
    """
    co-PCA su 3 scale DIFT (s5,s4,s3) e costruzione descriptor:
      - input layer: sd_*_featmap[0/1/2] = (s5/s4/s3)
      - output: sd_src_desc, sd_trg_desc: [1,1,P,Dsd]
               dims_used: [d_s5, d_s4, d_s3]
    """
    dims_used: List[int] = []
    src_red_list = []
    trg_red_list = []

    src_layers = [sd_src_featmap[0], sd_src_featmap[1], sd_src_featmap[2]]
    trg_layers = [sd_trg_featmap[0], sd_trg_featmap[1], sd_trg_featmap[2]]

    for i, out_dim in enumerate(pca_dims):
        fs = src_layers[i]
        ft = trg_layers[i]

        if fs.ndim == 3:
            fs = fs.unsqueeze(0)
        if ft.ndim == 3:
            ft = ft.unsqueeze(0)

        if fs.shape[-2:] != featmap_size:
            fs = F.interpolate(fs, size=featmap_size, mode="bilinear", align_corners=False)
        if ft.shape[-2:] != featmap_size:
            ft = F.interpolate(ft, size=featmap_size, mode="bilinear", align_corners=False)

        _, channels, height, width = fs.shape
        num_patches = height * width
        q = min(out_dim, channels)
        dims_used.append(q)

        fs_tok = fs.permute(0, 2, 3, 1).reshape(num_patches, channels)
        ft_tok = ft.permute(0, 2, 3, 1).reshape(num_patches, channels)

        x = torch.cat([fs_tok, ft_tok], dim=0)
        mean = x.mean(dim=0, keepdim=True)
        xc = (x - mean).float()

        _, _, v = torch.pca_lowrank(xc, q=q)
        z = xc @ v[:, :q]

        zs = z[:num_patches, :]
        zt = z[num_patches:, :]

        fs_red = zs.reshape(1, height, width, q).permute(0, 3, 1, 2).contiguous().to(fs.dtype)
        ft_red = zt.reshape(1, height, width, q).permute(0, 3, 1, 2).contiguous().to(ft.dtype)

        src_red_list.append(fs_red)
        trg_red_list.append(ft_red)

    sd_src_proc = torch.cat(src_red_list, dim=1)
    sd_trg_proc = torch.cat(trg_red_list, dim=1)

    height, width = featmap_size
    num_patches = height * width
    sd_src_desc = sd_src_proc.reshape(1, -1, num_patches).permute(0, 2, 1).unsqueeze(1).contiguous()
    sd_trg_desc = sd_trg_proc.reshape(1, -1, num_patches).permute(0, 2, 1).unsqueeze(1).contiguous()

    d0, d1, d2 = dims_used
    sd_src_desc[..., :d0] *= weights[0]
    sd_src_desc[..., d0:d0 + d1] *= weights[1]
    sd_src_desc[..., d0 + d1:d0 + d1 + d2] *= weights[2]

    sd_trg_desc[..., :d0] *= weights[0]
    sd_trg_desc[..., d0:d0 + d1] *= weights[1]
    sd_trg_desc[..., d0 + d1:d0 + d1 + d2] *= weights[2]

    return sd_src_desc, sd_trg_desc, dims_used
