import math
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
import torch.nn.functional as F
from models.dinov3.PreProcess import PreProcess
from utils.soft_argmax_window import soft_argmax_window
from utils.utils_convert import pixel_to_patch_idx, patch_idx_to_pixel
from utils.utils_featuremaps import load_featuremap, save_featuremap
from utils.utils_init_dataloader import init_dataloader
from utils.utils_results import CorrespondenceResult


class Dinov3Eval:
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.custom_weights = args.custom_weights
        self.win_soft_argmax = args.win_soft_argmax
        self.wsam_win_size = args.wsam_win_size
        self.wsam_beta = args.wsam_beta
        self.device = args.device
        self.base_dir = args.base_dir

        self._init_model()

        transform = PreProcess()
        self.dataset, self.dataloader = init_dataloader(self.dataset_name, 'test', transform=transform)

        self.feat_dir = Path(self.base_dir) / "data" / "features" / "dinov3"
        self.processed_img = defaultdict(set)

    def _init_model(self):
        if self.custom_weights is not None:
            checkpoint = self.custom_weights
        else:
            checkpoint = Path(self.base_dir) / "checkpoints" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

        model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16', weights=str(checkpoint))

        self.model = model.to(self.device).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

    def compute_features(self, img_tensor: torch.Tensor, img_name: str, category: str) -> torch.Tensor:
        PATCH = 16
        if self.dataset_name == "ap-10k":
            category = "all"

        # ----------------------------
        # helper: safe_tokens + tokens_to_featuremap
        # ----------------------------
        def safe_tokens(out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            if (not torch.is_tensor(out)) or out.ndim != 3:
                raise RuntimeError(f"Unexpected tokens output: {type(out)} shape={getattr(out, 'shape', None)}")
            return out  # (B, N, C)

        def tokens_to_featuremap(tokens_bnc: torch.Tensor, h_grid: int, w_grid: int) -> torch.Tensor:
            tok = tokens_bnc.squeeze(0)  # (Ntok, C)  [assume B=1]
            Npatch = h_grid * w_grid
            if tok.shape[0] < Npatch:
                raise RuntimeError(f"Ntok={tok.shape[0]} < Npatch={Npatch} (h_grid={h_grid}, w_grid={w_grid})")
            patch_tok = tok[-Npatch:]  # drop CLS/register if present
            Ft = patch_tok.view(h_grid, w_grid, -1)  # (h_grid, w_grid, C)
            return F.normalize(Ft, dim=-1)

        # ----------------------------
        # load cache
        # ----------------------------
        if img_name in self.processed_img[category]:
            return load_featuremap(img_name, self.feat_dir, device=self.device)

        # ----------------------------
        # forward
        # ----------------------------
        # img_tensor is expected to be CHW (not batched). If your PreProcess already returns BCHW,
        # remove the unsqueeze(0) below.
        x = img_tensor.to(self.device)
        if x.ndim == 3:  # CHW
            x = x.unsqueeze(0)  # -> BCHW
        elif x.ndim != 4:
            raise RuntimeError(f"Unexpected img_tensor shape {x.shape}, expected CHW or BCHW")

        dict_out = self.model.forward_features(x)

        # take patch tokens and make them safe
        tokens = safe_tokens(dict_out["x_norm_patchtokens"])  # (B, N, C)

        # ----------------------------
        # grid size from model input resolution
        # (must match how the model sees the image!)
        # ----------------------------
        H, W = x.shape[-2], x.shape[-1]
        h_grid = (H + PATCH - 1) // PATCH
        w_grid = (W + PATCH - 1) // PATCH

        featmap = tokens_to_featuremap(tokens, h_grid, w_grid)  # (h_grid, w_grid, C)

        # ----------------------------
        # save cache
        # ----------------------------
        self.processed_img[category].add(img_name)
        save_featuremap(featmap, img_name, self.feat_dir)

        return featmap

    def evaluate(self) -> list[CorrespondenceResult]:
        results = []

        PATCH = 16
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in tqdm(self.dataloader, total=len(self.dataloader),
                              desc=f"Computing correspondences with DINOv3 on {self.dataset_name}",
                              smoothing=0.1, mininterval=0.7, maxinterval=2.0):
                category = batch["category"]

                feats_src = self.compute_features(batch["src_img"], batch["src_imname"], category)  # Resized image
                feats_trg = self.compute_features(batch["trg_img"], batch["trg_imname"], category)  # Resized image

                src_kps = batch["src_kps"].to(self.device)  # (N,2)
                trg_kps = batch["trg_kps"].to(self.device)  # (N,2)

                _, Hs_pad, Ws_pad = batch["src_imsize"]
                _, Ht_pad, Wt_pad = batch["trg_imsize"]

                Hs, Ws = batch["src_orig_size"]
                Ht, Wt = batch["trg_orig_size"]

                hv_s, wv_s = int(Hs_pad) // PATCH, int(Ws_pad) // PATCH
                hv_t, wv_t = int(Ht_pad) // PATCH, int(Wt_pad) // PATCH

                distances_this_image: list[float] = []

                for i in range(src_kps.shape[0]):
                    src_kp = src_kps[i]  # (2,)
                    trg_kp = trg_kps[i]  # (2,)

                    # indice patch sorgente usando la tua funzione
                    x_idx, y_idx = pixel_to_patch_idx(
                        xy=src_kp,
                        stride=PATCH,
                        grid_hw=(hv_s, wv_s),
                        img_hw=(Hs, Ws),  # ORIGINAL SIZE (non paddata)
                    )
                    idx = y_idx * wv_s + x_idx

                    src_feat = feats_src[y_idx, x_idx]  # (C,)
                    sim_2d = (feats_trg * src_feat).sum(dim=-1)

                    if self.win_soft_argmax:
                        y_pred_patch, x_pred_patch = soft_argmax_window(sim_2d, window_radius=self.wsam_win_size,
                                                                        temperature=self.wsam_beta)
                    else:
                        y_pred_patch, x_pred_patch = soft_argmax_window(sim_2d, window_radius=1)

                    x_pred, y_pred = patch_idx_to_pixel((x_pred_patch, y_pred_patch), stride=PATCH)

                    x_pred = max(0.0, min(x_pred, float(Wt - 1)))
                    y_pred = max(0.0, min(y_pred, float(Ht - 1)))

                    # distanza in pixel originali
                    dx = x_pred - float(trg_kp[0])
                    dy = y_pred - float(trg_kp[1])
                    dist = math.sqrt(dx * dx + dy * dy)
                    distances_this_image.append(dist)

                    # salva risultato
                results.append(
                    CorrespondenceResult(
                        category=category,
                        distances=distances_this_image,
                        pck_threshold_0_05=batch["pck_threshold_0_05"],
                        pck_threshold_0_1=batch["pck_threshold_0_1"],
                        pck_threshold_0_2=batch["pck_threshold_0_2"]
                    )
                )

        return results
