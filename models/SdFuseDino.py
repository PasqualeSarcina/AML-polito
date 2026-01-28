import math
from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ap10k import AP10KDataset
from data.dataset import get_pckthres
from data.pfpascal import PFPascalDataset
from data.pfwillow import PFWillowDataset
from data.spair import SPairDataset

from models.dift import DiftEval
from models.dift.PreProcess import PreProcess as DiftPreProcess
from models.dinov2 import Dinov2Eval
from models.dinov2.PreProcess import PreProcess as Dinov2PreProcess

from utils.soft_argmax_window import soft_argmax_window
from utils.utils_convert import pixel_to_patch_idx, patch_idx_to_pixel
from utils.utils_init_dataloader import init_dataloader
from utils.utils_results import CorrespondenceResult


class SdFuseDino:
    # "Repo-like"
    PCA_DIMS = [256, 256, 256]          # s5,s4,s3
    WEIGHT = [1, 1, 1, 1, 1]            # [w_s5,w_s4,w_s3,w_sd,w_dino]

    def __init__(self, args):
        self.dataset_name = args.dataset
        self.win_soft_argmax = args.win_soft_argmax
        self.wsam_win_size = args.wsam_win_size
        self.wsam_beta = args.wsam_beta
        self.device = args.device
        self.base_dir = args.base_dir

        self.sd = DiftEval(args)
        self.dino = Dinov2Eval(args)

        self.featmap_size: tuple[int, int] = (48, 48)  # H,W
        self.H, self.W = self.featmap_size
        self.P = self.H * self.W

        self.dino_stride = 14
        self.sd_stride = 16

        self.sd_preproc = DiftPreProcess(out_dim=(self.H * self.sd_stride, self.W * self.sd_stride))
        self.dino_preproc = Dinov2PreProcess(out_dim=(self.H * self.dino_stride, self.W * self.dino_stride))

        self.dataset, self.dataloader = init_dataloader(self.dataset_name, 'test', transform=None)

    def _compute_features(self, batch: dict):
        # ----------------------------
        # SD
        # ----------------------------
        sd_batch = self.sd_preproc(deepcopy(batch))
        sd_src_featmap = self.sd.compute_features(
            sd_batch["src_img"], sd_batch["src_imname"], sd_batch["category"], up_ft_index=[0, 1, 2], t=100
        )
        sd_trg_featmap = self.sd.compute_features(
            sd_batch["trg_img"], sd_batch["trg_imname"], sd_batch["category"], up_ft_index=[0, 1, 2], t=100
        )

        # ----------------------------
        # DINO
        # ----------------------------
        dino_batch = self.dino_preproc(deepcopy(batch))
        dino_src_tokens = self.dino.compute_features(
            dino_batch["src_img"], dino_batch["src_imname"], dino_batch["category"]
        )
        dino_trg_tokens = self.dino.compute_features(
            dino_batch["trg_img"], dino_batch["trg_imname"], dino_batch["category"]
        )

        # rimuovi CLS se presente: [1, 1+P, C] -> [1, P, C]
        if dino_src_tokens.ndim == 3 and dino_src_tokens.shape[1] == 1 + self.P:
            dino_src_tokens = dino_src_tokens[:, 1:, :]
        if dino_trg_tokens.ndim == 3 and dino_trg_tokens.shape[1] == 1 + self.P:
            dino_trg_tokens = dino_trg_tokens[:, 1:, :]

        # [1,P,C] -> [1,C,H,W]
        dino_src_featmap = dino_src_tokens.permute(0, 2, 1).reshape(1, dino_src_tokens.shape[2], self.H, self.W)
        dino_trg_featmap = dino_trg_tokens.permute(0, 2, 1).reshape(1, dino_trg_tokens.shape[2], self.H, self.W)

        return sd_src_featmap, sd_trg_featmap, dino_src_featmap, dino_trg_featmap

    def _compute_pca(self, sd_src_featmap, sd_trg_featmap):
        """
        co-PCA su 3 scale (s5,s4,s3) e costruzione descriptor SD:
          - input layer: sd_*_featmap[0/1/2] = (s5/s4/s3)
          - output: sd_src_desc, sd_trg_desc: [1,1,P, Dsd]
                   dims_used: [d_s5, d_s4, d_s3] (per slicing/pesi)
        """
        dims_used: List[int] = []
        src_red_list = []
        trg_red_list = []

        src_layers = [sd_src_featmap[0], sd_src_featmap[1], sd_src_featmap[2]]
        trg_layers = [sd_trg_featmap[0], sd_trg_featmap[1], sd_trg_featmap[2]]

        for i, out_dim in enumerate(self.PCA_DIMS):
            fs = src_layers[i]
            ft = trg_layers[i]

            # -> [1,C,H,W]
            if fs.ndim == 3:
                fs = fs.unsqueeze(0)
            if ft.ndim == 3:
                ft = ft.unsqueeze(0)

            # rescale a griglia comune (H,W)
            if fs.shape[-2:] != self.featmap_size:
                fs = F.interpolate(fs, size=self.featmap_size, mode="bilinear", align_corners=False)
            if ft.shape[-2:] != self.featmap_size:
                ft = F.interpolate(ft, size=self.featmap_size, mode="bilinear", align_corners=False)

            _, C, H, W = fs.shape
            P = H * W
            q = min(out_dim, C)  # robust
            dims_used.append(q)

            # [1,C,H,W] -> [P,C]
            fs_tok = fs.permute(0, 2, 3, 1).reshape(P, C)
            ft_tok = ft.permute(0, 2, 3, 1).reshape(P, C)

            # co-PCA su [2P,C]
            X = torch.cat([fs_tok, ft_tok], dim=0)  # [2P,C]
            mean = X.mean(dim=0, keepdim=True)
            Xc = (X - mean).float()

            _, _, V = torch.pca_lowrank(Xc, q=q)    # V: [C,q]
            Z = Xc @ V[:, :q]                       # [2P,q]

            Zs = Z[:P, :]
            Zt = Z[P:, :]

            # -> [1,q,H,W]
            fs_red = Zs.reshape(1, H, W, q).permute(0, 3, 1, 2).contiguous().to(fs.dtype)
            ft_red = Zt.reshape(1, H, W, q).permute(0, 3, 1, 2).contiguous().to(ft.dtype)

            src_red_list.append(fs_red)
            trg_red_list.append(ft_red)

        # concat canale: [1, Dsd, H, W]
        sd_src_proc = torch.cat(src_red_list, dim=1)
        sd_trg_proc = torch.cat(trg_red_list, dim=1)

        # -> [1,1,P,Dsd]
        sd_src_desc = sd_src_proc.reshape(1, -1, self.P).permute(0, 2, 1).unsqueeze(1).contiguous()
        sd_trg_desc = sd_trg_proc.reshape(1, -1, self.P).permute(0, 2, 1).unsqueeze(1).contiguous()

        # pesi intra-SD (su dims reali)
        d0, d1, d2 = dims_used
        sd_src_desc[..., :d0] *= self.WEIGHT[0]
        sd_src_desc[..., d0:d0 + d1] *= self.WEIGHT[1]
        sd_src_desc[..., d0 + d1:d0 + d1 + d2] *= self.WEIGHT[2]

        sd_trg_desc[..., :d0] *= self.WEIGHT[0]
        sd_trg_desc[..., d0:d0 + d1] *= self.WEIGHT[1]
        sd_trg_desc[..., d0 + d1:d0 + d1 + d2] *= self.WEIGHT[2]

        return sd_src_desc, sd_trg_desc, dims_used

    def evaluate(self) -> List[CorrespondenceResult]:
        results: List[CorrespondenceResult] = []

        with torch.no_grad():
            for batch in tqdm(
                self.dataloader,
                total=len(self.dataloader),
                desc=f"DIFT + DINOv2 Eval on {self.dataset_name}",
            ):
                # ------------------------------------------------------------
                # 1) Features
                # ------------------------------------------------------------
                sd_src_featmap, sd_trg_featmap, dino_src_featmap, dino_trg_featmap = self._compute_features(batch)

                # ------------------------------------------------------------
                # 2) SD: co-PCA -> descriptor [1,1,P,Dsd]
                # ------------------------------------------------------------
                sd_src_desc, sd_trg_desc, dims_used = self._compute_pca(sd_src_featmap, sd_trg_featmap)
                sd_dim = sum(dims_used)

                # ------------------------------------------------------------
                # 3) DINO: featmap -> descriptor [1,1,P,Ddino]
                # ------------------------------------------------------------
                if dino_src_featmap.shape[-2:] != self.featmap_size:
                    dino_src_featmap = F.interpolate(
                        dino_src_featmap, size=self.featmap_size, mode="bilinear", align_corners=False
                    )
                    dino_trg_featmap = F.interpolate(
                        dino_trg_featmap, size=self.featmap_size, mode="bilinear", align_corners=False
                    )

                dino_src_desc = dino_src_featmap.permute(0, 2, 3, 1).reshape(1, 1, self.P, -1).contiguous()
                dino_trg_desc = dino_trg_featmap.permute(0, 2, 3, 1).reshape(1, 1, self.P, -1).contiguous()

                # ------------------------------------------------------------
                # 4) SOLO L2:
                #    - L2 normalize SEMPRE (repo-like quando usa l2)
                #    - similarity = -||x - y||^2
                # ------------------------------------------------------------
                sd_src_desc = F.normalize(sd_src_desc, p=2, dim=-1, eps=1e-6)
                sd_trg_desc = F.normalize(sd_trg_desc, p=2, dim=-1, eps=1e-6)
                dino_src_desc = F.normalize(dino_src_desc, p=2, dim=-1, eps=1e-6)
                dino_trg_desc = F.normalize(dino_trg_desc, p=2, dim=-1, eps=1e-6)

                fuse_src_desc = torch.cat((sd_src_desc, dino_src_desc), dim=-1)
                fuse_trg_desc = torch.cat((sd_trg_desc, dino_trg_desc), dim=-1)

                fuse_src_desc[..., :sd_dim] *= self.WEIGHT[3]
                fuse_src_desc[..., sd_dim:] *= self.WEIGHT[4]
                fuse_trg_desc[..., :sd_dim] *= self.WEIGHT[3]
                fuse_trg_desc[..., sd_dim:] *= self.WEIGHT[4]

                # ------------------------------------------------------------
                # 5) Keypoints + similarity map (48x48) -> pred
                # ------------------------------------------------------------
                sd_batch = self.sd_preproc(deepcopy(batch))
                src_kps = sd_batch["src_kps"].to(self.device)  # (N,2)
                trg_kps = sd_batch["trg_kps"].to(self.device)  # (N,2)

                trg_all = fuse_trg_desc[0, 0, :, :]  # [P,D]
                distances_this_image: List[float] = []

                for i in range(src_kps.shape[0]):
                    kp_src = src_kps[i]
                    kp_trg = trg_kps[i]

                    if (kp_src[0] < 0) or (kp_src[1] < 0) or (kp_trg[0] < 0) or (kp_trg[1] < 0):
                        continue

                    # pixel (es 768) -> patch index (48x48)
                    x_idx, y_idx = pixel_to_patch_idx(
                        kp_src,
                        stride=self.sd_stride,
                        grid_hw=self.featmap_size,
                        img_hw=(self.H * self.sd_stride, self.W * self.sd_stride),
                    )
                    patch_index_src = int(y_idx) * self.W + int(x_idx)

                    src_vec = fuse_src_desc[0, 0, patch_index_src, :]  # [D]

                    # SOLO L2: sim = -||x-y||^2
                    diff = trg_all - src_vec.unsqueeze(0)               # [P,D]
                    sim_1d = -(diff * diff).sum(dim=-1)                 # [P]
                    sim2d = sim_1d.view(self.H, self.W)                 # [H,W]

                    if self.win_soft_argmax:
                        y_tok, x_tok = soft_argmax_window(
                            sim2d, window_radius=self.wsam_win_size, temperature=self.wsam_beta
                        )
                    else:
                        y_tok, x_tok = soft_argmax_window(sim2d, window_radius=1)

                    x_pred, y_pred = patch_idx_to_pixel((x_tok, y_tok), stride=self.sd_stride)

                    dx = float(x_pred) - float(kp_trg[0].item())
                    dy = float(y_pred) - float(kp_trg[1].item())
                    distances_this_image.append(math.sqrt(dx * dx + dy * dy))

                results.append(
                    CorrespondenceResult(
                        category=sd_batch["category"],
                        distances=distances_this_image,
                        pck_threshold_0_05=get_pckthres(sd_batch["trg_bndbox"], 0.05),
                        pck_threshold_0_1=get_pckthres(sd_batch["trg_bndbox"], 0.1),
                        pck_threshold_0_2=get_pckthres(sd_batch["trg_bndbox"], 0.2),
                    )
                )

        return results
