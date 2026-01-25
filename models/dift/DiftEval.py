import math
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
import torch.nn.functional as F

from models.dift.PreProcess import PreProcess
from models.dift.SDFeaturizer import SDFeaturizer
from utils.soft_argmax_window import soft_argmax_window
from utils.utils_convert import pixel_to_patch_idx, patch_idx_to_pixel
from utils.utils_featuremaps import save_featuremap, load_featuremap
from utils.utils_init_dataloader import init_dataloader
from utils.utils_results import CorrespondenceResult


class DiftEval:
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.win_soft_argmax = args.win_soft_argmax
        self.wsam_win_size = args.wsam_win_size
        self.wsam_beta = args.wsam_beta
        self.device = args.device
        self.base_dir = args.base_dir
        self.enseble_size = args.ensemble_size

        self.featurizer = SDFeaturizer(device=self.device)

        self.sd_stride = 16
        self.featmap_size: tuple[int, int] = (48, 48)  # H,W
        self.H, self.W = self.featmap_size
        self.P = self.H * self.W

        self.feat_dir = Path(self.base_dir) / "data" / "features" / "dift"

        transform = PreProcess(ensemble_size=self.enseble_size)
        self.dataset, self.dataloader = init_dataloader(self.dataset_name, 'test', transform=transform)

        self.processed_img = defaultdict(set)

        categories = self.dataset.get_categories()
        self.prompt_embeds = self.featurizer.encode_category_prompts(categories)

    def compute_features(
        self,
        img_tensor: torch.Tensor,
        img_name: str,
        category: str,
        up_ft_index: list[int] | int = 1,
        t: int = 261
    ) -> torch.Tensor:
        """
        Estrae feature UNet per un SOLO timestep t, ma può estrarre più blocchi insieme
        se up_ft_index è una lista.

        IMPORTANTE: la cache deve includere t e up_ft_index, altrimenti ricicli feature sbagliate.
        """
        if self.dataset_name == "ap-10k":
            category_opt = "all"
        else:
            category_opt = category

        prompt_embed = self.prompt_embeds[category_opt]

        # ---- chiave cache unica (include t e blocchi) ----
        if isinstance(up_ft_index, list):
            up_str = "-".join(map(str, up_ft_index))
        else:
            up_str = str(up_ft_index)

        feat_key = f"{img_name}_t{t}_up{up_str}"

        if feat_key in self.processed_img[category_opt]:
            unet_ft = load_featuremap(feat_key, self.feat_dir, self.device)
            return unet_ft

        unet_ft = self.featurizer.forward(
            img_tensor=img_tensor,
            prompt_embed=prompt_embed,
            ensemble_size=self.enseble_size,
            up_ft_index=up_ft_index,
            t=t
        )

        save_featuremap(unet_ft, feat_key, self.feat_dir)
        self.processed_img[category_opt].add(feat_key)
        return unet_ft

    def _compute_pca(self, sd_src_featmap, sd_trg_featmap):
        """
        co-PCA su 3 scale (s5,s4,s3) e costruzione feature map ridotte:
          - input: liste/tuple di 3 tensori (s5,s4,s3), ciascuno [C,H,W] o [1,C,H,W]
          - output: src_proc, trg_proc: [1, Dsd, Hc, Wc]  (Hc,Wc = self.featmap_size)
                    dims_used: [d_s5, d_s4, d_s3]

        Modifica importante per cosine+argmax:
        - normalizzazione per-blocco (scala) + peso per-blocco prima della concat
          così il peso del layer di mezzo resta “significativo”.
        """
        PCA_DIMS = [256, 512, 256]      # s5,s4,s3
        LAYER_W  = [1.0, 2.0, 1.0]      # più importanza al layer di mezzo (s4)

        dims_used = []
        src_red_list = []
        trg_red_list = []

        src_layers = [sd_src_featmap[0], sd_src_featmap[1], sd_src_featmap[2]]
        trg_layers = [sd_trg_featmap[0], sd_trg_featmap[1], sd_trg_featmap[2]]

        for i, out_dim in enumerate(PCA_DIMS):
            fs = src_layers[i]
            ft = trg_layers[i]

            if fs.ndim == 3:
                fs = fs.unsqueeze(0)
            if ft.ndim == 3:
                ft = ft.unsqueeze(0)

            if fs.shape[-2:] != self.featmap_size:
                fs = F.interpolate(fs, size=self.featmap_size, mode="bilinear", align_corners=False)
            if ft.shape[-2:] != self.featmap_size:
                ft = F.interpolate(ft, size=self.featmap_size, mode="bilinear", align_corners=False)

            _, C, H, W = fs.shape
            P = H * W
            q = min(out_dim, C)
            dims_used.append(q)

            fs_tok = fs.permute(0, 2, 3, 1).reshape(P, C)  # [P,C]
            ft_tok = ft.permute(0, 2, 3, 1).reshape(P, C)  # [P,C]

            X = torch.cat([fs_tok, ft_tok], dim=0)         # [2P,C]
            mean = X.mean(dim=0, keepdim=True)
            Xc = (X - mean).float()

            _, _, V = torch.pca_lowrank(Xc, q=q)           # V: [C,q]
            Z = Xc @ V                                     # [2P,q]

            Zs = Z[:P, :]
            Zt = Z[P:, :]

            fs_red = Zs.reshape(1, H, W, q).permute(0, 3, 1, 2).contiguous()  # [1,q,H,W]
            ft_red = Zt.reshape(1, H, W, q).permute(0, 3, 1, 2).contiguous()  # [1,q,H,W]

            # ---- block-normalize (per pixel) + peso scala ----
            fs_red = F.normalize(fs_red, p=2, dim=1, eps=1e-6)
            ft_red = F.normalize(ft_red, p=2, dim=1, eps=1e-6)

            fs_red = fs_red * LAYER_W[i]
            ft_red = ft_red * LAYER_W[i]

            src_red_list.append(fs_red)
            trg_red_list.append(ft_red)

        src_proc = torch.cat(src_red_list, dim=1)  # [1,Dsd,H,W]
        trg_proc = torch.cat(trg_red_list, dim=1)  # [1,Dsd,H,W]

        return src_proc, trg_proc, dims_used

    def evaluate(self) -> list["CorrespondenceResult"]:
        results = []

        # input DIFT (dopo preprocess) e grid feature
        OUT_H = OUT_W = 768
        HV = WV = 48
        PATCH = OUT_W // WV   # 768/48 = 16

        # ---- multi-timestep fusion (senza toccare forward) ----
        T_LIST = [10, 51, 261]
        T_W    = [0.2, 0.4, 0.4]   # più peso a mid e mid-high

        with torch.no_grad():
            for batch in tqdm(self.dataloader, total=len(self.dataloader),
                              desc=f"DIFT Eval on {self.dataset_name}"):

                category = batch["category"]

                # === 1) Estrai features per ogni timestep ===
                src_list = []
                trg_list = []

                for t in T_LIST:
                    src_ft_t = self.compute_features(
                        batch["src_img"], batch["src_imname"], category,
                        up_ft_index=[0, 1, 2], t=t
                    )
                    trg_ft_t = self.compute_features(
                        batch["trg_img"], batch["trg_imname"], category,
                        up_ft_index=[0, 1, 2], t=t
                    )

                    # co-PCA per questo timestep (include già multi-layer)
                    src_p, trg_p, _ = self._compute_pca(src_ft_t, trg_ft_t)

                    # opzionale: normalizzazione globale (ok, ma non indispensabile con cosine_similarity)
                    src_p = F.normalize(src_p, p=2, dim=1, eps=1e-6)
                    trg_p = F.normalize(trg_p, p=2, dim=1, eps=1e-6)

                    src_list.append(src_p)
                    trg_list.append(trg_p)

                # keypoints già nello spazio 768×768
                src_kps = batch["src_kps"].to(self.device)  # (N,2) in 768
                trg_kps = batch["trg_kps"].to(self.device)  # (N,2) in 768

                distances_this_image: list[float] = []

                n_kps = min(src_kps.shape[0], trg_kps.shape[0])
                for i in range(n_kps):
                    kp_src = src_kps[i]   # (x,y) in 768
                    kp_trg = trg_kps[i]   # (x,y) in 768

                    if torch.isnan(kp_src).any() or torch.isnan(kp_trg).any():
                        continue

                    # ---- SRC pixel(768) -> token idx (48x48) ----
                    x_idx, y_idx = pixel_to_patch_idx(
                        kp_src,
                        stride=PATCH,
                        grid_hw=(HV, WV),
                        img_hw=(OUT_H, OUT_W)
                    )

                    # === 2) Similarity fusion (somma pesata delle cosine-map) ===
                    sim2d = 0.0
                    for k, w in enumerate(T_W):
                        src_ft = src_list[k]  # [1,C,48,48]
                        trg_ft = trg_list[k]  # [1,C,48,48]

                        Ck = src_ft.shape[1]
                        src_vec = src_ft[0, :, y_idx, x_idx].view(Ck, 1, 1)  # [C,1,1]

                        sim_k = F.cosine_similarity(trg_ft[0], src_vec, dim=0)  # [48,48]
                        sim2d = sim2d + w * sim_k

                    # ---- pred token coords (y,x) ----
                    if self.win_soft_argmax:
                        y_tok, x_tok = soft_argmax_window(
                            sim2d,
                            window_radius=self.wsam_win_size,
                            temperature=self.wsam_beta
                        )
                    else:
                        y_tok, x_tok = soft_argmax_window(sim2d, window_radius=1)

                    # ---- token -> pixel nello spazio 768 (centro patch) ----
                    x_pred, y_pred = patch_idx_to_pixel((x_tok, y_tok), stride=PATCH)

                    dx = x_pred - float(kp_trg[0].item())
                    dy = y_pred - float(kp_trg[1].item())
                    dist = math.sqrt(dx * dx + dy * dy)
                    distances_this_image.append(dist)

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
