import math
from copy import deepcopy
from itertools import islice
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

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
from utils.utils_results import CorrespondenceResult


class SdFuseDino:
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.win_soft_argmax = args.win_soft_argmax
        self.wsam_win_size = args.wsam_win_size
        self.wsam_beta = args.wsam_beta
        self.device = args.device
        self.base_dir = args.base_dir

        self.sd = DiftEval(args)
        self.dino = Dinov2Eval(args)

        self.featmap_size: tuple[int, int] = (48, 48)  # DIFT feature map size
        self.dino_stride = 14  # DINO patch stride
        self.sd_stride = 16  # DIFT patch stride

        self.sd_preproc = DiftPreProcess(out_dim=(self.featmap_size[0] * self.sd_stride, self.featmap_size[1] * self.sd_stride))
        self.dino_preproc = Dinov2PreProcess(out_dim=(self.featmap_size[0] * self.dino_stride, self.featmap_size[1] * self.dino_stride))

        self._init_dataset()

    def _init_dataset(self):
        match self.dataset_name:
            case 'spair-71k':
                self.dataset = SPairDataset(datatype='test',dataset_size='small')
            case 'pf-pascal':
                self.dataset = PFPascalDataset(datatype='test')
            case 'pf-willow':
                self.dataset = PFWillowDataset(datatype='test')
            case 'ap-10k':
                self.dataset = AP10KDataset(datatype='test')

        def collate_single(batch_list):
            return batch_list[0]

        self.dataloader = DataLoader(self.dataset, num_workers=4, batch_size=1, collate_fn=collate_single)

    def _compute_features(self, batch: dict):
        sd_batch = deepcopy(batch)
        sd_batch = self.sd_preproc(sd_batch)
        sd_src_featmap = self.sd.compute_features(sd_batch['src_img'], sd_batch['src_imname'], sd_batch['category'])
        sd_trg_featmap = self.sd.compute_features(sd_batch['trg_img'], sd_batch['trg_imname'], sd_batch['category'])
        if sd_src_featmap.ndim == 3:  # [C,48,48] -> [1,C,48,48]
            sd_src_featmap = sd_src_featmap.unsqueeze(0)
        if sd_trg_featmap.ndim == 3:
            sd_trg_featmap = sd_trg_featmap.unsqueeze(0)

        dino_batch = deepcopy(batch)
        dino_batch = self.dino_preproc(dino_batch)
        dino_src_featmap = self.dino.compute_features(dino_batch['src_img'], dino_batch['src_imname'], dino_batch['category'])
        dino_trg_featmap = self.dino.compute_features(dino_batch['trg_img'], dino_batch['trg_imname'], dino_batch['category'])

        # se includono CLS token (1 + 1369), rimuovilo
        if dino_src_featmap.ndim == 3 and dino_src_featmap.shape[1] == 1 + self.featmap_size[0] * self.featmap_size[1]:
            dino_src_featmap = dino_src_featmap[:, 1:, :]
        if dino_trg_featmap.ndim == 3 and dino_trg_featmap.shape[1] == 1 + self.featmap_size[0] * self.featmap_size[1]:
            dino_trg_featmap = dino_trg_featmap[:, 1:, :]

        dino_src_featmap = dino_src_featmap.permute(0, 2, 1).reshape(1, dino_src_featmap.shape[2], self.featmap_size[0], self.featmap_size[1])
        dino_trg_featmap = dino_trg_featmap.permute(0, 2, 1).reshape(1, dino_trg_featmap.shape[2], self.featmap_size[0], self.featmap_size[1])

        return sd_src_featmap, sd_trg_featmap, dino_src_featmap, dino_trg_featmap

    def evaluate(self) -> List[CorrespondenceResult]:
        results = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader, total=len(self.dataloader),
                              desc=f"DIFT + DINOv2 Eval on {self.dataset_name}"):
                sd_src_featmap, sd_trg_featmap, dino_src_featmap, dino_trg_featmap = self._compute_features(batch)

                sd_src_featmap = F.normalize(sd_src_featmap, p=2, dim=1)
                sd_trg_featmap = F.normalize(sd_trg_featmap, p=2, dim=1)
                dino_src_featmap = F.normalize(dino_src_featmap, p=2, dim=1)
                dino_trg_featmap = F.normalize(dino_trg_featmap, p=2, dim=1)

                alpha = 0.5

                fuse_src = torch.cat([alpha * sd_src_featmap, alpha * dino_src_featmap], dim=1)
                fuse_trg = torch.cat([alpha * sd_trg_featmap, alpha * dino_trg_featmap], dim=1)

                batch = self.sd_preproc(batch)

                src_kps = batch["src_kps"].to(self.device)  # (N,2) in 768
                trg_kps = batch["trg_kps"].to(self.device)  # (N,2) in 768

                C = fuse_src.shape[1]

                distances_this_image: list[float] = []

                n_kps = min(src_kps.shape[0], trg_kps.shape[0])
                for i in range(n_kps):
                    kp_src = src_kps[i]  # (x,y) in 768
                    kp_trg = trg_kps[i]  # (x,y) in 768

                    if torch.isnan(kp_src).any() or torch.isnan(kp_trg).any():
                        continue

                    # ---- SRC pixel(768) -> token idx (48x48) ----
                    x_idx, y_idx = pixel_to_patch_idx(
                        kp_src,
                        stride=self.sd_stride,
                        grid_hw=self.featmap_size,
                        img_hw=(self.featmap_size[0] * self.sd_stride, self.featmap_size[1] * self.sd_stride)
                    )

                    # ---- src feature vector ----
                    src_vec = fuse_src[0, :, y_idx, x_idx].view(C, 1, 1)  # [C,1,1]

                    # ---- similarity map in token space (48x48) ----
                    sim2d = torch.nn.functional.cosine_similarity(fuse_trg[0], src_vec, dim=0)  # [48,48]

                    # ---- pred token coords (y,x) ----
                    if self.win_soft_argmax:
                        # la tua soft_argmax_window ritorna (y,x)
                        y_tok, x_tok = soft_argmax_window(
                            sim2d,
                            window_radius=self.wsam_win_size,
                            temperature=self.wsam_beta
                        )
                    else:
                        y_tok, x_tok = soft_argmax_window(sim2d, window_radius=1)

                    # ---- token -> pixel nello spazio 768 (centro patch) ----
                    x_pred, y_pred = patch_idx_to_pixel((x_tok, y_tok), stride=self.sd_stride)

                    dx = x_pred - float(kp_trg[0].item())
                    dy = y_pred - float(kp_trg[1].item())
                    dist = math.sqrt(dx * dx + dy * dy)
                    distances_this_image.append(dist)

                results.append(
                    CorrespondenceResult(
                        category=batch["category"],
                        distances=distances_this_image,
                        pck_threshold_0_05=get_pckthres(batch["trg_bndbox"], 0.05),  # Resized PCK thresholds
                        pck_threshold_0_1=get_pckthres(batch["trg_bndbox"], 0.1),    # Resized PCK thresholds
                        pck_threshold_0_2=get_pckthres(batch["trg_bndbox"], 0.2)     # Resized PCK thresholds
                    )
                )

            return results




