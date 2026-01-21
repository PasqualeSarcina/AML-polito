import gc
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.ap10k import AP10KDataset
from data.pfpascal import PFPascalDataset
from data.pfwillow import PFWillowDataset
from data.spair import SPairDataset
from models.dift.PreProcess import PreProcess
from models.dift.SDFeaturizer import SDFeaturizer
from utils.soft_argmax_window import soft_argmax_window
from utils.utils_convert import pixel_to_patch_idx, patch_idx_to_pixel
from utils.utils_featuremaps import save_featuremap, load_featuremap
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

        self.feat_dir = Path(self.base_dir) / "data" / "features" / "dift"

        self._init_dataset()
        self.processed_img = defaultdict(set)

        categories = self.dataset.get_categories()
        self.prompt_embeds = self.featurizer.encode_category_prompts(categories)

    def _init_dataset(self):
        transforms = PreProcess(ensemble_size=self.enseble_size)
        match self.dataset_name:
            case 'spair-71k':
                self.dataset = SPairDataset(datatype='test', transform=transforms, dataset_size='large')
            case 'pf-pascal':
                self.dataset = PFPascalDataset(datatype='test', transform=transforms)
            case 'pf-willow':
                self.dataset = PFWillowDataset(datatype='test', transform=transforms)
            case 'ap-10k':
                self.dataset = AP10KDataset(datatype='test', transform=transforms)

        def collate_single(batch_list):
            return batch_list[0]

        self.dataloader = DataLoader(self.dataset, num_workers=4, batch_size=1, collate_fn=collate_single)

    def compute_features(self, img_tensor: torch.Tensor, img_name: str,
                          category: str, up_ft_index: list[int] | int = 1, t:int = 261) -> torch.Tensor:
        if self.dataset_name == "ap-10k":
            category_opt = "all"
        else:
            category_opt = category
        if img_name in self.processed_img[category_opt]:
            unet_ft = load_featuremap(img_name, self.feat_dir, self.device)
            return unet_ft

        prompt_embed = self.prompt_embeds[category_opt]  # (1,77,dim)

        unet_ft = self.featurizer.forward(
            img_tensor=img_tensor,
            prompt_embed=prompt_embed,
            ensemble_size=self.enseble_size,
            up_ft_index=up_ft_index,
            t=t
        )  # (1,c,h,w)
        save_featuremap(unet_ft, img_name, self.feat_dir)
        self.processed_img[category_opt].add(img_name)
        return unet_ft

    def evaluate(self) -> list[CorrespondenceResult]:
        results = []

        # input DIFT (dopo preprocess) e grid feature
        OUT_H = OUT_W = 768
        HV = WV = 48
        PATCH = OUT_W // WV   # 768/48 = 16  (patch stride effettivo)

        with torch.no_grad():
            for batch in tqdm(self.dataloader, total=len(self.dataloader),
                              desc=f"DIFT Eval on {self.dataset_name}"):

                category = batch["category"]

                # features: [1,C,48,48]
                src_ft = self.compute_features(batch["src_img"], batch["src_imname"], category)
                trg_ft = self.compute_features(batch["trg_img"], batch["trg_imname"], category)

                if src_ft.ndim == 3:  # [C,48,48] -> [1,C,48,48]
                    src_ft = src_ft.unsqueeze(0)
                if trg_ft.ndim == 3:
                    trg_ft = trg_ft.unsqueeze(0)

                # keypoints già nello spazio 768×768
                src_kps = batch["src_kps"].to(self.device)  # (N,2) in 768
                trg_kps = batch["trg_kps"].to(self.device)  # (N,2) in 768

                C = src_ft.shape[1]
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

                    # ---- src feature vector ----
                    src_vec = src_ft[0, :, y_idx, x_idx].view(C, 1, 1)  # [C,1,1]

                    # ---- similarity map in token space (48x48) ----
                    sim2d = torch.nn.functional.cosine_similarity(trg_ft[0], src_vec, dim=0)  # [48,48]

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
