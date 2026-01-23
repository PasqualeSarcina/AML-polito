import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.ap10k import AP10KDataset
from data.pfpascal import PFPascalDataset
from data.pfwillow import PFWillowDataset
from data.spair import SPairDataset
from models.dinov2.PreProcess import PreProcess
from utils.soft_argmax_window import soft_argmax_window
from utils.utils_convert import pixel_to_patch_idx, patch_idx_to_pixel
from utils.utils_featuremaps import save_featuremap, load_featuremap
from utils.utils_init_dataloader import init_dataloader
from utils.utils_results import CorrespondenceResult


class Dinov2Eval:
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.custom_weights = args.custom_weights
        self.win_soft_argmax = args.win_soft_argmax
        self.wsam_win_size = args.wsam_win_size
        self.wsam_beta = args.wsam_beta
        self.device = args.device
        self.base_dir = args.base_dir

        self._init_model()

        transform = PreProcess(out_dim=(518, 518))
        self.dataset, self.dataloader = init_dataloader(self.dataset_name, 'test', transform=transform)

        self.feat_dir = Path(self.base_dir) / "data" / "features" / "dinov2"
        self.processed_img = defaultdict(set)

    def _init_model(self):
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        if self.custom_weights:
            checkpoint = torch.load(self.custom_weights, map_location=self.device)
            state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint

            # Clean up key names (e.g., remove "module." prefix)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # Load weights into the model
            model.load_state_dict(new_state_dict, strict=False)

        self.model = model.to(self.device).eval()

    def compute_features(self, img_tensor: torch.Tensor, img_name: str,
                          category: str) -> torch.Tensor:
        if self.dataset_name == "ap-10k":
            category = "all"
        if img_name in self.processed_img[category]:
            dict_out = load_featuremap(img_name, self.feat_dir, device=self.device)
            return dict_out

        dict_out = self.model.forward_features(img_tensor.to(self.device).unsqueeze(0))
        featmap = dict_out["x_norm_patchtokens"]
        self.processed_img[category].add(img_name)
        save_featuremap(featmap, img_name, self.feat_dir)
        return featmap

    def evaluate(self) -> list[CorrespondenceResult]:
        results = []

        out_h, out_w = 518, 518
        patch_size = 14
        w_grid = out_w // patch_size  # 37
        h_grid = out_h // patch_size  # 37

        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in tqdm(self.dataloader, total=len(self.dataloader), desc=f"Computing correspondences with DINOv2 on {self.dataset_name}",
                              smoothing=0.1, mininterval=0.7, maxinterval=2.0):
                category = batch["category"]

                # featuremaps salvate su immagini resized 518x518
                feats_src = self.compute_features(batch["src_img"], batch["src_imname"], category) # Resized image
                feats_trg = self.compute_features(batch["trg_img"], batch["trg_imname"], category) # Resized image

                feats_src = feats_src.to(self.device)
                feats_trg = feats_trg.to(self.device)

                # se includono CLS token (1 + 1369), rimuovilo
                if feats_src.ndim == 3 and feats_src.shape[1] == 1 + h_grid * w_grid:
                    feats_src = feats_src[:, 1:, :]
                if feats_trg.ndim == 3 and feats_trg.shape[1] == 1 + h_grid * w_grid:
                    feats_trg = feats_trg[:, 1:, :]

                # keypoints
                src_kps = batch["src_kps"].to(self.device)  # (N,2) in resized 518x518
                trg_kps = batch["trg_kps"].to(self.device)  # (N,2) in resized 518x518

                distances_this_image: list[float] = []

                # loop keypoints
                n_kps = min(src_kps.shape[0], trg_kps.shape[0])
                for i in range(n_kps):
                    kp_src = src_kps[i]
                    kp_trg = trg_kps[i]

                    # skip keypoints "invalidi" tipici (es. -1, -1) o NaN
                    if torch.isnan(kp_src).any() or torch.isnan(kp_trg).any():
                        continue

                    # --- SRC kp (in 518) -> patch index ---
                    x_patch, y_patch = pixel_to_patch_idx(
                        kp_src,
                        stride=patch_size,
                        grid_hw=(h_grid, w_grid),
                        img_hw=(out_h, out_w),
                    )

                    patch_index_src = y_patch * w_grid + x_patch

                    # --- source vector ---
                    source_vec = feats_src[0, patch_index_src, :]  # (D,)

                    # --- similarity su tutti i patch target ---
                    sim_1d = torch.nn.functional.cosine_similarity(
                        feats_trg[0], source_vec.unsqueeze(0), dim=-1
                    )  # (1369,)
                    sim_2d = sim_1d.view(h_grid, w_grid)  # (37,37)

                    if self.win_soft_argmax:
                        y_pred_patch, x_pred_patch = soft_argmax_window(sim_2d, window_radius=self.wsam_win_size, temperature=self.wsam_beta)  # ritorna x,y
                    else:
                        y_pred_patch, x_pred_patch = soft_argmax_window(sim_2d, window_radius=1)

                    # --- 518 -> ORIG target  ---
                    x_pred, y_pred = patch_idx_to_pixel((x_pred_patch, y_pred_patch), stride=patch_size)

                    gt_x = float(kp_trg[0].item())
                    gt_y = float(kp_trg[1].item())

                    dx = x_pred - gt_x
                    dy = y_pred - gt_y
                    dist = math.sqrt(dx * dx + dy * dy)
                    distances_this_image.append(dist)

                results.append(
                    CorrespondenceResult(
                        category=category,
                        distances=distances_this_image,
                        pck_threshold_0_05=batch["pck_threshold_0_05"], # Resized PCK thresholds
                        pck_threshold_0_1=batch["pck_threshold_0_1"], # Resized PCK thresholds
                        pck_threshold_0_2=batch["pck_threshold_0_2"], # Resized PCK thresholds
                    )
                )

        return results
