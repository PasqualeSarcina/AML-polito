import math
import os
from collections import defaultdict
from pathlib import Path

import torch
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ap10k import AP10KDataset
from data.pfpascal import PFPascalDataset
from data.pfwillow import PFWillowDataset
from data.spair import SPairDataset
from models.sam.PreProcess import PreProcess
from utils.soft_argmax_window import soft_argmax_window
from utils.utils_convert import pixel_to_patch_idx, patch_idx_to_pixel
from utils.utils_download import download
from utils.utils_featuremaps import save_featuremap, load_featuremap
from utils.utils_results import CorrespondenceResult


class SamEval:
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.custom_weights = args.custom_weights
        self.win_soft_argmax = args.win_soft_argmax
        self.wsam_win_size = args.wsam_win_size
        self.wsam_beta = args.wsam_beta
        self.device = args.device
        # self.using_colab = args.using_colab
        self.base_dir = args.base_dir
        self._init_model()
        self._init_dataset()

        self.feat_dir = Path(self.base_dir) / "data" / "features" / "sam"
        self.processed_img = defaultdict(set)

    def _init_model(self):
        if self.custom_weights is not None:
            sam_checkpoint = self.custom_weights
        else:
            sam_checkpoint = Path(self.base_dir) / "checkpoints" / "sam_vit_b_01ec64.pth"

        if not os.path.exists(sam_checkpoint):
            if self.custom_weights is None:
                download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", sam_checkpoint)
            else:
                raise FileNotFoundError(f"SAM checkpoint not found at {sam_checkpoint}")

        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.transform = self.predictor.transform

    def _init_dataset(self):
        transform = PreProcess(sam_transform=self.transform)
        match self.dataset_name:
            case 'spair-71k':
                self.dataset = SPairDataset(datatype='test', transform=transform, dataset_size='large')
            case 'pf-pascal':
                self.dataset = PFPascalDataset(datatype='test', transform=transform)
            case 'pf-willow':
                self.dataset = PFWillowDataset(datatype='test', transform=transform)
            case 'ap-10k':
                self.dataset = AP10KDataset(datatype='test', transform=transform)

        def collate_single(batch_list):
            return batch_list[0]

        self.dataloader = DataLoader(self.dataset, num_workers=4, batch_size=1, collate_fn=collate_single)

    def _compute_features(self, img_tensor: torch.Tensor, img_size: torch.Size, img_name: str,
                          category: str) -> torch.Tensor:
        if self.dataset_name == "ap-10k":
            category = "all"
        if img_name in self.processed_img[category]:
            # Carica featuremap salvata
            img_emb = load_featuremap(img_name, self.feat_dir, self.device)  # [C,h',w']
            return img_emb

        img_tensor = img_tensor.to(self.device)  # [1,3,H,W]
        self.predictor.set_torch_image(img_tensor, img_size)
        img_emb = self.predictor.get_image_embedding()[0]  # [C,h',w']
        self.processed_img[category].add(img_name)
        save_featuremap(img_emb, img_name, self.feat_dir)

        return img_emb

    def evaluate(self) -> list[CorrespondenceResult]:
        results = []
        IMG_SIZE = self.predictor.model.image_encoder.img_size  # 1024
        PATCH = int(self.predictor.model.image_encoder.patch_embed.proj.kernel_size[0])
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in tqdm(
                    self.dataloader,
                    total=len(self.dataloader),
                    desc=f"Computing correspondences with SAM on {self.dataset_name}",
                    smoothing=0.1,
                    mininterval=0.7,
                    maxinterval=2.0
            ):
                category = batch["category"]

                src_imname = batch["src_imname"]
                trg_imname = batch["trg_imname"]

                src_emb = self._compute_features(batch["src_img"], batch["src_orig_size"], src_imname, category)
                trg_emb = self._compute_features(batch["trg_img"], batch["trg_orig_size"], trg_imname, category)

                # Keypoints (N,2) in pixel originali, ordine (x,y)
                src_kps = batch["src_kps"].to(self.device)
                trg_kps = batch["trg_kps"].to(self.device)

                # Dimensioni originali (H,W) e resized (new_h,new_w) già nel batch
                Hs_prime, Ws_prime = batch["src_resized_size"]

                Ht_prime, Wt_prime = batch["trg_resized_size"]

                # Regione valida in token-space (NO padding), con PATCH tipico SAM = 16
                hv_s = (Hs_prime + PATCH - 1) // PATCH
                wv_s = (Ws_prime + PATCH - 1) // PATCH

                hv_t = (Ht_prime + PATCH - 1) // PATCH
                wv_t = (Wt_prime + PATCH - 1) // PATCH

                # (opzionale) clamp su griglia encoder (di solito 1024/16 = 64)
                grid = IMG_SIZE // PATCH  # es. 1024//16 = 64
                hv_s = min(hv_s, grid)
                wv_s = min(wv_s, grid)
                hv_t = min(hv_t, grid)
                wv_t = min(wv_t, grid)

                # ---- taglia embeddings alla regione valida ----
                # src_valid: [C, hv_s, wv_s], trg_valid: [C, hv_t, wv_t]
                src_valid = src_emb[:, :hv_s, :wv_s]
                trg_valid = trg_emb[:, :hv_t, :wv_t]

                C_ft = trg_valid.shape[0]
                trg_flat = trg_valid.permute(1, 2, 0).reshape(-1, C_ft)  # [Pvalid, C]

                N_kps = src_kps.shape[0]
                distances_this_image = []

                # -------------------------
                # Loop keypoints
                # -------------------------
                for i in range(N_kps):
                    src_kp = src_kps[i]
                    trg_kp = trg_kps[i]

                    if torch.isnan(src_kp).any() or torch.isnan(trg_kp).any():
                        continue

                    # ---- src pixel (resized) -> src token idx ----
                    x_idx, y_idx = pixel_to_patch_idx(
                        src_kp,
                        stride=PATCH,
                        grid_hw=(hv_s, wv_s),
                        img_hw=(Hs_prime, Ws_prime),
                    )

                    # vettore feature sorgente: [C]
                    src_vec = src_valid[:, y_idx, x_idx]

                    # cosine similarity su tutti i token target validi
                    sim = torch.nn.functional.cosine_similarity(trg_flat, src_vec.unsqueeze(0), dim=1)  # [Pvalid]
                    sim_2d = sim.view(hv_t, wv_t)  # [hv_t, wv_t]

                    # soft-argmax o argmax classico
                    if self.win_soft_argmax:
                        y_pred_patch, x_pred_patch = soft_argmax_window(sim_2d, window_radius=self.wsam_win_size,
                                                                        temperature=self.wsam_beta)
                    else:
                        y_pred_patch, x_pred_patch = soft_argmax_window(sim_2d, window_radius=1)

                    # ---- token -> pixel nello spazio resized ----
                    x_pred, y_pred = patch_idx_to_pixel((x_pred_patch, y_pred_patch), stride=PATCH)

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
