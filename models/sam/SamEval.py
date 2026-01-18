import math
import os
from pathlib import Path

import torch
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ap10k import AP10KDataset
from data.pfpascal import PFPascalDataset
from data.pfwillow import PFWillowDataset
from data.spair import SPairDataset
from utils.utils_correspondence import argmax
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
        #self.using_colab = args.using_colab
        self.base_dir = args.base_dir
        self._init_model()
        self._init_dataset()

        self.feat_dir = Path(self.base_dir) / "data" / "features" / "sam"


    def _init_model(self):
        if self.custom_weights is not None:
            sam_checkpoint = self.custom_weights
        else:
            sam_checkpoint = os.path.join(self.base_dir, "checkpoints/sam_vit_b_01ec64.pth")

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
        self.IMG_SIZE = self.predictor.model.image_encoder.img_size  # 1024
        self.PATCH = int(self.predictor.model.image_encoder.patch_embed.proj.kernel_size[0])


    def _init_dataset(self):
        match self.dataset_name:
            case 'spair-71k':
                self.dataset = SPairDataset(datatype='test', transform=None, dataset_size='large')
            case 'pf-pascal':
                self.dataset = PFPascalDataset(datatype='test', transform=None)
            case 'pf-willow':
                self.dataset = PFWillowDataset(datatype='test', transform=None)
            case 'ap-10k':
                self.dataset = AP10KDataset(datatype='test', transform=None)

        def collate_single(batch_list):
            return batch_list[0]

        self.dataloader = DataLoader(self.dataset, num_workers=4, batch_size=1, collate_fn=collate_single)

    def _compute_features(self):
        print("Saving features to:", self.feat_dir)

        torch.cuda.empty_cache()
        with torch.no_grad():
            for img_name, img_tensor, img_size in tqdm(
                    self.dataset.iter_test_distinct_images(),
                    total=self.dataset.len_test_distinct_images(),
                    desc="Generating embeddings"
            ):
                img_tensor = img_tensor.to(self.device).unsqueeze(0)  # [1,3,H,W]
                orig_size = tuple(img_size[1:])  # (H,W)
                resized = self.predictor.transform.apply_image_torch(img_tensor)  # [1,3,H',W']
                self.predictor.set_torch_image(resized, orig_size)
                img_emb = self.predictor.get_image_embedding()[0]  # [C,h',w']

                save_featuremap(img_emb, img_name, self.feat_dir)

    def _kps_src_to_featmap(self, kps_src: torch.Tensor, img_src_size: torch.Size):
        img_h = int(img_src_size[-2])
        img_w = int(img_src_size[-1])

        # (N,2) coords nella resized (no padding)
        coords = self.predictor.transform.apply_coords_torch(kps_src, (img_h, img_w))  # (N,2)

        img_resized_h, img_resized_w = self.predictor.transform.get_preprocess_shape(img_h, img_w, self.IMG_SIZE)

        xf = torch.floor(coords[:, 0] / self.PATCH).long()
        yf = torch.floor(coords[:, 1] / self.PATCH).long()

        wv = math.ceil(img_resized_w / self.PATCH)
        hv = math.ceil(img_resized_h / self.PATCH)

        xf = xf.clamp(0, wv - 1)
        yf = yf.clamp(0, hv - 1)

        return torch.stack([xf, yf], dim=1)  # (N,2) (x_idx,y_idx)

    def _compute_distances(self) -> list[CorrespondenceResult]:
        results = []
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in tqdm(
                    self.dataloader,
                    total=len(self.dataloader),
                    desc=f"Elaborazione con SAM"
            ):
                category = batch["category"]

                orig_size_src = batch["src_imsize"]  # torch.Size([C, Hs, Ws]) o simile
                orig_size_trg = batch["trg_imsize"]  # torch.Size([C, Ht, Wt]) o simile

                src_imname = batch["src_imname"]
                trg_imname = batch["trg_imname"]

                # Embeddings SAM (C, h, w) tipicamente (256, 64, 64) o simili
                src_emb = load_featuremap(src_imname, self.feat_dir, self.device)  # [C,hs,ws]
                trg_emb = load_featuremap(trg_imname, self.feat_dir, self.device)  # [C,ht,wt]

                # Keypoints (N,2) in pixel originali, ordine (x,y)
                src_kps = batch["src_kps"].to(self.device)
                trg_kps = batch["trg_kps"].to(self.device)

                # -------------------------
                # Target: dimensioni originali + dimensioni resize (no padding)
                # -------------------------
                Ht = int(orig_size_trg[-2])
                Wt = int(orig_size_trg[-1])

                H_prime, W_prime = self.predictor.transform.get_preprocess_shape(
                    Ht, Wt, self.predictor.transform.target_length
                )

                # Regione valida in token (no padding)
                hv_t = (H_prime + self.PATCH - 1) // self.PATCH
                wv_t = (W_prime + self.PATCH - 1) // self.PATCH

                # -------------------------
                # Prepara target flat sulla regione valida
                # -------------------------
                C_ft = trg_emb.shape[0]
                trg_valid = trg_emb[:, :hv_t, :wv_t]  # [C, hv, wv]
                trg_flat = trg_valid.permute(1, 2, 0).reshape(-1, C_ft)  # [Pvalid, C]

                # -------------------------
                # Mappa i keypoint SRC -> indici featuremap SRC (token space)
                # (Assumo che kps_src_to_featmap ritorni (N,2) (x_idx, y_idx) long)
                # -------------------------
                src_kps_idx = self._kps_src_to_featmap(src_kps, orig_size_src)  # (N,2) (x_idx,y_idx)

                N_kps = src_kps_idx.shape[0]
                distances_this_image = []

                # scala SAM (uniforme sul lato lungo)
                if Wt >= Ht:
                    scale = W_prime / Wt
                else:
                    scale = H_prime / Ht

                # -------------------------
                # Loop keypoints
                # -------------------------
                for i in range(N_kps):
                    src_idx = src_kps_idx[i]  # (x_idx, y_idx) su featuremap
                    trg_kp = trg_kps[i]  # (x,y) originale

                    if torch.isnan(src_idx).any() or torch.isnan(trg_kp).any():
                        continue

                    x_idx = int(src_idx[0].item())
                    y_idx = int(src_idx[1].item())

                    # Feature vector sorgente (C,)
                    src_vec = src_emb[:, y_idx, x_idx]  # [C]

                    # Cosine similarity su tutte le posizioni target valide
                    sim = torch.cosine_similarity(trg_flat, src_vec.unsqueeze(0), dim=1)  # [Pvalid]

                    # (hv_t, wv_t) similarity map in token space
                    sim2d = sim.view(hv_t, wv_t)

                    # ------------------------------------------------------------
                    # UPSAMPLE SOLO DELLA SIMILARITY MAP (hard argmax più fine)
                    # token space (hv,wv) -> resized pixel space (H_prime,W_prime)
                    # ------------------------------------------------------------
                    sim_r = torch.nn.functional.interpolate(
                        sim2d[None, None],  # (1,1,hv,wv)
                        size=(H_prime, W_prime),  # resized (no pad)
                        mode="bilinear",
                        align_corners=False
                    )[0, 0]  # (H_prime, W_prime)


                    if self.win_soft_argmax:
                        # windowed soft-argmax
                        x_r, y_r = argmax(
                            sim_r,
                            window_size=self.wsam_win_size,
                            beta=self.wsam_beta
                        )
                    else:
                        # hard argmax
                        x_r, y_r = argmax(sim_r, window_size=1)

                    # resized -> originale
                    x_pred = x_r / scale
                    y_pred = y_r / scale

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


    def evaluate(self) -> list[CorrespondenceResult]:
        # Step 1: compute and save features
        self._compute_features()

        # Step 2: compute distances
        results = self._compute_distances()

        return results