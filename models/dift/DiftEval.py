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
from models.dift.SDFeaturizer import SDFeaturizer
from utils.utils_correspondence import argmax
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

    @staticmethod
    def _resize_image(
            sample: torch.Tensor,
            img_output_size: tuple[int, int] = (768, 768),  # (H, W)
            ensemble_size: int = 4,
    ) -> torch.Tensor:
        """
        sample: (C,H,W) con valori 0..255 (uint8 o float)
        return: (E,C,H',W') normalizzata in [-1,1]
        """
        if sample.ndim != 3:
            raise ValueError(f"`sample` deve essere (C,H,W). Trovato shape={tuple(sample.shape)}")

        resize = transforms.Resize(
            img_output_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )

        img = resize(sample)  # (C,H',W')
        img = img.to(torch.float32)
        img = (img / 255.0 - 0.5) * 2.0  # [-1,1]
        img = img.unsqueeze(0).repeat(ensemble_size, 1, 1, 1)  # (E,C,H',W')
        return img

    def _compute_features(self, img_tensor: torch.Tensor, img_name: str,
                          category: str) -> torch.Tensor:
        if self.dataset_name == "ap-10k":
            category_opt = "all"
        else:
            category_opt = category
        if img_name in self.processed_img[category_opt]:
            unet_ft = load_featuremap(img_name, self.feat_dir, self.device)
            return unet_ft

        img_tensor_resized = self._resize_image(img_tensor)  # (E,C,H',W')
        prompt_embed = self.prompt_embeds[category_opt]  # (1,77,dim)

        unet_ft = self.featurizer.forward(
            img_tensor=img_tensor_resized,
            prompt_embed=prompt_embed,
            ensemble_size=self.enseble_size
        )  # (1,c,h,w)
        save_featuremap(unet_ft, img_name, self.feat_dir)
        self.processed_img[category_opt].add(img_name)
        print(unet_ft.size())
        return unet_ft

    @staticmethod
    def _kp_src_to_featmap(
            kp_orig: torch.Tensor,
            orig_hw: tuple[int, int],  # (H, W) originali
            feat_hw: tuple[int, int] = (48, 48),
            pre_hw: tuple[int, int] = (768, 768),
    ) -> torch.Tensor:
        """
        Converte keypoint originali (pixel) -> indici sulla feature map (x_idx, y_idx).

        Args:
            kp_orig: (N,2) keypoint in pixel sull'immagine originale, ordine (x,y)
            orig_hw: (H,W) dell'immagine originale
            feat_hw: (hv,wv) della feature map (es. 48x48)
            pre_hw:  (Hpre,Wpre) della preprocessata (es. 768x768)

        Returns:
            (N,2) indici featuremap in ordine (x_idx, y_idx)
        """
        if kp_orig.ndim != 2 or kp_orig.shape[-1] != 2:
            raise ValueError(f"`kp_orig` deve essere (N,2). Trovato shape={tuple(kp_orig.shape)}")

        H, W = orig_hw
        hv, wv = feat_hw
        Hpre, Wpre = pre_hw

        kp = kp_orig.to(torch.float32)

        # -------------------------
        # FASE 1) RESIZE (orig -> preprocess)
        # -------------------------
        sx_img = Wpre / W
        sy_img = Hpre / H

        # mapping coerente coi centri pixel: (x+0.5)*s - 0.5
        x_pre = (kp[:, 0] + 0.5) * sx_img - 0.5
        y_pre = (kp[:, 1] + 0.5) * sy_img - 0.5

        # -------------------------
        # FASE 2) TRASLAZIONE su FEATURE MAP (preprocess -> feat idx)
        # -------------------------
        sx_feat = Wpre / wv  # stride in preprocess (es. 16)
        sy_feat = Hpre / hv  # stride in preprocess (es. 16)

        # "center rule": token center a (i+0.5)*stride  => i ≈ x/stride - 0.5
        x_idx = torch.round(x_pre / sx_feat - 0.5).long()
        y_idx = torch.round(y_pre / sy_feat - 0.5).long()

        # clamp ai limiti della feature map
        x_idx = x_idx.clamp(0, wv - 1)
        y_idx = y_idx.clamp(0, hv - 1)

        return torch.stack([x_idx, y_idx], dim=1)  # (N,2) (x_idx,y_idx)

    def evaluate(self) -> list[CorrespondenceResult]:
        results = []

        with torch.no_grad():
            for batch in tqdm(
                    self.dataloader,
                    total=len(self.dataloader),
                    desc=f"DIFT Eval on {self.dataset_name}"
            ):
                category = batch["category"]

                orig_size_src = tuple(batch["src_imsize"][-2:])  # (H, W)
                orig_size_trg = tuple(batch["trg_imsize"][-2:])  # (H, W)

                src_imname = batch["src_imname"]
                trg_imname = batch["trg_imname"]

                src_ft = self._compute_features(batch["src_img"], batch["src_imname"], category)
                trg_ft = self._compute_features(batch["trg_img"], batch["trg_imname"], category)

                # Keypoints & metadata
                src_kps = batch["src_kps"].to(self.device)  # [N,2]
                trg_kps = batch["trg_kps"].to(self.device)
                # [N,2]
                pck_thr_0_05 = batch["pck_threshold_0_05"]
                pck_thr_0_1 = batch["pck_threshold_0_1"]
                pck_thr_0_2 = batch["pck_threshold_0_2"]

                # src_ft = nn.Upsample(size=orig_size_src, mode='bilinear')(src_ft)
                # trg_ft = nn.Upsample(size=orig_size_trg, mode='bilinear')(trg_ft)

                src_kps = self._kp_src_to_featmap(src_kps, orig_size_src)
                # trg_kps = kp_src_to_featmap(trg_kps, orig_size_trg)

                distances_this_image = []

                N_kps = src_kps.shape[0]
                C = src_ft.shape[1]  # 1280
                hv_t, wv_t = trg_ft.shape[-2:]  # 48,48

                for i in range(N_kps):
                    src_kp = src_kps[i]  # (x,y) originali
                    trg_kp = trg_kps[i]  # (x,y) originali

                    # se nel dataset possono esserci kp non validi, controlla qui
                    if torch.isnan(src_kp).any() or torch.isnan(trg_kp).any():
                        continue

                    # 1) indice sulla featuremap (48x48) per il kp sorgente
                    x_src_idx = int(src_kp[0].item())
                    y_src_idx = int(src_kp[1].item())

                    # 2) vettore feature sorgente (C,1,1) per broadcasting
                    src_vec = src_ft[0, :, y_src_idx, x_src_idx].view(C, 1, 1)

                    # 3) mappa similarità su target (48,48)
                    sim2d = torch.cosine_similarity(trg_ft[0], src_vec, dim=0)

                    sim_int = torch.nn.functional.interpolate(
                        sim2d.unsqueeze(0).unsqueeze(0),
                        size=(768, 768),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0).squeeze(0)

                    if self.win_soft_argmax:
                        # windowed soft-argmax
                        x_pre, y_pre = argmax(
                            sim_int,
                            window_size=self.wsam_win_size,
                            beta=self.wsam_beta
                        )
                    else:
                        # hard argmax
                        x_pre, y_pre = argmax(sim_int, window_size=1)

                    Ht, Wt = orig_size_trg
                    x_pred = x_pre * (Wt / 768)
                    y_pred = y_pre * (Ht / 768)

                    # distanza rispetto al GT (trg_point è (x,y))
                    dist = math.sqrt((x_pred - trg_kp[0]) ** 2 + (y_pred - trg_kp[1]) ** 2)
                    distances_this_image.append(dist)

                    del src_vec

                results.append(
                    CorrespondenceResult(
                        category=category,
                        distances=distances_this_image,
                        pck_threshold_0_05=pck_thr_0_05,
                        pck_threshold_0_1=pck_thr_0_1,
                        pck_threshold_0_2=pck_thr_0_2
                    )
                )
        return results
