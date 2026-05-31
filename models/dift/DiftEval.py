import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dift.PreProcess import PreProcess
from models.dift.SDFeaturizer import SDFeaturizer
from models.dift.pca import _compute_pca
from utils.soft_argmax_window import soft_argmax_window
from utils.utils_convert import pixel_to_patch_idx, patch_idx_to_pixel
from utils.utils_featuremaps import save_featuremap, load_featuremap
from utils.utils_init_dataloader import init_dataloader
from utils.utils_results import CorrespondenceResult


class DiftEval:
    PCA_DIMS = [256, 256, 256]  # s5,s4,s3

    def __init__(self, args):
        self.dataset_name = args.dataset
        self.wsam_win_radius = args.wsam_win_radius
        self.wsam_temp = args.wsam_temp
        self.device = args.device
        self.base_dir = args.base_dir
        self.enseble_size = args.ensemble_size
        self.timestep = args.timestep
        self.use_blip_prompt = getattr(args, "use_blip_prompt", False)
        self.blip_model_id = "Salesforce/blip-image-captioning-large"

        self.featurizer = SDFeaturizer(device=self.device)

        feat_subdir = "dift_blip" if self.use_blip_prompt else "dift"
        self.feat_dir = Path(self.base_dir) / "data" / "features" / feat_subdir
        self.featmap_size: tuple[int, int] = (48, 48)
        self.H, self.W = self.featmap_size
        self.P = self.H * self.W
        self.sd_stride = 16

        transform = PreProcess(ensemble_size=self.enseble_size)
        self.dataset, self.dataloader = init_dataloader(self.dataset_name, base_dir=self.base_dir, datatype='test', transform=transform)

        self.processed_img = defaultdict(set)

        categories = self.dataset.get_categories()
        self.prompt_embeds = {}
        self.blip_prompt_embeds = {}
        if self.use_blip_prompt:
            self.featurizer.load_blip_captioner(self.blip_model_id)
        else:
            self.prompt_embeds = self.featurizer.encode_category_prompts(categories)

    def compute_features(self, img_tensor: torch.Tensor, img_name: str,
                          category: str, up_ft_index: list[int] | int = 1, t:int = 261) -> torch.Tensor:
        if self.dataset_name == "ap-10k":
            category_opt = "all"
        else:
            category_opt = category
        if img_name in self.processed_img[category_opt]:
            unet_ft = load_featuremap(img_name, self.feat_dir, self.device)
            return unet_ft

        if self.use_blip_prompt:
            prompt_key = (category_opt, img_name)
            if prompt_key not in self.blip_prompt_embeds:
                prompt_embed, caption = self.featurizer.encode_blip_prompt(img_tensor, category)
                self.blip_prompt_embeds[prompt_key] = prompt_embed
                print(f"BLIP prompt for {img_name}: {caption}")
            prompt_embed = self.blip_prompt_embeds[prompt_key]
        else:
            prompt_embed = self.prompt_embeds[category_opt]

        unet_ft = self.featurizer.forward(
            img_tensor=img_tensor,
            prompt_embed=prompt_embed,
            ensemble_size=self.enseble_size,
            up_ft_index=up_ft_index,
            t=self.timestep
        )
        save_featuremap(unet_ft, img_name, self.feat_dir)
        self.processed_img[category_opt].add(img_name)
        return unet_ft

    def evaluate(self) -> list[CorrespondenceResult]:
        results = []

        out_h = self.H * self.sd_stride
        out_w = self.W * self.sd_stride

        with torch.no_grad():
            for batch in tqdm(self.dataloader, total=len(self.dataloader),
                              desc=f"DIFT Eval on {self.dataset_name}"):

                category = batch["category"]

                src_ft = self.compute_features(
                    batch["src_img"], batch["src_imname"], category, up_ft_index=[0, 1, 2], t=self.timestep
                )
                trg_ft = self.compute_features(
                    batch["trg_img"], batch["trg_imname"], category, up_ft_index=[0, 1, 2], t=self.timestep
                )

                src_desc, trg_desc, _ = _compute_pca(
                    src_ft,
                    trg_ft,
                    featmap_size=self.featmap_size,
                    pca_dims=self.PCA_DIMS
                )
                src_desc = F.normalize(src_desc, p=2, dim=-1, eps=1e-6)
                trg_desc = F.normalize(trg_desc, p=2, dim=-1, eps=1e-6)

                # keypoints resized 768×768
                src_kps = batch["src_kps"].to(self.device)  # (N,2) in 768
                trg_kps = batch["trg_kps"].to(self.device)  # (N,2) in 768

                trg_all = trg_desc[0, 0, :, :]
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
                        stride=self.sd_stride,
                        grid_hw=self.featmap_size,
                        img_hw=(out_h, out_w)
                    )

                    patch_index_src = int(y_idx) * self.W + int(x_idx)
                    src_vec = src_desc[0, 0, patch_index_src, :]

                    # ---- similarity map in token space (48x48) ----
                    sim2d = F.cosine_similarity(
                        trg_all,
                        src_vec.unsqueeze(0),
                        dim=-1
                    ).view(self.H, self.W)

                    # ---- pred token coords (y,x) ----

                    y_tok, x_tok = soft_argmax_window(
                        sim2d,
                        window_radius=self.wsam_win_radius,
                        temperature=self.wsam_temp
                    )

                    # ---- token -> pixel in 768 (patch center) ----
                    x_pred, y_pred = patch_idx_to_pixel((x_tok, y_tok), stride=self.sd_stride)

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
