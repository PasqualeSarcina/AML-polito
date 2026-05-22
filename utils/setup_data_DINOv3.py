import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.dinov3.PreProcess import PreProcess
from models.dinov3.model_DINOv3 import load_dinov3_backbone
from utils.soft_argmax_window import soft_argmax_window
from utils.utils_convert import pixel_to_patch_idx, patch_idx_to_pixel
from utils.utils_featuremaps import load_featuremap, save_featuremap
from utils.utils_init_dataloader import init_dataloader
from utils.utils_results import CorrespondenceResult


class Dinov3Eval:
    """
    DINOv3 evaluation for semantic correspondence.

    Pipeline:
        image -> resize 512x512 -> DINOv3 ViT-B/16
        patch size = 16
        feature grid = 32x32
        keypoints and bounding boxes are resized by PreProcess
        PCK thresholds are computed on the resized target bounding box

    Soft-argmax behavior:
        wsam_win_radius = 0  -> hard argmax
        wsam_win_radius > 0  -> window soft-argmax

    Note:
        soft_argmax_window uses temperature, not beta.
        beta = 50 corresponds to temperature = 1 / 50 = 0.02.
    """

    def __init__(self, args):
        """
        Initialize DINOv3 evaluation settings, model, dataloader, and feature cache.
        """
        self.dataset_name = args.dataset
        self.custom_weights = args.custom_weights

        self.device = args.device
        self.base_dir = args.base_dir

        self.patch_size = 16
        self.input_size = 512

        # eval.py is not modified:
        # wsam_win_radius decides whether to use hard argmax or window soft-argmax.
        self.wsam_win_radius = int(getattr(args, "wsam_win_radius", 0))
        self.wsam_temp = float(getattr(args, "wsam_temp", 0.05))
        self.win_soft_argmax = self.wsam_win_radius > 0

        if self.wsam_win_radius < 0:
            raise ValueError(
                f"wsam_win_radius must be >= 0, got {self.wsam_win_radius}."
            )

        if self.wsam_temp <= 0:
            raise ValueError(
                f"wsam_temp must be > 0, got {self.wsam_temp}."
            )

        self._init_model()

        transform = PreProcess(out_dim=(self.input_size, self.input_size))

        self.dataset, self.dataloader = init_dataloader(
            self.dataset_name,
            base_dir=self.base_dir,
            datatype="test",
            transform=transform,
        )

        if self.custom_weights is None:
            run_name = "pretrained"
        else:
            run_name = Path(self.custom_weights).stem

        self.feat_dir = (
            Path(self.base_dir)
            / "data"
            / "features"
            / "dinov3"
            / run_name
        )

        self.processed_img = defaultdict(set)

        print(
            "[DINOv3 Eval] "
            f"window_soft_argmax={self.win_soft_argmax} | "
            f"wsam_win_radius={self.wsam_win_radius} | "
            f"wsam_temp={self.wsam_temp}"
        )

    def _init_model(self):
        """
        Load the DINOv3 ViT-B/16 backbone and optional custom fine-tuned weights.
        """
        pretrained_checkpoint = (
            Path(self.base_dir)
            / "checkpoints"
            / "dinov3"
            / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        )

        if not pretrained_checkpoint.exists():
            pretrained_checkpoint = (
                Path(self.base_dir)
                / "checkpoints"
                / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            )

        if not pretrained_checkpoint.exists():
            raise FileNotFoundError(
                f"DINOv3 pretrained checkpoint not found: {pretrained_checkpoint}"
            )

        dinov3_dir = Path(self.base_dir) / "third_party" / "dinov3"

        model = load_dinov3_backbone(
            dinov3_dir=dinov3_dir,
            weights_path=pretrained_checkpoint,
            device=self.device,
            sanity_input_size=self.input_size,
            verbose=True,
        )

        if self.custom_weights is not None:
            checkpoint = torch.load(self.custom_weights, map_location=self.device)

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            state_dict = {
                k.replace("module.", "").replace("model.", ""): v
                for k, v in state_dict.items()
            }

            msg = model.load_state_dict(state_dict, strict=False)

            print(f"Loaded custom fine-tuned weights from: {self.custom_weights}")
            print(f"Missing keys: {len(msg.missing_keys)}")
            print(msg.missing_keys)
            print(f"Unexpected keys: {len(msg.unexpected_keys)}")
            print(msg.unexpected_keys)

        self.model = model.to(self.device).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _safe_tokens(out: torch.Tensor) -> torch.Tensor:
        """
        Validate and return patch tokens in [B, N, C] format.
        """
        if isinstance(out, (tuple, list)):
            out = out[0]

        if not torch.is_tensor(out) or out.ndim != 3:
            raise RuntimeError(
                f"Unexpected tokens output: {type(out)} "
                f"shape={getattr(out, 'shape', None)}"
            )

        return out

    @staticmethod
    def _tokens_to_featuremap(
        tokens_bnc: torch.Tensor,
        h_grid: int,
        w_grid: int,
    ) -> torch.Tensor:
        """
        Convert DINOv3 patch tokens [1, N, C] into a normalized feature map [H, W, C].
        """
        if tokens_bnc.shape[0] != 1:
            raise RuntimeError(
                f"DINOv3 evaluation expects batch size 1, got {tokens_bnc.shape[0]}"
            )

        tok = tokens_bnc.squeeze(0)

        n_patches = h_grid * w_grid

        if tok.shape[0] < n_patches:
            raise RuntimeError(
                f"Ntok={tok.shape[0]} < Npatch={n_patches} "
                f"(h_grid={h_grid}, w_grid={w_grid})"
            )

        # x_norm_patchtokens should already contain patch tokens.
        # Keeping the last n_patches makes the conversion robust to possible extra tokens.
        patch_tok = tok[-n_patches:]

        featmap = patch_tok.view(h_grid, w_grid, -1)

        return F.normalize(featmap, dim=-1)

    def _cache_name(self, img_name: str, category: str) -> str:
        """
        Build a category-aware cache filename to avoid feature collisions.
        """
        safe_category = str(category).replace("/", "_")
        safe_img_name = str(img_name).replace("/", "_").replace("\\", "_")

        return f"{safe_category}__{safe_img_name}"

    def compute_features(
        self,
        img_tensor: torch.Tensor,
        img_name: str,
        category: str,
    ) -> torch.Tensor:
        """
        Load cached features or extract and cache normalized DINOv3 feature maps.
        """
        if self.dataset_name == "ap-10k":
            category = "all"

        cache_name = self._cache_name(img_name, category)
        cache_path = self.feat_dir / f"{cache_name}.pt"

        if cache_name in self.processed_img[category] or cache_path.exists():
            self.processed_img[category].add(cache_name)
            return load_featuremap(
                cache_name,
                self.feat_dir,
                device=self.device,
            )

        x = img_tensor.to(self.device)

        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise RuntimeError(
                f"Unexpected img_tensor shape {x.shape}, expected CHW or BCHW."
            )

        _, _, h_img, w_img = x.shape

        if h_img != self.input_size or w_img != self.input_size:
            raise RuntimeError(
                f"DINOv3 fixed-resize pipeline expects "
                f"{self.input_size}x{self.input_size}, but got {h_img}x{w_img}."
            )

        h_grid = h_img // self.patch_size
        w_grid = w_img // self.patch_size

        with torch.no_grad():
            dict_out = self.model.forward_features(x)

        if "x_norm_patchtokens" not in dict_out:
            raise RuntimeError(
                "DINOv3 forward_features output does not contain 'x_norm_patchtokens'."
            )

        tokens = self._safe_tokens(dict_out["x_norm_patchtokens"])
        featmap = self._tokens_to_featuremap(tokens, h_grid, w_grid)

        self.processed_img[category].add(cache_name)

        save_featuremap(
            featmap,
            cache_name,
            self.feat_dir,
        )

        return featmap

    def evaluate(self) -> list[CorrespondenceResult]:
        """
        Run source-to-target keypoint matching and return per-pair correspondence results.
        """
        results = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in tqdm(
                self.dataloader,
                total=len(self.dataloader),
                desc=f"Computing correspondences with DINOv3 on {self.dataset_name}",
                smoothing=0.1,
                mininterval=0.7,
                maxinterval=2.0,
            ):
                category = batch["category"]

                feats_src = self.compute_features(
                    batch["src_img"],
                    batch["src_imname"],
                    category,
                )

                feats_trg = self.compute_features(
                    batch["trg_img"],
                    batch["trg_imname"],
                    category,
                )

                src_kps = batch["src_kps"].to(self.device)
                trg_kps = batch["trg_kps"].to(self.device)

                _, h_src, w_src = batch["src_imsize"]
                _, h_trg, w_trg = batch["trg_imsize"]

                h_src = int(h_src)
                w_src = int(w_src)
                h_trg = int(h_trg)
                w_trg = int(w_trg)

                h_grid_src = h_src // self.patch_size
                w_grid_src = w_src // self.patch_size

                distances_this_image: list[float] = []

                n_kps = min(src_kps.shape[0], trg_kps.shape[0])

                for i in range(n_kps):
                    src_kp = src_kps[i]
                    trg_kp = trg_kps[i]

                    if torch.isnan(src_kp).any() or torch.isnan(trg_kp).any():
                        continue

                    x_idx, y_idx = pixel_to_patch_idx(
                        xy=src_kp,
                        stride=self.patch_size,
                        grid_hw=(h_grid_src, w_grid_src),
                        img_hw=(h_src, w_src),
                    )

                    src_feat = feats_src[y_idx, x_idx]

                    # Cosine similarity because feature maps are L2-normalized.
                    sim_2d = (feats_trg * src_feat).sum(dim=-1)

                    y_pred_patch, x_pred_patch = soft_argmax_window(
                        sim_2d,
                        window_radius=self.wsam_win_radius,
                        temperature=self.wsam_temp,
                    )

                    x_pred, y_pred = patch_idx_to_pixel(
                        (x_pred_patch, y_pred_patch),
                        stride=self.patch_size,
                    )

                    x_pred = max(0.0, min(float(x_pred), float(w_trg - 1)))
                    y_pred = max(0.0, min(float(y_pred), float(h_trg - 1)))

                    dx = x_pred - float(trg_kp[0])
                    dy = y_pred - float(trg_kp[1])

                    dist = math.sqrt(dx * dx + dy * dy)
                    distances_this_image.append(dist)

                results.append(
                    CorrespondenceResult(
                        category=category,
                        distances=distances_this_image,
                        pck_threshold_0_05=batch["pck_threshold_0_05"],
                        pck_threshold_0_1=batch["pck_threshold_0_1"],
                        pck_threshold_0_2=batch["pck_threshold_0_2"],
                    )
                )

        return results