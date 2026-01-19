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
from data.transform import ImageNetNorm
from utils.utils_correspondence import argmax
from utils.utils_featuremaps import save_featuremap, load_featuremap
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
        self._init_dataset()

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

    def _init_dataset(self):
        match self.dataset_name:
            case 'spair-71k':
                self.dataset = SPairDataset(datatype='test', transform=ImageNetNorm(['src_img', 'trg_img']),
                                            dataset_size='large')
            case 'pf-pascal':
                self.dataset = PFPascalDataset(datatype='test', transform=ImageNetNorm(['src_img', 'trg_img']))
            case 'pf-willow':
                self.dataset = PFWillowDataset(datatype='test', transform=ImageNetNorm(['src_img', 'trg_img']))
            case 'ap-10k':
                self.dataset = AP10KDataset(datatype='test', transform=ImageNetNorm(['src_img', 'trg_img']))

        def collate_single(batch_list):
            return batch_list[0]

        self.dataloader = DataLoader(self.dataset, num_workers=4, batch_size=1, collate_fn=collate_single)

    @staticmethod
    def _preprocess_tensor(img: torch.Tensor, out_dim: tuple[int, int] = (518, 518)) -> torch.Tensor:
        """
        Preprocessing completo:
        - input: img CHW, float32, range tipico 0..255
        - resize stretch a out_dim con bilinear align_corners=False
        - /255
        - Normalize ImageNet
        - output: (1, C, out_h, out_w)
        """
        if img.ndim != 3:
            raise ValueError(f"Expected img shape (C,H,W), got {tuple(img.shape)}")
        if img.shape[0] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got C={img.shape[0]}")

        x = img.unsqueeze(0)  # (1,C,H,W)
        x = torch.nn.functional.interpolate(x, size=out_dim, mode="bilinear", align_corners=False)

        # Assicura float32
        if x.dtype != torch.float32:
            x = x.float()

        # Porta in [0,1] se sembra 0..255
        # (se già in 0..1 non cambia praticamente nulla)
        if x.max() > 1.5:
            x = x / 255.0

        # Normalize ImageNet
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std

        return x

    def _compute_features(self, img_tensor: torch.Tensor, img_name: str,
                          category: str) -> torch.Tensor:
        if self.dataset_name == "ap-10k":
            category = "all"
        if img_name in self.processed_img[category]:
            dict_out = load_featuremap(img_name, self.feat_dir, device=self.device)
            return dict_out

        dict_out = self.model.forward_features(self._preprocess_tensor(img_tensor.to(self.device)))
        featmap = dict_out["x_norm_patchtokens"]
        self.processed_img[category].add(img_name)
        save_featuremap(featmap, img_name, self.feat_dir)
        return featmap

    @staticmethod
    def _resize_keypoints(
            kps: torch.Tensor,
            orig_img_size: torch.Tensor | tuple,
            out_img_size: tuple[int, int] = (518, 518),
    ) -> torch.Tensor:
        out_h, out_w = out_img_size

        # deduci H,W
        if isinstance(orig_img_size, torch.Tensor):
            size = orig_img_size
            if size.numel() == 2:  # (H,W)
                orig_h, orig_w = size[0], size[1]
            elif size.numel() == 3:  # (C,H,W) default
                orig_h, orig_w = size[1], size[2]
            else:
                raise ValueError(f"orig_img_size non supportato: {tuple(size.shape)}")
        else:
            size = tuple(orig_img_size)
            if len(size) == 2:
                orig_h, orig_w = size
            elif len(size) == 3:
                _, orig_h, orig_w = size
            else:
                raise ValueError(f"orig_img_size non supportato: {size}")

        k = kps.clone().to(dtype=torch.float32)

        orig_h = torch.as_tensor(orig_h, device=k.device, dtype=torch.float32)
        orig_w = torch.as_tensor(orig_w, device=k.device, dtype=torch.float32)

        sx = float(out_w) / orig_w
        sy = float(out_h) / orig_h

        # replica align_corners=False (half-pixel)
        k[:, 0] = (k[:, 0] + 0.5) * sx - 0.5
        k[:, 1] = (k[:, 1] + 0.5) * sy - 0.5
        return k

    def evaluate(self) -> list[CorrespondenceResult]:
        results = []

        out_h, out_w = 518, 518
        patch_size = 14
        w_grid = out_w // patch_size  # 37
        h_grid = out_h // patch_size  # 37

        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in tqdm(self.dataloader, total=len(self.dataloader), desc=f"Computing correspondeces with DINOv2 on {self.dataset_name}",
                              smoothing=0.1, mininterval=0.7, maxinterval=2.0):
                category = batch["category"]
                # featuremaps salvate su immagini resized 518x518
                feats_src = self._compute_features(batch["src_img"], batch["src_imname"], category)
                feats_trg = self._compute_features(batch["trg_img"], batch["trg_imname"], category)

                feats_src = feats_src.to(self.device)
                feats_trg = feats_trg.to(self.device)

                # se includono CLS token (1 + 1369), rimuovilo
                if feats_src.ndim == 3 and feats_src.shape[1] == 1 + h_grid * w_grid:
                    feats_src = feats_src[:, 1:, :]
                if feats_trg.ndim == 3 and feats_trg.shape[1] == 1 + h_grid * w_grid:
                    feats_trg = feats_trg[:, 1:, :]

                # keypoints ORIGINALI dal dataset
                src_kps_orig = batch["src_kps"].to(self.device)  # (N,2) in pixel originali
                trg_kps_orig = batch["trg_kps"].to(self.device)  # (N,2) in pixel originali

                # SRC originale -> 518 (coerente con align_corners=False)
                src_kps_518 = self._resize_keypoints(src_kps_orig, batch["src_imsize"], (out_h, out_w))

                # size originale target (per inversione 518->originale)
                tsize = batch["trg_imsize"]
                Ht = float(tsize[1])
                Wt = float(tsize[2])

                sx_t = out_w / Wt
                sy_t = out_h / Ht

                distances_this_image: list[float] = []

                # loop keypoints
                N = min(src_kps_518.shape[0], trg_kps_orig.shape[0])
                for i in range(N):
                    kp_src_518 = src_kps_518[i]
                    kp_trg_o = trg_kps_orig[i]

                    # skip keypoints "invalidi" tipici (es. -1, -1) o NaN
                    if torch.isnan(kp_src_518).any() or torch.isnan(kp_trg_o).any():
                        continue
                    if kp_src_518[0] < 0 or kp_src_518[1] < 0:
                        continue
                    if kp_trg_o[0] < 0 or kp_trg_o[1] < 0:
                        continue

                    # --- SRC kp (in 518) -> patch index ---
                    x_518 = float(kp_src_518[0].item())
                    y_518 = float(kp_src_518[1].item())

                    x_518 = max(0.0, min(out_w - 1.0, x_518))
                    y_518 = max(0.0, min(out_h - 1.0, y_518))

                    x_patch = int(x_518 // patch_size)
                    y_patch = int(y_518 // patch_size)
                    x_patch = min(max(0, x_patch), w_grid - 1)
                    y_patch = min(max(0, y_patch), h_grid - 1)

                    patch_index_src = y_patch * w_grid + x_patch

                    # --- source vector ---
                    source_vec = feats_src[0, patch_index_src, :]  # (D,)

                    # --- similarity su tutti i patch target ---
                    sim_1d = torch.nn.functional.cosine_similarity(
                        feats_trg[0], source_vec.unsqueeze(0), dim=-1
                    )  # (1369,)
                    sim_2d = sim_1d.view(h_grid, w_grid)  # (37,37)

                    # --- SAM-like: upsample similarity map -> 518x518 e argmax lì ---
                    sim_r = torch.nn.functional.interpolate(
                        sim_2d[None, None],  # (1,1,37,37)
                        size=(out_h, out_w),
                        mode="bilinear",
                        align_corners=False
                    )[0, 0]  # (518,518)

                    if self.win_soft_argmax:
                        x_r, y_r = argmax(sim_r, window_size=self.wsam_win_size, beta=self.wsam_beta)  # ritorna x,y
                    else:
                        x_r, y_r = argmax(sim_r, window_size=1)

                    x_r = float(x_r.item()) if isinstance(x_r, torch.Tensor) else float(x_r)
                    y_r = float(y_r.item()) if isinstance(y_r, torch.Tensor) else float(y_r)

                    # --- 518 -> ORIG target (inverse half-pixel, align_corners=False) ---
                    x_pred_orig = (x_r + 0.5) / sx_t - 0.5
                    y_pred_orig = (y_r + 0.5) / sy_t - 0.5

                    gt_x = float(kp_trg_o[0].item())
                    gt_y = float(kp_trg_o[1].item())

                    dx = x_pred_orig - gt_x
                    dy = y_pred_orig - gt_y
                    dist = math.sqrt(dx * dx + dy * dy)
                    distances_this_image.append(dist)

                results.append(
                    CorrespondenceResult(
                        category=category,
                        distances=distances_this_image,
                        # soglie già in pixel ORIGINALI (coerenti con dist)
                        pck_threshold_0_05=batch["pck_threshold_0_05"],
                        pck_threshold_0_1=batch["pck_threshold_0_1"],
                        pck_threshold_0_2=batch["pck_threshold_0_2"],
                    )
                )

        return results

    @staticmethod
    def soft_argmax_window(sim_map_2d, window_radius=3, temperature=20):
        """
        Args:
            sim_map_2d: Tensor shape (H, W) containing similarity scores.
            window_radius: How many neighbors to look at (e.g., 3).
            temperature: Sharpening factor. Higher = closer to hard argmax.
        Returns:
            y_soft, x_soft: Float coordinates on the grid.
        """
        H, W = sim_map_2d.shape

        # 1. Find the Hard Peak (Integer)
        flattened = sim_map_2d.view(-1)
        idx = torch.argmax(flattened)
        y_hard = idx // W
        x_hard = idx % W

        # 2. Define the Window around the peak
        y_min = max(0, y_hard - window_radius)
        y_max = min(H, y_hard + window_radius + 1)
        x_min = max(0, x_hard - window_radius)
        x_max = min(W, x_hard + window_radius + 1)

        # 3. Crop the window
        window = sim_map_2d[y_min:y_max, x_min:x_max]

        # 4. Convert Scores to Probabilities (Softmax)
        # We subtract max for numerical stability, then multiply by temperature
        # Cosine similarity is usually -1 to 1. We scale it up so Softmax isn't too flat.
        # 1. Flatten the window to 1D so Softmax considers ALL pixels together
        flat_input = ((window - window.max()) * temperature).view(-1)

        # 2. Apply Softmax on the flat array (dim=0)
        flat_weights = torch.nn.functional.softmax(flat_input, dim=0)

        # 3. Reshape back to the original 2D square shape
        weights = flat_weights.view(window.shape)
        # 5. Calculate Center of Mass (Weighted Sum)
        # Create a grid of coordinates for the window
        device = sim_map_2d.device
        local_y, local_x = torch.meshgrid(
            torch.arange(y_min, y_max, device=device).float(),
            torch.arange(x_min, x_max, device=device).float(),
            indexing='ij'
        )

        y_soft = torch.sum(weights * local_y)
        x_soft = torch.sum(weights * local_x)

        return y_soft, x_soft
