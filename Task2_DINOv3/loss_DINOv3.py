import torch
import torch.nn.functional as F

from utils.matching_DINOv3 import grid_valid_mask_from_meta 

class GaussianCrossEntropyLoss:
    def __init__(self, *, out_size, patch_size, temperature=0.5, sigma=2.0, eps=1e-8,
                 enable_l2_norm=True, window=7, use_windowed=True):
        self.out_size = int(out_size)
        self.patch_size = int(patch_size)
        self.temperature = float(temperature)
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.enable_l2_norm = bool(enable_l2_norm)

        self.window = int(window)
        assert self.window % 2 == 1, "window must be odd (e.g., 7)"
        self.use_windowed = bool(use_windowed)

        # kernel in patch-grid coords (not pixel coords)
        self._kernel_cache = None  # built lazily on device

    def __call__(
        self,
        src_feats: torch.Tensor,     # [B,C,Hf,Wf]
        trg_feats: torch.Tensor,     # [B,C,Hf,Wf]
        src_kps_list,                # list len B of [K,2] pixel coords
        trg_kps_list,                # list len B of [K,2] pixel coords
        src_meta_list,               # list len B of dict
        trg_meta_list,               # list len B of dict
        *,
        temperature: float | None = None,
        return_kps: bool = False,
        sigma: float | None = None,
    ) -> torch.Tensor:
        assert src_feats.ndim == 4 and trg_feats.ndim == 4, "Expected [B,C,Hf,Wf]"
        B, C, Hf, Wf = trg_feats.shape
        device = trg_feats.device

        out_size = self.out_size
        patch = self.patch_size
        temp = max(float(self.temperature if temperature is None else temperature), self.eps)
        sigma_eff = float(self.sigma if sigma is None else sigma)
        sig2 = 2.0 * (sigma_eff ** 2)

        # (optional) L2 norm on channel dim for cosine sim
        if self.enable_l2_norm:
            srcn = F.normalize(src_feats, dim=1)
            trgn = F.normalize(trg_feats, dim=1)
        else:
            srcn = src_feats
            trgn = trg_feats

        N = Hf * Wf
        trgn_flat = trgn.view(B, C, N).transpose(1, 2).contiguous()  # [B,N,C]

        # patch-grid coordinates for gaussian construction
        yy, xx = torch.meshgrid(
            torch.arange(Hf, device=device, dtype=torch.float32),
            torch.arange(Wf, device=device, dtype=torch.float32),
            indexing="ij",
        )

        total_loss = torch.zeros((), device=device)
        total_kps = 0

        for b in range(B):
            src_kps, trg_kps = self._to_kps_tensors(src_kps_list[b], trg_kps_list[b], device)
            if src_kps.numel() == 0:
                continue

            # filter visible/in-bounds in pixel coords (your current behavior)
            valid = self._valid_pixel_kps(src_kps, trg_kps, out_size)
            if valid.sum().item() == 0:
                continue

            src_kps_v = src_kps[valid]
            trg_kps_v = trg_kps[valid]

            # padding-aware valid patch masks
            src_grid_valid = grid_valid_mask_from_meta(src_meta_list[b], out_size=out_size, patch=patch, device=device)
            trg_grid_valid = grid_valid_mask_from_meta(trg_meta_list[b], out_size=out_size, patch=patch, device=device)
            trg_valid_flat = trg_grid_valid.view(-1)  # [N] bool

            # source patch index = nearest patch CENTER (your convention)
            sx = torch.clamp(((src_kps_v[:, 0] / patch) - 0.5).round().long(), 0, Wf - 1)
            sy = torch.clamp(((src_kps_v[:, 1] / patch) - 0.5).round().long(), 0, Hf - 1)

            # drop src kps falling on padded src patches
            src_on_valid = src_grid_valid[sy, sx]
            if src_on_valid.sum().item() == 0:
                continue

            sx = sx[src_on_valid]
            sy = sy[src_on_valid]
            trg_kps_v = trg_kps_v[src_on_valid]
            Kv = int(trg_kps_v.shape[0])
            # -------------------------------------------------
            # DROP TARGET KPS FALLING ON PADDED TARGET PATCHES
            # -------------------------------------------------
            tx = torch.clamp(((trg_kps_v[:, 0] / patch) - 0.5).round().long(), 0, Wf - 1)
            ty = torch.clamp(((trg_kps_v[:, 1] / patch) - 0.5).round().long(), 0, Hf - 1)

            trg_on_valid = trg_grid_valid[ty, tx]
            if trg_on_valid.sum().item() == 0:
                continue

            # apply target validity mask (keep geometry consistent)
            sx = sx[trg_on_valid]
            sy = sy[trg_on_valid]
            trg_kps_v = trg_kps_v[trg_on_valid]
            Kv = int(trg_kps_v.shape[0])

            # [Kv,C] source descriptors
            src_desc = srcn[b, :, sy, sx].transpose(0, 1).contiguous()

            # logits over target patches [Kv,N]
            logits = (trgn_flat[b] @ src_desc.T).transpose(0, 1).contiguous() / temp

            # stable log_probs with padding mask (your key property)
            log_probs = self._masked_log_softmax_fp32(logits, trg_valid_flat)

            # gaussian centers in patch coords (your convention)
            cx = (trg_kps_v[:, 0] / patch) - 0.5
            cy = (trg_kps_v[:, 1] / patch) - 0.5


            if self.use_windowed:
                loss_b = self._gaussian_ce_windowed(
                    log_probs=log_probs,
                    trg_grid_valid=trg_grid_valid,
                    cx=cx, cy=cy,
                    Hf=Hf, Wf=Wf,
                    sigma_eff=sigma_eff,
                )
            else:
                loss_b = self._gaussian_ce_over_grid(
                    log_probs=log_probs,
                    xx=xx, yy=yy,
                    cx=cx, cy=cy,
                    trg_valid_flat=trg_valid_flat,
                    sig2=2.0 * (sigma_eff ** 2),
                )

            total_loss = total_loss + loss_b
            total_kps += Kv

        if total_kps == 0:
            if return_kps:
                return torch.zeros((), device=device), 0
            return torch.zeros((), device=device)

        out = total_loss / total_kps
        if return_kps:
            return out, total_kps
        return out

    # ------------------------
    # Helpers (SD4Match-like)
    # ------------------------
    def _get_kernel(self, device, sigma_eff: float):
        key = (device, int(self.window), float(sigma_eff))
        if getattr(self, "_kernel_key", None) == key and self._kernel_cache is not None:
            return self._kernel_cache

        w = self.window
        r = w // 2
        yy, xx = torch.meshgrid(
            torch.arange(-r, r + 1, device=device, dtype=torch.float32),
            torch.arange(-r, r + 1, device=device, dtype=torch.float32),
            indexing="ij",
        )
        sig2 = 2.0 * (float(sigma_eff) ** 2)
        k = torch.exp(-(xx * xx + yy * yy) / (sig2 + self.eps))
        k = k / (k.sum() + self.eps)

        self._kernel_cache = k
        self._kernel_key = key
        return k

    def _gaussian_ce_windowed(
        self,
        *,
        log_probs: torch.Tensor,      # [Kv,N]
        trg_grid_valid: torch.Tensor, # [Hf,Wf] bool
        cx: torch.Tensor, cy: torch.Tensor,  # [Kv] in patch coords (float)
        Hf: int, Wf: int,
        sigma_eff: float
    ) -> torch.Tensor:
        eps = self.eps
        Kv = int(cx.shape[0])
        logp_map = log_probs.view(Kv, Hf, Wf)  # [Kv,Hf,Wf]
        kernel = self._get_kernel(log_probs.device, sigma_eff)
        w = self.window
        r = w // 2

        loss_sum = torch.zeros((), device=log_probs.device, dtype=torch.float32)

        # choose center patch index (SD4Match uses subpixel; here nearest patch)
        ix0 = torch.clamp(cx.round().long(), 0, Wf - 1)
        iy0 = torch.clamp(cy.round().long(), 0, Hf - 1)

        for k in range(Kv):
            ix = int(ix0[k].item())
            iy = int(iy0[k].item())

            x0 = ix - r; x1 = ix + r + 1
            y0 = iy - r; y1 = iy + r + 1

            # clip to grid bounds
            xc0 = max(0, x0); xc1 = min(Wf, x1)
            yc0 = max(0, y0); yc1 = min(Hf, y1)

            # slice window
            lp = logp_map[k, yc0:yc1, xc0:xc1]                # [hwin, wwin]
            m  = trg_grid_valid[yc0:yc1, xc0:xc1].float()     # [hwin, wwin]

            # kernel slice (because we clipped window)
            ky0 = yc0 - y0
            kx0 = xc0 - x0
            ky1 = ky0 + (yc1 - yc0)
            kx1 = kx0 + (xc1 - xc0)
            kw = kernel[ky0:ky1, kx0:kx1]                     # [hwin, wwin]

            # padding-aware: zero invalid + renorm
            wgt = kw * m
            denom = wgt.sum()
            if denom.item() <= 0:
                continue
            wgt = wgt / (denom + eps)

            loss_sum = loss_sum - torch.sum(wgt * lp)

        return loss_sum

    def _to_kps_tensors(self, src_kps, trg_kps, device):
        if not isinstance(src_kps, torch.Tensor):
            src_kps = torch.tensor(src_kps, device=device, dtype=torch.float32)
        else:
            src_kps = src_kps.to(device=device, dtype=torch.float32)

        if not isinstance(trg_kps, torch.Tensor):
            trg_kps = torch.tensor(trg_kps, device=device, dtype=torch.float32)
        else:
            trg_kps = trg_kps.to(device=device, dtype=torch.float32)

        return src_kps, trg_kps

    def _valid_pixel_kps(self, src_kps: torch.Tensor, trg_kps: torch.Tensor, out_size: int) -> torch.Tensor:
        return (
            (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) &
            (src_kps[:, 0] < out_size) & (src_kps[:, 1] < out_size) &
            (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0) &
            (trg_kps[:, 0] < out_size) & (trg_kps[:, 1] < out_size)
        )

    def _masked_log_softmax_fp32(self, logits: torch.Tensor, trg_valid_flat: torch.Tensor) -> torch.Tensor:
        # logits: [Kv,N] (float16/32 ok)
        mask = ~trg_valid_flat.unsqueeze(0)  # [1,N] bool
        logits_fp32 = logits.float().masked_fill(mask, -1e9)
        return F.log_softmax(logits_fp32, dim=-1)  # [Kv,N] fp32

    def _gaussian_ce_over_grid(
        self,
        *,
        log_probs: torch.Tensor,         # [Kv,N]
        xx: torch.Tensor, yy: torch.Tensor,  # [Hf,Wf]
        cx: torch.Tensor, cy: torch.Tensor,  # [Kv]
        trg_valid_flat: torch.Tensor,    # [N] bool
        sig2: float,
    ) -> torch.Tensor:
        eps = self.eps
        Hf, Wf = xx.shape
        Kv = int(cx.shape[0])

        # compute per-kp gaussian CE (loop is fine; Kv small)
        loss_sum = torch.zeros((), device=log_probs.device, dtype=torch.float32)

        valid_f = trg_valid_flat.float()
        for k in range(Kv):
            dx = xx - cx[k]
            dy = yy - cy[k]
            g = torch.exp(-(dx * dx + dy * dy) / (sig2 + eps))  # [Hf,Wf]
            g_flat = g.reshape(-1) * valid_f                    # [N]
            denom = g_flat.sum()
            if denom.item() <= 0:
                continue
            g_flat = g_flat / (denom + eps)

            loss_sum = loss_sum - torch.sum(g_flat * log_probs[k])

        return loss_sum

