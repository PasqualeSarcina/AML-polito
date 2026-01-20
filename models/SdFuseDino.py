import math
from copy import deepcopy
from itertools import islice
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from data.ap10k import AP10KDataset
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

        self.sd_preproc = DiftPreProcess(
            out_dim=(self.featmap_size[0] * self.sd_stride, self.featmap_size[1] * self.sd_stride))
        self.dino_preproc = Dinov2PreProcess(
            out_dim=(self.featmap_size[0] * self.dino_stride, self.featmap_size[1] * self.dino_stride))

        self._init_dataset()

    def _init_dataset(self):
        match self.dataset_name:
            case 'spair-71k':
                self.dataset = SPairDataset(datatype='test', dataset_size='small')
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
        sd_src_featmap = self.sd.compute_features(sd_batch['src_img'], sd_batch['src_imname'], sd_batch['category'], up_ft_index=[0, 1, 2])
        sd_trg_featmap = self.sd.compute_features(sd_batch['trg_img'], sd_batch['trg_imname'], sd_batch['category'], up_ft_index=[0, 1, 2])
        #if sd_src_featmap.ndim == 3:  # [C,48,48] -> [1,C,48,48]
        #    sd_src_featmap = sd_src_featmap.unsqueeze(0)
        #if sd_trg_featmap.ndim == 3:
        #    sd_trg_featmap = sd_trg_featmap.unsqueeze(0)

        dino_batch = deepcopy(batch)
        dino_batch = self.dino_preproc(dino_batch)
        dino_src_featmap = self.dino.compute_features(dino_batch['src_img'], dino_batch['src_imname'],
                                                      dino_batch['category'])
        dino_trg_featmap = self.dino.compute_features(dino_batch['trg_img'], dino_batch['trg_imname'],
                                                      dino_batch['category'])

        # se includono CLS token (1 + 1369), rimuovilo
        if dino_src_featmap.ndim == 3 and dino_src_featmap.shape[1] == 1 + self.featmap_size[0] * self.featmap_size[1]:
            dino_src_featmap = dino_src_featmap[:, 1:, :]
        if dino_trg_featmap.ndim == 3 and dino_trg_featmap.shape[1] == 1 + self.featmap_size[0] * self.featmap_size[1]:
            dino_trg_featmap = dino_trg_featmap[:, 1:, :]

        dino_src_featmap = dino_src_featmap.permute(0, 2, 1).reshape(1, dino_src_featmap.shape[2], self.featmap_size[0],
                                                                     self.featmap_size[1])
        dino_trg_featmap = dino_trg_featmap.permute(0, 2, 1).reshape(1, dino_trg_featmap.shape[2], self.featmap_size[0],
                                                                     self.featmap_size[1])

        return sd_src_featmap, sd_trg_featmap, dino_src_featmap, dino_trg_featmap

    def evaluate(self) -> List[CorrespondenceResult]:
        results = []
        up_ft_index = [0, 1, 2]
        torch.cuda.empty_cache()
        a = islice(self.dataloader, 100)
        with torch.no_grad():
            for batch in tqdm(a, total=100,
                              desc=f"DIFT + DINOv2 Eval on {self.dataset_name}"):
                sd_src_featmap, sd_trg_featmap, dino_src_featmap, dino_trg_featmap = self._compute_features(batch)

                # Adesso sd_src_featmap, sd_trg_featmap saranno dizionari, che hanno come chiavi il numero di layer

                # -------------------------------------------------------------------
                # PARAMETRI “repo-like”
                # -------------------------------------------------------------------
                PCA_DIMS = [256, 256, 256]  # come nel repo (3 blocchi SD)
                WEIGHT = [1, 1, 1, 1, 1]  # [w_s5,w_s4,w_s3,w_sd,w_dino] (repo)
                DIST = "l2"  # nel repo spesso forzato a l2
                H, W = self.featmap_size  # es. (48,48)
                P = H * W  # 2304

                def to_1chw(x: torch.Tensor) -> torch.Tensor:
                    # accetta [C,H,W] oppure [1,C,H,W] -> [1,C,H,W]
                    if x.ndim == 3:
                        return x.unsqueeze(0)
                    return x

                # -------------------------------------------------------------------
                # 1) SD: costruisci dict come repo: s5,s4,s3 (3 layer)
                #    Qui mappo i tuoi layer 0,1,2 -> s5,s4,s3
                # -------------------------------------------------------------------
                sd_src_dict = {
                    "s5": to_1chw(sd_src_featmap[0]),
                    "s4": to_1chw(sd_src_featmap[1]),
                    "s3": to_1chw(sd_src_featmap[2]),
                }
                sd_trg_dict = {
                    "s5": to_1chw(sd_trg_featmap[0]),
                    "s4": to_1chw(sd_trg_featmap[1]),
                    "s3": to_1chw(sd_trg_featmap[2]),
                }

                # -------------------------------------------------------------------
                # 2) co-PCA SD (repo): co_pca(features1, features2, PCA_DIMS)
                #    -> ritorna due tensori [1, Csum, H?, W?]
                # -------------------------------------------------------------------
                sd_src_proc, sd_trg_proc = co_pca(sd_src_dict, sd_trg_dict, PCA_DIMS)

                # repo: rescale features to (num_patches, num_patches)
                if sd_src_proc.shape[-2:] != (H, W):
                    sd_src_proc = F.interpolate(sd_src_proc, size=(H, W), mode="bilinear", align_corners=False)
                    sd_trg_proc = F.interpolate(sd_trg_proc, size=(H, W), mode="bilinear", align_corners=False)

                # repo: reshape -> [1,1,P,Dsd]
                sd_src_desc = sd_src_proc.reshape(1, 1, -1, P).permute(0, 1, 3, 2).contiguous()  # [1,1,P,Csd]
                sd_trg_desc = sd_trg_proc.reshape(1, 1, -1, P).permute(0, 1, 3, 2).contiguous()

                # repo: reweight intra-SD blocks dopo PCA (solo se non RAW)
                a, b, c = PCA_DIMS
                sd_src_desc[..., :a] *= WEIGHT[0]
                sd_src_desc[..., a:a + b] *= WEIGHT[1]
                sd_src_desc[..., a + b:a + b + c] *= WEIGHT[2]

                sd_trg_desc[..., :a] *= WEIGHT[0]
                sd_trg_desc[..., a:a + b] *= WEIGHT[1]
                sd_trg_desc[..., a + b:a + b + c] *= WEIGHT[2]

                # -------------------------------------------------------------------
                # 3) DINO: porta a descriptor [1,1,P,Ddino] (repo)
                #    Tu hai già dino_* come [1,C,H,W] (48x48) -> ok
                # -------------------------------------------------------------------
                if dino_src_featmap.shape[-2:] != (H, W):
                    dino_src_featmap = F.interpolate(dino_src_featmap, size=(H, W), mode="bilinear",
                                                     align_corners=False)
                    dino_trg_featmap = F.interpolate(dino_trg_featmap, size=(H, W), mode="bilinear",
                                                     align_corners=False)

                dino_src_desc = dino_src_featmap.permute(0, 2, 3, 1).reshape(1, 1, P, -1).contiguous()  # [1,1,P,768]
                dino_trg_desc = dino_trg_featmap.permute(0, 2, 3, 1).reshape(1, 1, P, -1).contiguous()

                # -------------------------------------------------------------------
                # 4) repo: normalize se dist contiene l1/l2 o plus_norm
                # -------------------------------------------------------------------
                if ("l1" in DIST) or ("l2" in DIST) or (DIST == "plus_norm"):
                    sd_src_desc = sd_src_desc / sd_src_desc.norm(dim=-1, keepdim=True)
                    sd_trg_desc = sd_trg_desc / sd_trg_desc.norm(dim=-1, keepdim=True)
                    dino_src_desc = dino_src_desc / dino_src_desc.norm(dim=-1, keepdim=True)
                    dino_trg_desc = dino_trg_desc / dino_trg_desc.norm(dim=-1, keepdim=True)

                # -------------------------------------------------------------------
                # 5) repo: fuse (concat) e reweight SD vs DINO
                # -------------------------------------------------------------------
                fuse_src_desc = torch.cat((sd_src_desc, dino_src_desc), dim=-1)  # [1,1,P,Dtot]
                fuse_trg_desc = torch.cat((sd_trg_desc, dino_trg_desc), dim=-1)

                sd_dim = sum(PCA_DIMS)  # a+b+c (repo)
                fuse_src_desc[..., :sd_dim] *= WEIGHT[3]  # SD weight
                fuse_src_desc[..., sd_dim:] *= WEIGHT[4]  # DINO weight
                fuse_trg_desc[..., :sd_dim] *= WEIGHT[3]
                fuse_trg_desc[..., sd_dim:] *= WEIGHT[4]

                # -------------------------------------------------------------------
                # 6) ORA: per ogni keypoint SRC calcoli la SIMILARITY MAP 2D (stop qui)
                # -------------------------------------------------------------------
                src_kps = batch["src_kps"].to(self.device)  # (N,2) nello spazio SD preprocess (es. 768)
                trg_kps = batch["trg_kps"].to(self.device)  # (N,2) nello spazio SD preprocess (es. 768)
                # oppure se li hai già scalati nel dataset a 768, ok.
                # altrimenti assicurati che kp_src sia nella stessa scala dell’immagine SD in input.
                distances_this_image = []

                for i in range(src_kps.shape[0]):
                    kp_src = src_kps[i]
                    kp_trg = trg_kps[i]

                    if torch.isnan(kp_src).any():
                        continue

                    # pixel (es 768) -> patch idx (48x48)
                    y_idx, x_idx = pixel_to_patch_idx(
                        kp_src,
                        stride=self.sd_stride,  # es 16
                        grid_hw=(H, W),
                        img_hw=(H * self.sd_stride, W * self.sd_stride)  # es (768,768)
                    )
                    patch_index_src = int(y_idx) * W + int(x_idx)

                    # vector source: [D]
                    src_vec = fuse_src_desc[0, 0, patch_index_src, :]  # [D]
                    trg_all = fuse_trg_desc[0, 0, :, :]  # [P,D]

                    # Nel repo con DIST="l2" fanno pairwise_sim + argmax.
                    # Equivalentemente (ranking identico) puoi usare:
                    #   - dot product se i token sono L2-normalizzati
                    # oppure
                    #   - negative squared L2 (più “l2-like”)
                    if ("l1" in DIST) or ("l2" in DIST) or (DIST == "plus_norm"):
                        # token già normalizzati -> dot ~ cosine, ranking ~ l2
                        sim_1d = (trg_all * src_vec.unsqueeze(0)).sum(dim=-1)  # [P]
                    else:
                        # negative squared L2 (se vuoi proprio l2 “puro”)
                        diff = trg_all - src_vec.unsqueeze(0)
                        sim_1d = -(diff * diff).sum(dim=-1)  # [P]

                    sim2d = sim_1d.view(H, W)  # <-- QUESTA è la similarity map 2D (48x48)

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
                        pck_threshold_0_05=batch["pck_threshold_0_05"],
                        pck_threshold_0_1=batch["pck_threshold_0_1"],
                        pck_threshold_0_2=batch["pck_threshold_0_2"]
                    )
                )

            return results


def copca_pair_featmaps(
    src: torch.Tensor,   # [1, C, H, W]
    trg: torch.Tensor,   # [1, C, H, W]
    out_dim: int = 256,
    eps: float = 1e-6,
    center: bool = True,
    l2_after: bool = True,
):
    """
    Co-PCA (paired PCA) su una coppia di feature map.
    - Fit PCA su [tokens_src ; tokens_trg] (stessa base per entrambe)
    - Proietta src e trg nella stessa sottobase
    - Ritorna feature map ridotte: [1, out_dim, H, W] ciascuna
    """

    assert src.ndim == 4 and trg.ndim == 4, "src/trg devono essere [1,C,H,W]"
    assert src.shape[0] == 1 and trg.shape[0] == 1, "batch=1 atteso"
    assert src.shape[-2:] == trg.shape[-2:], "H,W devono coincidere"
    assert src.shape[1] == trg.shape[1], "C deve coincidere per co-PCA"

    _, C, H, W = src.shape
    P = H * W

    # [1,C,H,W] -> [P,C]
    src_tok = src.permute(0, 2, 3, 1).reshape(P, C)
    trg_tok = trg.permute(0, 2, 3, 1).reshape(P, C)

    # concat per fit condiviso: [(2P), C]
    X = torch.cat([src_tok, trg_tok], dim=0)

    if center:
        mean = X.mean(dim=0, keepdim=True)
        Xc = X - mean
    else:
        mean = None
        Xc = X

    # PCA low-rank: Xc ≈ U S V^T , V: [C, q]
    q = min(out_dim, C)
    # nota: pca_lowrank lavora meglio con float32
    Xc32 = Xc.float()
    U, S, V = torch.pca_lowrank(Xc32, q=q)  # V: [C,q]

    # proiezione
    Z = Xc32 @ V[:, :q]  # [(2P), q]

    # split
    Zs = Z[:P, :]    # [P,q]
    Zt = Z[P:, :]    # [P,q]

    # reshape back: [1,q,H,W]
    src_red = Zs.reshape(1, H, W, q).permute(0, 3, 1, 2).contiguous()
    trg_red = Zt.reshape(1, H, W, q).permute(0, 3, 1, 2).contiguous()

    if l2_after:
        src_red = F.normalize(src_red, p=2, dim=1, eps=eps)
        trg_red = F.normalize(trg_red, p=2, dim=1, eps=eps)

    return src_red.to(src.dtype), trg_red.to(trg.dtype)

def _to_1chw(x: torch.Tensor) -> torch.Tensor:
    # accetta [C,H,W] o [1,C,H,W] -> [1,C,H,W]
    if x.ndim == 3:
        return x.unsqueeze(0)
    return x

def _copca_one_scale(src_1chw: torch.Tensor,
                     trg_1chw: torch.Tensor,
                     out_dim: int,
                     out_hw: tuple[int, int],
                     center: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    src_1chw, trg_1chw: [1,C,H,W]
    out_dim: q
    out_hw: (Hout,Wout)
    return: src_red, trg_red as [1,q,Hout,Wout]
    """
    src_1chw = _to_1chw(src_1chw)
    trg_1chw = _to_1chw(trg_1chw)

    # porta a griglia comune (repo: "rescale the features")
    if src_1chw.shape[-2:] != out_hw:
        src_1chw = F.interpolate(src_1chw, size=out_hw, mode="bilinear", align_corners=False)
    if trg_1chw.shape[-2:] != out_hw:
        trg_1chw = F.interpolate(trg_1chw, size=out_hw, mode="bilinear", align_corners=False)

    _, C, H, W = src_1chw.shape
    P = H * W
    q = min(out_dim, C)

    # [1,C,H,W] -> [P,C]
    src_tok = src_1chw.permute(0, 2, 3, 1).reshape(P, C)
    trg_tok = trg_1chw.permute(0, 2, 3, 1).reshape(P, C)

    # concat per co-PCA: [2P, C]
    X = torch.cat([src_tok, trg_tok], dim=0)

    # center (repo fa mean centering)
    if center:
        mean = X.mean(dim=0, keepdim=True)
        Xc = X - mean
    else:
        Xc = X

    # PCA lowrank su float32
    Xc32 = Xc.float()
    # V: [C,q]
    _, _, V = torch.pca_lowrank(Xc32, q=q)

    Z = Xc32 @ V[:, :q]  # [2P, q]
    Zs = Z[:P, :]
    Zt = Z[P:, :]

    # reshape back: [1,q,H,W]
    src_red = Zs.reshape(1, H, W, q).permute(0, 3, 1, 2).contiguous()
    trg_red = Zt.reshape(1, H, W, q).permute(0, 3, 1, 2).contiguous()

    # torna al dtype originale (fp16/fp32)
    return src_red.to(src_1chw.dtype), trg_red.to(trg_1chw.dtype)

def co_pca(features1: dict,
           features2: dict,
           pca_dims: list[int],
           out_hw: tuple[int, int],
           keys: tuple[str, ...] = ("s5", "s4", "s3"),
           center: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replica lo schema del repo:
    - per ogni key in keys: co-PCA paired + proiezione a pca_dims[i]
    - concat lungo canale
    return: processed_features1, processed_features2 as [1, sum(pca_dims), Hout, Wout]
    """
    assert len(pca_dims) == len(keys), "pca_dims e keys devono avere stessa lunghezza"

    src_list = []
    trg_list = []
    for k, q in zip(keys, pca_dims):
        if k not in features1 or k not in features2:
            raise KeyError(f"Chiave {k} mancante in features1/features2. Presenti: {list(features1.keys())}")

        fs = features1[k]
        ft = features2[k]
        fs_red, ft_red = _copca_one_scale(fs, ft, out_dim=q, out_hw=out_hw, center=center)
        src_list.append(fs_red)  # [1,q,H,W]
        trg_list.append(ft_red)

    processed_features1 = torch.cat(src_list, dim=1)  # [1, sum(q), H, W]
    processed_features2 = torch.cat(trg_list, dim=1)
    return processed_features1, processed_features2
