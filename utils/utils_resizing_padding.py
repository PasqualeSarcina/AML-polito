import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# ----------------------------
# Utils: I/O + resizing/padding
# ----------------------------
def read_rgb_chw(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, copy=True)        # <-- FIX
    t = torch.from_numpy(arr).permute(2,0,1).contiguous().float()
    return t

def normalize_imagenet(x_chw_255: torch.Tensor) -> torch.Tensor:
    x = x_chw_255 / 255.0
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std  = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std

def resize_keep_ar(img_chw: torch.Tensor, long_side: int) -> tuple[torch.Tensor, float, float]:
    """Resize keeping aspect ratio so that max(H,W)=long_side. Returns resized and (sx, sy)."""
    _, H, W = img_chw.shape
    scale = long_side / max(H, W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))
    img_rs = F.interpolate(img_chw.unsqueeze(0), size=(newH, newW), mode="bilinear", align_corners=False).squeeze(0)
    sx = newW / W
    sy = newH / H
    return img_rs, sx, sy

def pad_to_square(img_chw: torch.Tensor, out_size: int, pad_mode: str = "center", pad_value: float = 0.0):
    """
    Pad resized CHW to out_size x out_size. Returns (padded_img, pad_x, pad_y),
    where pad_x/pad_y are the left/top padding offsets applied.
    pad_mode: "center" or "br" (bottom-right only).
    """
    _, H, W = img_chw.shape
    assert H <= out_size and W <= out_size, "Image must be <= out_size after resize."
    pad_h = out_size - H
    pad_w = out_size - W

    if pad_mode == "center":
        pad_left = pad_w // 2
        pad_top  = pad_h // 2
    elif pad_mode == "br":
        pad_left = 0
        pad_top  = 0
    else:
        raise ValueError(f"pad_mode={pad_mode} not supported")

    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    img_pad = F.pad(img_chw, (pad_left, pad_right, pad_top, pad_bottom), value=float(pad_value))
    return img_pad, pad_left, pad_top

def preprocess_square(img_chw_255: torch.Tensor, out_size: int, pad_mode: str = "center"):
    img_rs, sx, sy = resize_keep_ar(img_chw_255, long_side=out_size)
    img_rs_norm = normalize_imagenet(img_rs)

    # 3) pad in normalized space: value=0 == ImageNet mean after normalization
    img_norm, pad_x, pad_y = pad_to_square(img_rs_norm, out_size=out_size, pad_mode=pad_mode, pad_value=0.0)

    meta = {
        "sx": sx, "sy": sy,
        "pad_x": pad_x, "pad_y": pad_y,
        "H_rs": img_rs.shape[1], "W_rs": img_rs.shape[2],
        "out_size": out_size,
    }
    return img_norm, meta


def to_tensor_kps(kps_any) -> torch.Tensor:
    """
    Robust conversion of keypoints to float tensor [K,2].
    Handles list with -1 or None. Converts None -> -1.
    """
    # kps_any is expected to be list-like length K of [x,y]
    kps = []
    for p in kps_any:
        if p is None:
            kps.append([-1.0, -1.0])
        else:
            x, y = p
            if x is None or y is None:
                kps.append([-1.0, -1.0])
            else:
                kps.append([float(x), float(y)])
    return torch.tensor(kps, dtype=torch.float32)

def transform_coords_xy(kps_xy: torch.Tensor, sx: float, sy: float, pad_x: int, pad_y: int) -> torch.Tensor:
    """Apply resize (sx,sy) and padding (pad_x,pad_y) to keypoints [K,2]."""
    out = kps_xy.clone()
    valid = (out[:,0] >= 0) & (out[:,1] >= 0)
    out[valid,0] = out[valid,0] * sx + pad_x
    out[valid,1] = out[valid,1] * sy + pad_y
    return out

def transform_bbox_xyxy(bbox, sx: float, sy: float, pad_x: int, pad_y: int, out_size: int):
    """bbox=[xmin,ymin,xmax,ymax] -> scaled/padded/clamped."""
    x1,y1,x2,y2 = bbox
    x1 = x1 * sx + pad_x
    x2 = x2 * sx + pad_x
    y1 = y1 * sy + pad_y
    y2 = y2 * sy + pad_y
    x1 = float(np.clip(x1, 0, out_size-1))
    x2 = float(np.clip(x2, 0, out_size-1))
    y1 = float(np.clip(y1, 0, out_size-1))
    y2 = float(np.clip(y2, 0, out_size-1))
    return [x1,y1,x2,y2]
