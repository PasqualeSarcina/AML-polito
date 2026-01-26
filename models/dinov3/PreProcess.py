import math
from typing import Any, Tuple

import torch
from torchvision import transforms
import torch.nn.functional as F


class PreProcess(object):
    def __init__(self):
        self.PATCH = 16
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    @staticmethod
    def _pad_to_multiple(img_chw: torch.Tensor, k: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Pad SOLO a destra e in basso.
        Ritorna (img_pad, (new_h, new_w)).
        """

        _, h, w = img_chw.shape
        new_h = int(math.ceil(h / k) * k)
        new_w = int(math.ceil(w / k) * k)
        pad_bottom = new_h - h
        pad_right = new_w - w

        if pad_bottom == 0 and pad_right == 0:
            return img_chw, (0, 0)

        # F.pad uses (left, right, top, bottom) for 2D pads on last two dims
        img_pad = F.pad(img_chw, (0, pad_right, 0, pad_bottom), value=0.0)
        return img_pad, (new_h, new_w)

    def __call__(self, sample: dict[str, Any]):

        for key in ['src', 'trg']:
            img = sample[f'{key}_img']

            # original size coerente con kps/bbox (prima del padding)
            H, W = int(img.shape[-2]), int(img.shape[-1])

            img_pad, (new_h, new_w) = self._pad_to_multiple(img, self.PATCH)

            sample[f'{key}_img'] = self.normalize(img_pad / 255.0)
            sample[f"{key}_orig_size"] = (H, W)
            sample[f"{key}_resized_size"] = (new_h, new_w)

        return sample