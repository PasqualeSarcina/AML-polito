from typing import Any

from torchvision import transforms


class PreProcess(object):
    def __init__(self, out_dim=(518, 518)):
        self.out_dim = out_dim  # (H, W)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    @staticmethod
    def _resize_points(pts, sy, sx):
        # pts: (..., 2)
        pts[..., 0] = pts[..., 0] * sx
        pts[..., 1] = pts[..., 1] * sy
        return pts

    @staticmethod
    def _resize_bbox_xyxy(bb, sy, sx):
        # bb: (x1,y1,x2,y2)
        bb[0] *= sx
        bb[2] *= sx
        bb[1] *= sy
        bb[3] *= sy
        return bb

    def __call__(self, sample: dict[str, Any]):
        h_out, w_out = self.out_dim

        for key in ['src', 'trg']:
            img = sample[f'{key}_img']
            _, h, w = img.shape
            sy, sx = h_out / h, w_out / w

            sample[f'{key}_img'] = self.normalize(
                transforms.Resize((h_out, w_out))(img) / 255.0
            )
            sample[f'{key}_kps'] = self._resize_points(sample[f'{key}_kps'], sy, sx)
            sample[f'{key}_bndbox'] = self._resize_bbox_xyxy(sample[f'{key}_bndbox'], sy, sx)
            sample[f'{key}_scale'] = (sy, sx)

        return sample
