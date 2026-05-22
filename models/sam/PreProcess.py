from typing import Any, Dict
import torch
import torch.nn.functional as F


class PreProcess(object):
    def __init__(self, sam_transform):
        self.transform = sam_transform  # es. predictor.transform

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["src", "trg"]:
            img = sample[f"{key}_img"]  # CHW

            # original size coherent with kps/bbox before resize
            H, W = int(img.shape[-2]), int(img.shape[-1])

            # Resize based on transform.apply_image_torch
            new_h, new_w = self.transform.get_preprocess_shape(H, W, 1024)
            img_resized= F.interpolate(img.unsqueeze(0), (new_h, new_w), mode="bilinear", align_corners=False, antialias=True)

            # Scale coordinates
            sample[f"{key}_kps"] = self.transform.apply_coords_torch(
                sample[f"{key}_kps"], (H, W)
            )

            # Scale bbox
            bb = sample[f"{key}_bndbox"]
            if not torch.is_tensor(bb):
                bb = torch.tensor(bb, dtype=torch.float32)
            else:
                bb = bb.float()

            sample[f"{key}_bndbox"] = self.transform.apply_boxes_torch(
                bb.view(1, 4), (H, W)
            ).view(4)


            sample[f"{key}_img"] = img_resized
            new_h, new_w = int(img_resized.shape[-2]), int(img_resized.shape[-1])
            sample[f"{key}_orig_size"] = (H, W)
            sample[f"{key}_resized_size"] = (new_h, new_w)
            sample[f"{key}_scale"] = (new_h / H, new_w / W)  # (sy, sx)

        return sample
