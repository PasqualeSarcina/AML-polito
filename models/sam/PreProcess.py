from typing import Any, Dict
import torch


class PreProcess(object):
    def __init__(self, sam_transform):
        self.transform = sam_transform  # es. predictor.transform

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["src", "trg"]:
            img = sample[f"{key}_img"]  # CHW

            # original size coerente con kps/bbox (prima del resize SAM)
            H, W = int(img.shape[-2]), int(img.shape[-1])

            img_resized = self.transform.apply_image_torch(img.unsqueeze(0))  # BCHW

            # scala coords e bbox con le utilità SAM
            sample[f"{key}_kps"] = self.transform.apply_coords_torch(
                sample[f"{key}_kps"], (H, W)
            )

            # bbox: assicurati sia tensor float, shape (1,4) per apply_boxes_torch
            bb = sample[f"{key}_bndbox"]
            if not torch.is_tensor(bb):
                bb = torch.tensor(bb, dtype=torch.float32)
            else:
                bb = bb.float()

            sample[f"{key}_bndbox"] = self.transform.apply_boxes_torch(
                bb.view(1, 4), (H, W)
            ).view(4)

            # salva immagine resized
            sample[f"{key}_img"] = img_resized

            # salva scale e resized shape (utili per debug / conversioni)
            new_h, new_w = self.transform.get_preprocess_shape(H, W, self.transform.target_length)
            sample[f"{key}_orig_size"] = (H, W)
            sample[f"{key}_resized_size"] = (new_h, new_w)
            sample[f"{key}_scale"] = (new_h / H, new_w / W)  # (sy, sx)

        return sample
