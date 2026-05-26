import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import json
import random
from segment_anything.utils.transforms import ResizeLongestSide

class SAMTransform(object):
    def __init__(self, target_size=1024, datatype='trn'):
        self.target_size = target_size
        self.resizer = ResizeLongestSide(target_size)
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.mode = datatype
        self.color_jitter = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.blur = v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        
    def __call__(self, sample):
        do_flip = (self.mode == 'trn') and (random.random() > 0.5)

        for key in ["src", "trg"]:
            img = sample[f"{key}_img"]  # CHW
            H, W = int(img.shape[-2]), int(img.shape[-1])

            # AUGMENTATION
            if do_flip:
                img = torch.flip(img, dims=[-1])                            # flip image

                sample[f"{key}_kps"][:, 0] = W - sample[f"{key}_kps"][:, 0] # flip keypoints

                bb = sample[f"{key}_bndbox"]                                # flip bndbox
                x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
                new_x1 = W - x2
                new_x2 = W - x1
                sample[f"{key}_bndbox"] = torch.tensor([new_x1, y1, new_x2, y2])

            if self.mode=='trn':
                if random.random() > 0.5:
                    img = self.color_jitter(img)
                if random.random() > 0.5:
                    img = self.blur(img)

            # RESIZING
            # resizing immagine
            newh, neww = self.resizer.get_preprocess_shape(H, W, self.target_size)
            img_resized = F.interpolate(img.unsqueeze(0), (newh, neww), mode="bilinear", align_corners=False, antialias=True)[0]

            # scala coordinate keypoints
            sample[f"{key}_kps"] = self.resizer.apply_coords_torch(         # N, 2
                sample[f"{key}_kps"], (H, W)
            )

            # bbox: assicuriamoci sia tensor float, shape (1,4) per apply_boxes_torch
            bb = sample[f"{key}_bndbox"]
            if not torch.is_tensor(bb):
                bb = torch.tensor(bb, dtype=torch.float32)
            else:
                bb = bb.float()

            # scala coordinate boundingbox
            sample[f"{key}_bndbox"] = self.resizer.apply_boxes_torch(     
                bb.view(1, 4), (H, W)
            ).view(4)                                                       # 4

            # IMAGE NORMALIZATION AND PADDING
            # normalize colors
            img_resized = (img_resized - self.pixel_mean) / self.pixel_std
            # Pad
            padh = self.target_size - newh
            padw = self.target_size - neww
            img_resized = F.pad(img_resized, (0, padw, 0, padh))

            # immagine resized + normalized + padding
            sample[f"{key}_img"] = img_resized.squeeze(0)   # C, H', W'
            sample[f"{key}_orig_size"] = (H, W)
            sample[f"{key}_resized_size"] = (newh, neww)
            sample[f"{key}_scale"] = (newh / H, neww / W)  # (sy, sx)

        return sample
    
def read_img(path):
    img = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
    return torch.from_numpy(img).permute(2,0,1)

class SPairDataset(Dataset):
    def __init__(self, pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype):
        self.datatype = datatype
        self.pck_alpha = pck_alpha
        self.pair_ann_path = pair_ann_path
        self.image_path = image_path
        self.max_kps = 50

        split_file = os.path.join(layout_path, dataset_size, datatype + '.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"File non trovato: {split_file}")

        with open(split_file, "r") as f:
            self.ann_files = [line.strip() for line in f.readlines() if line.strip()]

        self.transform = SAMTransform(1024, datatype)

    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, idx):
        ann_filename = self.ann_files[idx]

        json_path = os.path.join(self.pair_ann_path, self.datatype, ann_filename + '.json')

        if not os.path.exists(json_path):
            safe_ann_filename = ann_filename.replace(":", "_")
            json_path = os.path.join(self.pair_ann_path, self.datatype, safe_ann_filename + '.json')

        with open(json_path, 'r') as f:
            annotation = json.load(f)

        category = annotation['category']
        src_img = read_img(os.path.join(self.image_path, category, annotation['src_imname']))
        trg_img = read_img(os.path.join(self.image_path, category, annotation['trg_imname']))

        raw_src_kps = torch.tensor(annotation['src_kps']).float()   # lista di coordinate x,y es. ((1,2), (3,4), (5,6))
        raw_trg_kps = torch.tensor(annotation['trg_kps']).float()
        num_kps = raw_src_kps.shape[0]                              # numero di kps  es. 3

        src_kps = torch.zeros((self.max_kps, 2))                    # [max_kps, 2]  ((0, 0),...(0,0))
        trg_kps = torch.zeros((self.max_kps, 2))
        kps_valid = torch.zeros(self.max_kps, dtype=torch.bool)     # [max_kps] (False, False, ..., False)

        src_kps[:num_kps] = raw_src_kps                             # [max_kps, 2] ((1,2), (3,4), (5,6), (0, 0),...(0,0))
        trg_kps[:num_kps] = raw_trg_kps
        kps_valid[:num_kps] = True                                  # [max_kps] (True, True, True, False, ..., False)

        sample = {
            'src_img': src_img,
            'trg_img': trg_img,
            'src_kps': src_kps,
            'trg_kps': trg_kps,
            'kps_valid': kps_valid,
            'num_kps': num_kps,
            'category': category,
            'src_bndbox': torch.tensor(annotation['src_bndbox']).float(),
            'trg_bndbox': torch.tensor(annotation['trg_bndbox']).float()
        }
        
        # scala immagine, keypoints e bounding box
        if self.transform: sample = self.transform(sample)
        return sample