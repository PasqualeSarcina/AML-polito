#CREAZIONE DATASET E DATALOADER
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import random
from segment_anything.utils.transforms import ResizeLongestSide

class SAMTransform(object):
    def __init__(self, target_size=1024):
        self.target_size = target_size
        self.transform_official = ResizeLongestSide(target_size)
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __call__(self, sample):
        sample = sample.copy()
        
        pairs = [('src_img', 'src_kps', 'src_bbox'), ('trg_img', 'trg_kps', 'trg_bbox')]

        for img_key, kps_key, bbox_key in pairs:
            img = sample[img_key]
            
            original_h, original_w = img.shape[-2:]
            
            new_h, new_w = self.transform_official.get_preprocess_shape(original_h, original_w, self.target_size)

            scale_h = new_h / original_h
            scale_w = new_w / original_w
            scale = scale_h

            img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
            
            img = (img - self.pixel_mean) / self.pixel_std

            pad_h = self.target_size - new_h
            pad_w = self.target_size - new_w
            # Pad order: (left, right, top, bottom)
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0)

            if kps_key in sample:
                sample[kps_key] = sample[kps_key] * scale
            
            if bbox_key in sample:
                sample[bbox_key] = sample[bbox_key] * scale

            sample[img_key] = img

            if img_key == 'trg_img':
                sample['pck_threshold'] = sample['pck_threshold'] * scale
                sample['scale'] = scale
                sample['pad_w'] = pad_w
                sample['pad_h'] = pad_h

        return sample
    
def read_img(path):
    img = np.array(Image.open(path).convert('RGB'))
    return torch.tensor(img.transpose(2, 0, 1).astype(np.float32))

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

        self.transform = SAMTransform(target_size=1024)

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

        raw_src_kps = torch.tensor(annotation['src_kps']).float()
        raw_trg_kps = torch.tensor(annotation['trg_kps']).float()
        num_kps = raw_src_kps.shape[0]

        src_kps = torch.zeros((self.max_kps, 2))
        trg_kps = torch.zeros((self.max_kps, 2))

        kps_valid = torch.zeros(self.max_kps, dtype=torch.bool)

        src_kps[:num_kps] = raw_src_kps
        trg_kps[:num_kps] = raw_trg_kps
        kps_valid[:num_kps] = True

        trg_bbox = annotation['trg_bndbox']
        pck_threshold = max(trg_bbox[2] - trg_bbox[0], trg_bbox[3] - trg_bbox[1]) * self.pck_alpha

        sample = {
            'src_img': src_img,
            'trg_img': trg_img,
            'src_kps': src_kps,
            'trg_kps': trg_kps,
            'kps_valid': kps_valid,
            'num_kps': num_kps,
            'category': category,
            'pck_threshold': pck_threshold,
            'src_bbox': torch.tensor(annotation['src_bndbox']).float(),
            'trg_bbox': torch.tensor(annotation['trg_bndbox']).float()
        }

        # --- DATA AUGMENTATION: RANDOM HORIZONTAL FLIP ---
        # Applichiamo il flip nel 50% dei casi durante il training
        if self.datatype == 'trn' and random.random() > 0.5:
            sample['src_img'] = torch.flip(sample['src_img'], dims=[-1])
            sample['trg_img'] = torch.flip(sample['trg_img'], dims=[-1])

            # Flip Keypoints: la coordinata X deve essere invertita rispetto alla larghezza originale
            original_w_src = sample['src_img'].shape[-1]
            original_w_trg = sample['trg_img'].shape[-1]

            # x_new = width - x_old
            num = sample['num_kps']
            sample['src_kps'][:num, 0] = original_w_src - 1 - sample['src_kps'][:num, 0]
            sample['trg_kps'][:num, 0] = original_w_trg - 1 - sample['trg_kps'][:num, 0]

            # x_min_new = width - x_max_old, x_max_new = width - x_min_old
            s_bbox = sample['src_bbox']
            sample['src_bbox'] = torch.tensor([original_w_src - 1 - s_bbox[2], s_bbox[1], original_w_src - 1 - s_bbox[0], s_bbox[3]])

            t_bbox = sample['trg_bbox']
            sample['trg_bbox'] = torch.tensor([original_w_trg - 1 - t_bbox[2], t_bbox[1], original_w_trg - 1 - t_bbox[0], t_bbox[3]])

        if self.transform: sample = self.transform(sample)
        return sample