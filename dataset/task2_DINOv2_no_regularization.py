import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(mode='train', img_size=518):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(), 
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# --- 2. Simple Image Reader ---
def read_img(path):
    img = np.array(Image.open(path).convert('RGB'))
    return img

class SPairDataset(Dataset):
    def __init__(self, pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype):
        self.datatype = datatype
        self.pck_alpha = pck_alpha
        self.pair_ann_path = pair_ann_path
        self.image_path = image_path
        
        list_file = os.path.join(layout_path, dataset_size, datatype + '.txt')
        self.ann_files = open(list_file, "r").read().split('\n')
        self.ann_files = [x for x in self.ann_files if x] 

        mode = 'train' if datatype == 'trn' else 'test'
        self.transform = get_transforms(mode=mode, img_size=518)

    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, idx):
        # 1. Load Metadata
        raw_line = self.ann_files[idx]
        ann_file = raw_line.replace(':', '_') + '.json'
        json_path = os.path.join(self.pair_ann_path, self.datatype, ann_file)

        with open(json_path) as f:
            annotation = json.load(f)

        category = annotation['category']
        src_path = os.path.join(self.image_path, category, annotation['src_imname'])
        trg_path = os.path.join(self.image_path, category, annotation['trg_imname'])

        # 2. Load Images
        src_img_raw = read_img(src_path)
        trg_img_raw = read_img(trg_path)

        # 3. Get Keypoints
        src_kps = np.array(annotation['src_kps']).astype(np.float32)
        trg_kps = np.array(annotation['trg_kps']).astype(np.float32)

        # 4. Apply Transforms
        src_aug = self.transform(image=src_img_raw, keypoints=src_kps)
        src_tensor = src_aug['image']
        src_kps_aug = np.array(src_aug['keypoints'])
        
        trg_aug = self.transform(image=trg_img_raw, keypoints=trg_kps)
        trg_tensor = trg_aug['image']
        trg_kps_aug = np.array(trg_aug['keypoints'])

        # 5. Padding Logic
        MAX_KPS = 40 
        src_kps_padded = np.zeros((MAX_KPS, 2), dtype=np.float32)
        trg_kps_padded = np.zeros((MAX_KPS, 2), dtype=np.float32)
        
        n_src = min(len(src_kps_aug), MAX_KPS)
        n_trg = min(len(trg_kps_aug), MAX_KPS)
        
        if n_src > 0: src_kps_padded[:n_src] = src_kps_aug[:n_src]
        if n_trg > 0: trg_kps_padded[:n_trg] = trg_kps_aug[:n_trg]

        # 6. Visibility Mask
        src_vis = self._check_visibility(src_kps_padded, 518, 518)
        trg_vis = self._check_visibility(trg_kps_padded, 518, 518)
        
        valid_mask = np.zeros(MAX_KPS, dtype=np.float32)
        common_len = min(n_src, n_trg)
        valid_mask[:common_len] = src_vis[:common_len] * trg_vis[:common_len]

        return {
            'src_img': src_tensor,
            'trg_img': trg_tensor,
            'src_kps': torch.from_numpy(src_kps_padded).float(), 
            'trg_kps': torch.from_numpy(trg_kps_padded).float(), 
            'valid_mask': torch.from_numpy(valid_mask).float(), 
            'category': category
        }

    def _check_visibility(self, kps, h, w):
        x = kps[:, 0]
        y = kps[:, 1]
        vis_x = (x >= 0) & (x < w)
        vis_y = (y >= 0) & (y < h)
        return (vis_x & vis_y).astype(np.float32)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    base_dir = os.path.join(project_root, 'data', 'SPair-71k_extracted', 'SPair-71k', 'SPair-71k')
    pair_ann_path = os.path.join(base_dir, 'PairAnnotation')
    layout_path = os.path.join(base_dir, 'Layout')
    image_path = os.path.join(base_dir, 'JPEGImages')

    if os.path.exists(base_dir):
        print(f"Found dataset at: {base_dir}")
        dataset = SPairDataset(pair_ann_path, layout_path, image_path, 'large', 0.05, 'trn')
        loader = DataLoader(dataset, batch_size=6, shuffle=True)
        batch = next(iter(loader))
        print(f"Batch Image Shape: {batch['src_img'].shape}") # Should be [4, 3, 518, 518]
        print(f"Batch Image Shape: {batch['trg_img'].shape}") # Should be [4, 3, 518, 518]
        print(f"Batch Keypoints: {batch['src_kps'].shape}")   # Should be [4, 40, 2]
    else:
        print(f"Path not found: {base_dir}")
        print("Run 'utils/data_setup.py' first!")