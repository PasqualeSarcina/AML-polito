import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def read_img(path):
    """Reads image and converts to Tensor (C, H, W) in range [0, 255]"""
    img = Image.open(path).convert('RGB')
    return torch.tensor(np.array(img).transpose(2, 0, 1)).float()

class SPairDataset(Dataset):
    def __init__(self, pair_ann_path, layout_path, image_path, dataset_size='large', datatype='test'):
        self.datatype = datatype
        self.pair_ann_path = pair_ann_path
        self.image_path = image_path
        
        # Load the list of files (e.g., 'test.txt') from the layout folder
        list_file = os.path.join(layout_path, dataset_size, datatype + '.txt')
        self.ann_files = open(list_file, "r").read().split('\n')
        self.ann_files = [x for x in self.ann_files if x] # Clean empty lines
        
        # Standard ImageNet Normalization
        # Mean and Std are standard for pre-trained PyTorch models
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, idx):
        # 1. Load Annotation Metadata
        raw_line = self.ann_files[idx]
        ann_file = raw_line.replace(':', '_') + '.json'
        json_path = os.path.join(self.pair_ann_path, self.datatype, ann_file)

        with open(json_path) as f:
            annotation = json.load(f)

        # 2. Construct Image Paths
        category = annotation['category']
        src_path = os.path.join(self.image_path, category, annotation['src_imname'])
        trg_path = os.path.join(self.image_path, category, annotation['trg_imname'])
        
        # 3. Load & Normalize Images
        # We divide by 255.0 to get range [0, 1] before normalizing
        src_img = self.normalize(read_img(src_path) / 255.0)
        trg_img = self.normalize(read_img(trg_path) / 255.0)

        # 4. Return Dictionary
        return {
            'src_img': src_img,
            'trg_img': trg_img,
            'src_kps': torch.tensor(annotation['src_kps']).float(),
            'trg_kps': torch.tensor(annotation['trg_kps']).float(),
            'trg_bbox': annotation['trg_bndbox'], 
            'category': category
        }