r""" Superclass for semantic correspondence datasets """

import os

from torch.ao.nn.quantized.functional import threshold
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np


def get_pckthres(bb_annotation, alpha: float):
    r""" Computes PCK threshold """
    tx1, ty1, tx2, ty2 = bb_annotation
    threshold = float(max(tx2 - tx1, ty2 - ty1) * alpha)
    return threshold


class CorrespondenceDataset(Dataset):
    r""" Parent class of PFPascal, PFWillow, and SPair """

    def __init__(self, dataset: str, split: str, dataset_dir: str):
        '''
        dataset: pfwillow, pfpascal, spair.
        img_size: image will be resized to this size。
        datapath: path to the benchmark folder.
        thres: bbox or img, the length used to measure pck.
        split: trn, test or val.
        '''
        """ CorrespondenceDataset constructor """
        super().__init__()
        self.dataset = dataset
        self.img_path = None
        self.split = split
        self.dataset_dir = dataset_dir
        self.ann_files = None

    def __len__(self):
        r""" Returns the number of pairs """
        return len(self.ann_files)

    def __getitem__(self, idx):
        r""" Constructs and return a batch """
        annotation = self.load_annotation(idx)

        # Image name
        batch = dict()

        # Object category
        batch['category'] = annotation['category']

        # Image as tensor
        batch['src_img'] = self.get_image(annotation['src_imname'])
        batch['trg_img'] = self.get_image(annotation['trg_imname'])

        # Key-points
        batch['src_kps'] = torch.tensor(annotation['src_kps'], dtype=torch.float32)
        batch['trg_kps'] = torch.tensor(annotation['trg_kps'], dtype=torch.float32)

        batch['pck_threshold_0_05'] = get_pckthres(annotation['trg_bndbox'], 0.05)
        batch['pck_threshold_0_1'] = get_pckthres(annotation['trg_bndbox'], 0.1)
        batch['pck_threshold_0_2'] = get_pckthres(annotation['trg_bndbox'], 0.2)

        return batch

    def get_image(self, imname):
        r""" Reads PIL image from path """
        path = os.path.join(self.img_path, imname)
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
        img = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
        return img

