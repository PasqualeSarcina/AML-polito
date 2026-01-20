r""" Superclass for semantic correspondence datasets """

import os
from collections import defaultdict
from pathlib import Path

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

    def __init__(self, dataset: str, datatype: str, transform = None):
        '''
        dataset: pfwillow, pfpascal, spair.
        datatype: trn, test or val.
        '''
        """ CorrespondenceDataset constructor """
        super().__init__()
        self.dataset_dir = os.path.join(os.path.dirname(Path(__file__).absolute()), '..', 'dataset')
        self.dataset = dataset
        self.datatype = datatype
        self.ann_files = None
        self.transform = transform

    def __len__(self):
        r""" Returns the number of pairs """
        return len(self.ann_files)

    def __getitem__(self, idx):
        r""" Constructs and return a batch """
        annotation = self._load_annotation(idx)

        # Image as tensor
        src_img = self._get_image(annotation["src_imname"], category=annotation["category"])
        trg_img = self._get_image(annotation["trg_imname"], category=annotation["category"])
        # Image name
        sample = dict()

        # Pair ID
        sample["pair_id"] = annotation["pair_id"]

        # Object category
        sample['category'] = annotation['category']

        sample['src_imname'] = annotation["src_imname"]
        sample['src_img'] = src_img

        sample['trg_imname'] = annotation["trg_imname"]
        sample['trg_img'] = trg_img

        sample['src_bndbox'] = annotation["src_bndbox"]
        sample['trg_bndbox'] = annotation["trg_bndbox"]

        # Key-points
        sample['src_kps'] = torch.tensor(annotation['src_kps'], dtype=torch.float32)
        sample['trg_kps'] = torch.tensor(annotation['trg_kps'], dtype=torch.float32)

        # Apply transform (e.g., SAM preprocessing)
        if self.transform:
            sample = self.transform(sample)

        # Compute and store image sizes
        sample['src_imsize'] = sample['src_img'].size()
        sample['trg_imsize'] = sample['trg_img'].size()

        # Compute PCK thresholds
        sample['pck_threshold_0_05'] = get_pckthres(sample['trg_bndbox'], 0.05)
        sample['pck_threshold_0_1'] = get_pckthres(sample['trg_bndbox'], 0.1)
        sample['pck_threshold_0_2'] = get_pckthres(sample['trg_bndbox'], 0.2)

        return sample

    def _get_image(self, imname: str, category: str | None = None):
        path = self._build_image_path(imname, category)
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1)


    def _build_image_path(self, imname: str, category: str | None = None) -> str:
        """Hook: subclasses must implement this method to build image path """
        raise NotImplementedError

    def _load_annotation(self, idx: int) -> dict:
        """Hook: subclasses must implement this method to load annotation """
        raise NotImplementedError

    def get_categories(self) -> set:
        """Hook: subclasses must implement this method to return the set of categories """
        raise NotImplementedError
