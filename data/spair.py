import json
from pathlib import Path

import torch

from data.dataset import CorrespondenceDataset
import os

from data.dataset_downloader import download_spair


class SPairDataset(CorrespondenceDataset):
    def __init__(self, dataset_size: str, datatype: str, transform = None):
        super().__init__(dataset='spair', datatype=datatype, transform=transform)

        self.spair_dir = os.path.join(self.dataset_dir, 'SPair-71k')
        if not os.path.exists(self.spair_dir):
            download_spair(self.dataset_dir)


        self.ann_files = open(
            os.path.join(self.spair_dir, 'Layout', dataset_size, self.datatype + '.txt'),
            "r").read().split('\n')
        self.ann_files = self.ann_files[:len(self.ann_files) - 1]

    def load_annotation(self, idx):
        r""" Loads the annotation of the pair with index idx """
        ann_filename = self.ann_files[idx]
        ann_file = ann_filename + '.json'
        json_path = os.path.join(self.spair_dir, 'PairAnnotation', self.datatype, ann_file)

        with open(json_path) as f:
            annotation = json.load(f)

        return annotation

    def build_image_path(self, imname, category=None):
        if category is None:
            raise ValueError("SPair requires the category to build the image path.")
        return os.path.join(self.spair_dir, "JPEGImages", category, imname)

    def iter_images(self):
        img_root = Path(self.spair_dir) / "JPEGImages"

        for category_dir in img_root.iterdir():
            category = category_dir.name

            for img_path in category_dir.iterdir():

                img_tensor = self.get_image(img_path.name, category)

                yield img_tensor, img_tensor.size(), category, img_path.name

    def num_images(self):
        img_root = Path(self.spair_dir) / "JPEGImages"
        exts = {".jpg", ".jpeg", ".png"}
        return sum(
            1
            for category_dir in img_root.iterdir() if category_dir.is_dir()
            for p in category_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
