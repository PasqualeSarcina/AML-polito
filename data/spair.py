import json
from typing import Literal

from data.dataset import CorrespondenceDataset
import os

from data.dataset_downloader import download_spair


class SPairDataset(CorrespondenceDataset):
    def __init__(self, dataset_size: str, datatype: Literal["train", "test", "val"], base_dir, transform = None):
        super().__init__(dataset='spair', datatype=datatype, transform=transform, base_dir=base_dir)

        self.spair_dir = os.path.join(self.dataset_dir, 'SPair-71k')
        if not os.path.exists(self.spair_dir):
            download_spair(self.dataset_dir)


        self.ann_files = open(
            os.path.join(self.spair_dir, 'Layout', dataset_size, self.datatype + '.txt'),
            "r").read().split('\n')
        self.ann_files = self.ann_files[:len(self.ann_files) - 1]

    def _load_annotation(self, idx):
        r""" Loads the annotation of the pair with index idx """
        ann_filename = self.ann_files[idx]
        ann_file = ann_filename + '.json'
        json_path = os.path.join(self.spair_dir, 'PairAnnotation', self.datatype, ann_file)

        with open(json_path) as f:
            annotation = json.load(f)

        return annotation

    def _build_image_path(self, imname, category=None):
        if category is None:
            raise ValueError("SPair requires the category to build the image path.")
        return os.path.join(self.spair_dir, "JPEGImages", category, imname)

    def get_categories(self) -> set:
        categories = set()
        for line in self.ann_files:
            categories.add(line.split(":")[1])
        return categories


