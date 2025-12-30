import json

from data.dataset import CorrespondenceDataset
import os


class SPairDataset(CorrespondenceDataset):
    def __init__(self, dataset_size: str, dataset_dir: str, datatype: str):
        self.spair_dir = os.path.join(dataset_dir, 'SPair-71k')

        super().__init__(dataset='spair', datatype=datatype)
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
