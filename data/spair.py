import json

from data.dataset import CorrespondenceDataset
import os

class SPairDataset(CorrespondenceDataset):
    def __init__(self, dataset_size: str):
        super().__init__(dataset ='spair')
        self.ann_files = open(os.path.join(self.dataset_dir, 'Layout', dataset_size, self.split + '.txt'), "r").read().split('\n')
        self.ann_files = self.ann_files[:len(self.ann_files) - 1]

    def load_annotation(self, idx):
        r""" Loads the annotation of the pair with index idx """
        ann_filename = self.ann_files[idx]
        ann_file = ann_filename + '.json'
        json_path = os.path.join(self.dataset_dir, 'PairAnnotation', self.split, ann_file)

        with open(json_path) as f:
            annotation = json.load(f)

        return annotation
