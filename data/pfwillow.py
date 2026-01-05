import csv

from data.dataset import CorrespondenceDataset
import os
import numpy as np


class PFWillowDataset(CorrespondenceDataset):
    def __init__(self, dataset_dir: str, datatype: str, transform = None):
        if datatype != 'test':
            raise ValueError("PF-Willow dataset only supports 'test' datatype.")

        self.pfwillow_dir = os.path.join(dataset_dir, 'pf-willow')

        super().__init__(dataset='pfwillow', datatype=datatype, transform=transform)

        pairs_csv = os.path.join(self.pfwillow_dir, f"{datatype}_pairs.csv")

        with open(pairs_csv, newline="") as f:
            reader = csv.DictReader(f)
            self.ann_files = list(reader)

    def load_annotation(self, idx):
        r""" Loads the annotation of the pair with index idx """
        pair_info = self.ann_files[idx]

        category = pair_info['imageA'].split('/')[1]

        src_imname = pair_info["imageA"].split('/')[-1]
        trg_imname = pair_info["imageB"].split('/')[-1]

        n_pts = 10

        src_kps = np.zeros((n_pts, 2), dtype=np.float32)
        trg_kps = np.zeros((n_pts, 2), dtype=np.float32)

        for i in range(1, n_pts + 1):
            src_kps[i - 1, 0] = float(pair_info[f"XA{i}"])
            src_kps[i - 1, 1] = float(pair_info[f"YA{i}"])

            trg_kps[i - 1, 0] = float(pair_info[f"XB{i}"])
            trg_kps[i - 1, 1] = float(pair_info[f"YB{i}"])

        xmin = trg_kps[:, 0].min()
        ymin = trg_kps[:, 1].min()
        xmax = trg_kps[:, 0].max()
        ymax = trg_kps[:, 1].max()

        trg_bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

        return {
            "category": category,
            "src_imname": src_imname,
            "trg_imname": trg_imname,
            "src_kps": src_kps,  # shape (10, 2)
            "trg_kps": trg_kps,  # shape (10, 2)
            "trg_bndbox": trg_bbox
        }

    def build_image_path(self, imname, category = None):
        if category is None:
            raise ValueError("PF-Willow requires the category to build the image path.")
        return os.path.join(self.pfwillow_dir, "PF-dataset", category, imname)




