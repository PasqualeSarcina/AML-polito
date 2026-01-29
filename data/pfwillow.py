import csv

from data.dataset import CorrespondenceDataset
import os
import numpy as np

from data.dataset_downloader import download_pfwillow


class PFWillowDataset(CorrespondenceDataset):
    def __init__(self, base_dir, transform=None):
        super().__init__(dataset='pfwillow', datatype="test", transform=transform, base_dir=base_dir)

        self.pfwillow_dir = os.path.join(self.dataset_dir, 'pf-willow')
        if not os.path.exists(self.pfwillow_dir):
            download_pfwillow(self.dataset_dir)

        pairs_csv = os.path.join(self.pfwillow_dir, "test_pairs.csv")

        with open(pairs_csv, newline="") as f:
            reader = csv.DictReader(f)
            self.ann_files = list(reader)

    def _load_annotation(self, idx):
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

        trg_xmin = trg_kps[:, 0].min()
        trg_ymin = trg_kps[:, 1].min()
        trg_xmax = trg_kps[:, 0].max()
        trg_ymax = trg_kps[:, 1].max()

        src_xmin = src_kps[:, 0].min()
        src_ymin = src_kps[:, 1].min()
        src_xmax = src_kps[:, 0].max()
        src_ymax = src_kps[:, 1].max()

        src_bbox = np.array([src_xmin, src_ymin, src_xmax, src_ymax], dtype=np.float32)

        trg_bbox = np.array([trg_xmin, trg_ymin, trg_xmax, trg_ymax], dtype=np.float32)

        return {
            "pair_id": idx,
            "category": category,
            "src_imname": src_imname,
            "trg_imname": trg_imname,
            "src_kps": src_kps,  # shape (10, 2)
            "trg_kps": trg_kps,  # shape (10, 2)
            "src_bndbox": src_bbox,
            "trg_bndbox": trg_bbox
        }

    def _build_image_path(self, imname, category=None):
        if category is None:
            raise ValueError("PF-Willow requires the category to build the image path.")
        return os.path.join(self.pfwillow_dir, "PF-dataset", category, imname)

    def get_categories(self) -> set:
        categories = set()
        for line in self.ann_files:
            category = line['imageA'].split('/')[1]
            categories.add(category)
        return categories
