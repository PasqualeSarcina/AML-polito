import csv

import scipy.io as sio
import os
import numpy as np

from data.dataset import CorrespondenceDataset
from data.dataset_downloader import download_pfpascal


class PFPascalDataset(CorrespondenceDataset):
    def __init__(self, datatype: str, transform = None):
        super().__init__(dataset='pfpascal', datatype=datatype, transform=transform)

        self.pfpascal_dir = os.path.join(self.dataset_dir, 'pf-pascal')
        if not os.path.exists(self.pfpascal_dir):
            download_pfpascal(self.dataset_dir)


        pairs_csv = os.path.join(self.pfpascal_dir, f"{datatype}_pairs.csv")

        with open(pairs_csv, newline="") as f:
            reader = csv.DictReader(f)
            self.ann_files = list(reader)

        self.categories = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def _load_annotation(self, idx):
        r""" Loads the annotation of the pair with index idx """
        pair_info = self.ann_files[idx]

        src_imname = pair_info["source_image"].split('/')[-1]
        trg_imname = pair_info["target_image"].split('/')[-1]

        category = self.categories[int(pair_info["class"]) - 1]

        src_kps, src_bbox, src_imsize = self._read_mat(
            os.path.join(self.pfpascal_dir, 'PF-dataset-PASCAL', 'Annotations', category,
                         src_imname.replace('.jpg', '.mat')))
        trg_kps, trg_bbox, trg_imsize = self._read_mat(
            os.path.join(self.pfpascal_dir, 'PF-dataset-PASCAL', 'Annotations', category,
                         trg_imname.replace('.jpg', '.mat')))

        src_kps, trg_kps = self._filter_valid_keypoints(src_kps, trg_kps)

        annotation = {
            "pair_id": idx,
            "src_imname": src_imname,
            "trg_imname": trg_imname,
            "category": category,
            "src_kps": src_kps,
            "trg_kps": trg_kps,
            "src_bndbox": src_bbox,
            "trg_bndbox": trg_bbox,
            "src_imsize": src_imsize,
            "trg_imsize": trg_imsize
        }
        return annotation

    @staticmethod
    def _read_mat(path):
        r"""Reads a PF-Pascal .mat annotation file."""
        mat = sio.loadmat(path)

        # --- Keypoints ---
        # Original shape: (16,2), NaN for missing keypoints
        kps = np.array(mat["kps"], dtype=np.float32)

        # --- Bounding box ---
        bbox = np.array(mat["bbox"], dtype=np.float32).reshape(-1)  # (4,)

        # --- Image size ---
        imsize = np.array(mat["imsize"], dtype=np.float32).reshape(-1)  # (2,)

        return kps, bbox, imsize

    @staticmethod
    def _filter_valid_keypoints(src_kps, trg_kps):
        """
        Mantiene solo i keypoint validi in entrambe le immagini.
        src_kps, trg_kps: (N,2)
        """
        src_valid = ~np.isnan(src_kps).any(axis=1)
        trg_valid = ~np.isnan(trg_kps).any(axis=1)
        valid = src_valid & trg_valid

        return src_kps[valid], trg_kps[valid]

    def _build_image_path(self, imname, category=None):
        return os.path.join(self.pfpascal_dir, "PF-dataset-PASCAL", "JPEGImages", imname)

    def iter_test_distinct_images(self):
        if self.datatype != 'test':
            raise ValueError("Distinct images are available only for test set.")
        for line in self.ann_files:
            src_imname = line["source_image"].split('/')[-1]
            trg_imname = line["target_image"].split('/')[-1]
            self.distinct_images['all'].add(src_imname)
            self.distinct_images['all'].add(trg_imname)

        for img_name in self.distinct_images['all']:
            img_tensor = self._get_image(img_name)
            img_size = img_tensor.size()
            yield img_name, img_tensor, img_size

