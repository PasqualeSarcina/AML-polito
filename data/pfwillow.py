import csv

from data.dataset import CorrespondenceDataset
import os
import numpy as np

from data.dataset_downloader import download_pfwillow


class PFWillowDataset(CorrespondenceDataset):
    def __init__(self, datatype: str, transform=None):
        if datatype != 'test':
            raise ValueError("PF-Willow dataset only supports 'test' datatype.")
        super().__init__(dataset='pfwillow', datatype=datatype, transform=transform)

        self.pfwillow_dir = os.path.join(self.dataset_dir, 'pf-willow')
        if not os.path.exists(self.pfwillow_dir):
            download_pfwillow(self.dataset_dir)

        pairs_csv = os.path.join(self.pfwillow_dir, f"{datatype}_pairs.csv")

        with open(pairs_csv, newline="") as f:
            reader = csv.DictReader(f)
            self.ann_files = list(reader)

        if self.datatype == 'test':
            self._load_distinct_images()

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

        xmin = trg_kps[:, 0].min()
        ymin = trg_kps[:, 1].min()
        xmax = trg_kps[:, 0].max()
        ymax = trg_kps[:, 1].max()

        trg_bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

        return {
            "pair_id": idx,
            "category": category,
            "src_imname": src_imname,
            "trg_imname": trg_imname,
            "src_kps": src_kps,  # shape (10, 2)
            "trg_kps": trg_kps,  # shape (10, 2)
            "trg_bndbox": trg_bbox
        }

    def _build_image_path(self, imname, category=None):
        if category is None:
            raise ValueError("PF-Willow requires the category to build the image path.")
        return os.path.join(self.pfwillow_dir, "PF-dataset", category, imname)

    def _load_distinct_images(self):
        for line in self.ann_files:
            category = line['imageA'].split('/')[1]
            src = line['imageA'].split('/')[-1]
            trg = line['imageB'].split('/')[-1]
            self.distinct_images[category].add(src)
            self.distinct_images[category].add(trg)

    def iter_test_distinct_images(self):
        if self.datatype != 'test':
            raise ValueError("Distinct images are available only for test set.")

        for img_name in self.distinct_images['all']:
            img_tensor = self._get_image(img_name, category=category)
            img_size = img_tensor.size()
            yield img_name, img_tensor, img_size
