import json
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

        if self.datatype == 'test':
            self._load_distinct_images()


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

    def _load_distinct_images(self):
        for line in self.ann_files:
            files, category = line.split(":")
            _, src, trg, *_ = files.split("-")
            self.distinct_images[category].add(src)
            self.distinct_images[category].add(trg)

    def iter_test_distinct_images(self):
        if self.datatype != 'test':
            raise ValueError("Distinct images are available only for test set.")

        for category, img_set in self.distinct_images.items():
            for img_name in img_set:
                img_tensor = self._get_image(img_name + ".jpg", category=category)
                img_size = img_tensor.size()
                yield img_name, img_tensor, img_size


