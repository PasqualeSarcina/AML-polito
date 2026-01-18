import json
import math
import os
import random

import numpy as np
import itertools

from data.dataset import CorrespondenceDataset
from data.dataset_downloader import download_ap10k


class AP10KDataset(CorrespondenceDataset):
    def __init__(self, datatype: str, transform=None, min_kps=3):
        super().__init__(dataset='ap10k', datatype=datatype, transform=transform)

        self.ap10kdir = os.path.join(self.dataset_dir, 'ap-10k')
        if not os.path.exists(self.ap10kdir):
            download_ap10k(self.dataset_dir)

        self.min_kps = min_kps

        ann_dir = os.path.join(self.ap10kdir, 'annotations')
        if self.datatype == 'trn':
            split1 = os.path.join(ann_dir, "ap10k-train-split1.json")
            split2 = os.path.join(ann_dir, "ap10k-train-split2.json")
            split3 = os.path.join(ann_dir, "ap10k-train-split3.json")
            merged_json = os.path.join(ann_dir, "ap10k-train-merged.jsonl")
            processed_json = os.path.join(ann_dir, "ap10k-train-processed.jsonl")
        elif self.datatype == "test":
            split1 = os.path.join(ann_dir, "ap10k-test-split1.json")
            split2 = os.path.join(ann_dir, "ap10k-test-split2.json")
            split3 = os.path.join(ann_dir, "ap10k-test-split3.json")
            merged_json = os.path.join(ann_dir, "ap10k-test-merged.jsonl")
            processed_json = os.path.join(ann_dir, "ap10k-test-processed.jsonl")
        else:
            split1 = os.path.join(ann_dir, "ap10k-val-split1.json")
            split2 = os.path.join(ann_dir, "ap10k-val-split2.json")
            split3 = os.path.join(ann_dir, "ap10k-val-split3.json")
            merged_json = os.path.join(ann_dir, "ap10k-val-merged.jsonl")
            processed_json = os.path.join(ann_dir, "ap10k-val-processed.jsonl")

        if not os.path.exists(merged_json):
            # merge splits
            data = self._load_data(split1, split2, split3)
            data = self._remove_duplicate_annotations(data)

            annotations = data["annotations"]
            images = data["images"]
            categories = data["categories"]

            images_dict = {image["id"]: image for image in images}
            categories_dict = {cat["id"]: cat for cat in categories}

            with open(merged_json, "w", encoding="utf-8") as f:
                for annotation in annotations:
                    image_id = annotation["image_id"]
                    category_id = annotation.get("category_id")

                    if image_id not in images_dict or category_id not in categories_dict:
                        continue

                    img_meta = images_dict[image_id]
                    cat_meta = categories_dict[category_id]

                    x_min, y_min, w, h = annotation["bbox"]
                    x_max = x_min + w
                    y_max = y_min + h

                    kps = np.array(annotation["keypoints"]).reshape(-1, 3)
                    kps_names = cat_meta.get("keypoints", [])

                    inside = (
                            (kps[:, 0] >= x_min) & (kps[:, 0] <= x_max) &
                            (kps[:, 1] >= y_min) & (kps[:, 1] <= y_max) &
                            (kps[:, 2] == 2)
                    )

                    filtered_named_kps = [
                        {"name": kps_names[i], "x": int(x), "y": int(y)}
                        for i, (x, y, v) in enumerate(kps)
                        if inside[i] and i < len(kps_names)
                    ]
                    if len(filtered_named_kps) < self.min_kps:
                        continue

                    image = {
                        "image_id": image_id,
                        "file_name": img_meta["file_name"],
                        "category": cat_meta["name"],
                        "supercategory": cat_meta["supercategory"],
                        "bbox": [x_min, y_min, x_max, y_max],
                        "keypoints": filtered_named_kps
                    }

                    f.write(json.dumps(image, ensure_ascii=False) + "\n")
            print(f"Merged annotation file created: {merged_json}")

        else:
            print(f"Merged annotation file already exists: {merged_json}")

        if not os.path.exists(processed_json):
            all_annotated_images = self._load_jsonl_as_dict(merged_json)

            with open(processed_json, "w", encoding="utf-8") as f:
                self._generate_intra_species_pairs(f, all_annotated_images)

                print("Intra-species pairs generated.")
                self._generate_cross_species_pairs(f, all_annotated_images)
                print("Cross-species pairs generated.")
                self._generate_cross_family_pairs(f, all_annotated_images)
                print("Cross-family pairs generated.")

            print(f"Processed annotation file created: {processed_json}")

        else:
            print(f"Processed annotation file already exists: {processed_json}")

        # load processed pairs
        with open(processed_json, "r", encoding="utf-8") as f:
            self.ann_files = f.read().splitlines()

        if self.datatype == 'test':
            self._load_distinct_images()

    def _load_annotation(self, idx):
        ann = json.loads(self.ann_files[idx])
        ann["pair_id"] = idx
        return ann

    def _build_image_path(self, imname, category=None):
        return os.path.join(self.ap10kdir, "data", imname)

    @staticmethod
    def _load_jsonl_as_dict(path):
        all_annotated_images = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                all_annotated_images[rec["image_id"]] = rec
        return all_annotated_images

    @staticmethod
    def _load_data(*file_paths):
        """Load and merge data from multiple JSON files."""
        merged_data = {'annotations': [], 'images': [], 'categories': []}
        for path in file_paths:
            with open(path, 'r') as file:
                data = json.load(file)
                merged_data['annotations'].extend(data['annotations'])
                merged_data['images'].extend(data['images'])
                # Assuming categories are the same across files or only needed from the first file
                if 'categories' in data and not merged_data['categories']:
                    merged_data['categories'] = data['categories']
        return merged_data

    @staticmethod
    def _remove_duplicate_annotations(data):
        """Remove duplicate annotations from the data based on image_id."""
        unique_image_ids = set()
        new_annotations = []
        for annotation in data['annotations']:
            if annotation['image_id'] not in unique_image_ids:
                unique_image_ids.add(annotation['image_id'])
                new_annotations.append(annotation)
        data['annotations'] = new_annotations
        return data

    @staticmethod
    def _match_keypoints(rec_a, rec_b):
        a_map = {kp["name"]: (kp["x"], kp["y"]) for kp in rec_a["keypoints"]}
        b_map = {kp["name"]: (kp["x"], kp["y"]) for kp in rec_b["keypoints"]}
        common = a_map.keys() & b_map.keys()
        src_kps = [a_map[n] for n in common]  # lista di tuple (x, y)
        trg_kps = [b_map[n] for n in common]
        return src_kps, trg_kps

    @staticmethod
    def _generate_intra_species_pairs(
            f,
            all_annotated_images,
            min_kps=3,
            n_multiplier=1,
            seed=42
    ):
        by_species = {}
        for rec in all_annotated_images.values():
            by_species.setdefault(rec["category"], []).append(rec)

        rng = random.Random(seed)
        total_written = 0

        for species in sorted(by_species.keys()):
            recs = by_species[species]

            if n_multiplier is not None:
                target_n = n_multiplier * len(recs)
            else:
                target_n = math.comb(len(recs), 2)

            possible_pairs = []
            for a, b in itertools.combinations(recs, 2):
                src_kps, trg_kps = AP10KDataset._match_keypoints(a, b)
                if len(src_kps) < min_kps:
                    continue
                possible_pairs.append({
                    "src_imname": a["file_name"],
                    "trg_imname": b["file_name"],
                    "category": species,
                    "src_kps": src_kps,
                    "trg_kps": trg_kps,
                    "src_bndbox": a["bbox"],
                    "trg_bndbox": b["bbox"],
                })

            target_n = min(target_n, len(possible_pairs))
            if target_n <= 0:
                continue

            sampled = rng.sample(possible_pairs, target_n)

            for pair in sampled:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            total_written += len(sampled)

        print(f"[INTRA-SPECIES] Total pairs: {total_written}")

    @staticmethod
    def _generate_cross_species_pairs(
            f,
            all_annotated_images,
            min_kps=3,
            n_pairs_per_combination=50,
            seed=42
    ):
        rng = random.Random(seed)
        total_written = 0

        fam_spec = {}
        for rec in all_annotated_images.values():
            fam = rec["supercategory"]
            spec = rec["category"]
            fam_spec.setdefault(fam, {}).setdefault(spec, []).append(rec)

        # ordine: supercategory
        for fam in sorted(fam_spec.keys()):
            spec_map = fam_spec[fam]
            specs = sorted(spec_map.keys())  # ordine: cat_a, cat_b

            if len(specs) <= 1:
                continue

            # ordine: cat_a-cat_b
            for cat_a, cat_b in itertools.combinations(specs, 2):
                A = spec_map[cat_a]
                B = spec_map[cat_b]

                cat_label = f"{fam}<{cat_a}-{cat_b}>"

                possible = []
                # direzione FISSA: a da cat_a (src), b da cat_b (trg)
                for a, b in itertools.product(A, B):
                    src_kps, trg_kps = AP10KDataset._match_keypoints(a, b)
                    if len(src_kps) < min_kps:
                        continue

                    possible.append((a, b, src_kps, trg_kps))

                N = min(n_pairs_per_combination, len(possible))
                if N <= 0:
                    continue

                sampled = rng.sample(possible, N)

                for a, b, src_kps, trg_kps in sampled:
                    pair_out = {
                        "src_imname": a["file_name"],  # sempre cat_a
                        "trg_imname": b["file_name"],  # sempre cat_b
                        "category": cat_label,
                        "src_kps": src_kps,
                        "trg_kps": trg_kps,
                        "src_bndbox": a["bbox"],
                        "trg_bndbox": b["bbox"],
                    }
                    f.write(json.dumps(pair_out, ensure_ascii=False) + "\n")

                total_written += len(sampled)

        print(f"[CROSS-SPECIES] Total pairs: {total_written}")

    @staticmethod
    def _generate_cross_family_pairs(
            f,
            all_annotated_images,
            min_kps=3,
            n_pairs_per_combination=20,
            seed=42
    ):
        rng = random.Random(seed)
        total_written = 0

        # famiglia -> list[rec]
        by_family = {}
        for rec in all_annotated_images.values():
            by_family.setdefault(rec["supercategory"], []).append(rec)

        # ORDINE: supercategory
        families = sorted(by_family.keys())

        # coppie di famiglie ordinate (fam1 < fam2)
        for fam1, fam2 in itertools.combinations(families, 2):
            A = by_family[fam1]
            B = by_family[fam2]

            # notebook: genera TUTTE le coppie tra famiglie (materializza)
            cross_family_pairs = list(itertools.product(A, B))

            # filtro richiesto: match_keypoints + min_kps
            possible_pairs = []
            for a, b in cross_family_pairs:
                src_kps, trg_kps = AP10KDataset._match_keypoints(a, b)
                if len(src_kps) < min_kps:
                    continue
                possible_pairs.append((a, b, src_kps, trg_kps))

            # notebook: campiona con cap
            N = min(n_pairs_per_combination, len(possible_pairs))
            if N <= 0:
                continue

            pairs_sampled = rng.sample(possible_pairs, N)

            # category = supercategoryA-supercategoryB
            cat_label = f"{fam1} - {fam2}"

            for a, b, src_kps, trg_kps in pairs_sampled:
                pair_out = {
                    "src_imname": a["file_name"],
                    "trg_imname": b["file_name"],
                    "category": cat_label,
                    "src_kps": src_kps,
                    "trg_kps": trg_kps,
                    "src_bndbox": a["bbox"],
                    "trg_bndbox": b["bbox"],
                }
                f.write(json.dumps(pair_out, ensure_ascii=False) + "\n")

            total_written += len(pairs_sampled)

        print(f"[CROSS-FAMILY] Total pairs: {total_written}")

    def _load_distinct_images(self):
        for line in self.ann_files:
            rec = json.loads(line)
            src_imname = rec["src_imname"]
            trg_imname = rec["trg_imname"]
            self.distinct_images['all'].add(src_imname)
            self.distinct_images['all'].add(trg_imname)

    def iter_test_distinct_images(self):
        if self.datatype != 'test':
            raise ValueError("Distinct images are available only for test set.")

        for img_name in self.distinct_images['all']:
            img_tensor = self._get_image(img_name)
            img_size = img_tensor.size()
            yield img_name, img_tensor, img_size
