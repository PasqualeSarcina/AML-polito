import json
import math
import os
import random
from typing import Literal

import numpy as np
import itertools

from data.dataset import CorrespondenceDataset
from data.dataset_downloader import download_ap10k


class AP10KDataset(CorrespondenceDataset):
    def __init__(self, datatype: Literal["train", "test", "val"], base_dir, transform=None, min_kps=3):
        super().__init__(dataset='ap10k', datatype=datatype, base_dir=base_dir, transform=transform)

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
        """Match keypoints between two annotated records by keypoint name."""
        a_map = {kp["name"]: (kp["x"], kp["y"]) for kp in rec_a["keypoints"]}
        b_map = {kp["name"]: (kp["x"], kp["y"]) for kp in rec_b["keypoints"]}
        # Keep only the common keypoints name (key)
        common = a_map.keys() & b_map.keys()
        src_kps = [a_map[n] for n in common]
        trg_kps = [b_map[n] for n in common]
        return src_kps, trg_kps

    @staticmethod
    def _generate_intra_species_pairs(
            f,
            all_annotated_images,
            min_kps=3,
            n_multiplier=7,
            max_categories=7,
            seed=42
    ):
        """
        Generate and write correspondence pairs of images from the same animal species.
        Pairs are valid only if they share at least `min_kps` matching keypoints (by name).

        Args:
            f: An open file object where pairs will be written as JSON lines.
            all_annotated_images: A dictionary mapping image_id to annotation records.
            min_kps: Minimum number of matching keypoints required for a pair
                to be valid. Defaults to 3. Pairs with fewer common keypoints are discarded.
            n_multiplier: Multiplier for the target number of pairs per species.
                Defaults to 7.
            max_categories: Maximum number of species (categories) to include.
                If there are more species than this value, a random subset is selected.
                Defaults to 7.
            seed: Random seed for reproducibility of species selection and pair sampling. Defaults to 42.
        """
        by_species = {}

        # Insert in the dictionary all the images grouped by category
        for rec in all_annotated_images.values():
            by_species.setdefault(rec["category"], []).append(rec)

        # Set a seed for reproducibility
        rng = random.Random(seed)

        # Sort the dictionary in a reproducible way
        total_written = 0
        species = list(by_species.keys())
        rng.shuffle(species)
        # Keep only the selected categories
        keep = set(species[:max_categories])
        by_species = {k: v for k, v in by_species.items() if k in keep}

        for species in sorted(by_species.keys()):
            recs = by_species[species]

            # Cap for number of generated pairs, based on n_multiplier and the number of images for the current species
            target_n = n_multiplier * len(recs)

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

            # Prevents taking more pairs than available
            target_n = min(target_n, len(possible_pairs))
            if target_n <= 0:
                continue

            # Sample target_n pairs from all the combination
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
            n_pairs_per_family=506,  # <-- nuovo cap per famiglia
            seed=42,
            max_families=8
    ):
        """
        Generate and write correspondence pairs of images from different species within the same family.
        Pairs are valid only if they share at least `min_kps` matching keypoints (by name).

        Args:
            f: An open file object where pairs will be written as JSON lines.
            all_annotated_images: A dictionary mapping image_id to annotation records.
        min_kps: Minimum number of matching keypoints required for a pair to be valid.
            Defaults to 3. Pairs with fewer common keypoints are discarded.
        n_pairs_per_family: Maximum cap on the number of pairs to generate per family.
            Defaults to 506. Uses reservoir sampling to ensure uniform random selection if more
            candidates exist than this limit.
        seed: Random seed for reproducibility of family selection and pair sampling.
            Defaults to 42.
        max_families: Maximum number of families (supercategories) to include.
            If there are more families with at least 2 species each, a random subset is selected.
            Defaults to 8.
        """
        rng = random.Random(seed)
        total_written = 0

        # fam -> spec -> list[rec]
        fam_spec = {}
        for rec in all_annotated_images.values():
            fam = rec["supercategory"]
            spec = rec["category"]
            # Create a dictionary with supercategory (family) as key, and value another dictionary with
            # category (species) as key and list of records as value
            fam_spec.setdefault(fam, {}).setdefault(spec, []).append(rec)

        if max_families is None or max_families <= 0:
            raise ValueError("[CROSS-SPECIES] max_families should be an integer > 0")

        if n_pairs_per_family is None or n_pairs_per_family <= 0:
            raise ValueError("[CROSS-SPECIES] n_pairs_per_family should be an integer > 0")

        # cross-species requires at leats 2 species per family
        eligible_fams = [
            fam for fam, spec_map in fam_spec.items()
            if len(spec_map) >= 2
        ]

        if len(eligible_fams) < max_families:
            raise ValueError(
                f"[CROSS-SPECIES] Failed to select {max_families} families with at least 2 species. "
                f"Available: {len(eligible_fams)}."
            )

        # Sample max_families out of all the eligible (min 2 species) families and re-create the dict
        selected_fams = rng.sample(eligible_fams, k=max_families)
        fam_spec = {fam: fam_spec[fam] for fam in selected_fams}

        for fam in sorted(fam_spec.keys()):
            spec_map = fam_spec[fam]
            specs = sorted(spec_map.keys())

            reservoir = []
            seen = 0  # Valid couples per family

            # Get all species combinations for this family
            for cat_a, cat_b in itertools.combinations(specs, 2):
                A = spec_map[cat_a]
                B = spec_map[cat_b]

                # Generate all possible pairs between the 2 selected species of this iteration
                for a, b in itertools.product(A, B):
                    src_kps, trg_kps = AP10KDataset._match_keypoints(a, b)
                    if len(src_kps) < min_kps:
                        continue

                    pair_out = {
                        "src_imname": a["file_name"],
                        "trg_imname": b["file_name"],
                        "category": fam,
                        "src_kps": src_kps,
                        "trg_kps": trg_kps,
                        "src_bndbox": a["bbox"],
                        "trg_bndbox": b["bbox"],
                    }

                    # Reservoir sampling to prevent OOM error
                    seen += 1
                    # If reservoir is not full, append
                    if len(reservoir) < n_pairs_per_family:
                        reservoir.append(pair_out)
                    else:
                        # Replace a random pair
                        j = rng.randrange(seen)  # 0..seen-1
                        if j < n_pairs_per_family:
                            reservoir[j] = pair_out

            for pair in reservoir:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            total_written += len(reservoir)

        print(f"[CROSS-SPECIES] Total pairs: {total_written}")

    @staticmethod
    def _generate_cross_family_pairs(
            f,
            all_annotated_images,
            min_kps=3,
            n_pairs_per_combination=400,
            seed=42,
            max_families=5
    ):
        """
        Generate and write correspondence pairs between images belonging to different animal families (supercategories).
        Pairs are valid only if they share at least `min_kps` matching keypoints (by name).
        Args:
            f: An open file object where pairs will be written as JSON lines.
                all_annotated_images: A dictionary mapping image_id to annotation records.
            min_kps: Minimum number of matching keypoints required for a pair to be valid.
                Defaults to 3. Pairs with fewer common keypoints are discarded.
            n_pairs_per_combination: Maximum number of pairs to sample for each
                selected family-to-family combination. Defaults to 400.
            seed: Random seed for reproducibility of family selection and pair sampling.
                Defaults to 42.
            max_families: Maximum number of families (supercategories) to include.
                If there are more families with at least 2 species each, a random subset is selected.
                Defaults to 5.
        """
        rng = random.Random(seed)
        total_written = 0

        by_family = {}
        for rec in all_annotated_images.values():
            # Populate the dict with supercategory (family) as key and list of records as value
            by_family.setdefault(rec["supercategory"], []).append(rec)

        # Shuffle the dict in a reproducible way and keep only the first max_families
        families = sorted(by_family.keys())
        rng.shuffle(families)
        families = families[:max_families]
        by_family = {fam: by_family[fam] for fam in families}

        for fam1, fam2 in itertools.combinations(families, 2):
            A = by_family[fam1]
            B = by_family[fam2]

            # Generate all possible species pairs between the 2 selected families of this iteration
            cross_family_pairs = list(itertools.product(A, B))

            possible_pairs = []
            for a, b in cross_family_pairs:
                src_kps, trg_kps = AP10KDataset._match_keypoints(a, b)
                if len(src_kps) < min_kps:
                    continue
                possible_pairs.append((a, b, src_kps, trg_kps))

            # Sample n_pairs_per_combination from all the possible pairs for this family combination
            N = min(n_pairs_per_combination, len(possible_pairs)) # Prevents taking more pair than available
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

    def get_categories(self) -> set:
        categories = set()
        for line in self.ann_files:
            ann = json.loads(line)
            categories.add(ann['category'])
        return categories
