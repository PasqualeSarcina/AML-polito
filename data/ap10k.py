import json
import os
import numpy as np
import itertools

from data.dataset import CorrespondenceDataset

def load_jsonl_as_dict(path):
    all_annotated_images = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            all_annotated_images[rec["image_id"]] = rec
    return all_annotated_images

def load_data(*file_paths):
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

def remove_duplicate_annotations(data):
    """Remove duplicate annotations from the data based on image_id."""
    unique_image_ids = set()
    new_annotations = []
    for annotation in data['annotations']:
        if annotation['image_id'] not in unique_image_ids:
            unique_image_ids.add(annotation['image_id'])
            new_annotations.append(annotation)
    data['annotations'] = new_annotations
    return data


def match_keypoints(rec_a, rec_b):
    a_map = {kp["name"]: (kp["x"], kp["y"]) for kp in rec_a["keypoints"]}
    b_map = {kp["name"]: (kp["x"], kp["y"]) for kp in rec_b["keypoints"]}
    common = a_map.keys() & b_map.keys()
    src_kps = [a_map[n] for n in common]  # lista di tuple (x, y)
    trg_kps = [b_map[n] for n in common]
    return src_kps, trg_kps

def generate_intra_species_pairs(f, all_annotated_images, min_kps=3):
    by_species = {}
    for rec in all_annotated_images.values():
        by_species.setdefault(rec["category"], []).append(rec)

    for species, recs in by_species.items():
        for a, b in itertools.combinations(recs, 2):
            src_kps, trg_kps = match_keypoints(a, b)
            if len(src_kps) < min_kps:
                continue

            pair = {
                "src_imname": a["file_name"],
                "trg_imname": b["file_name"],
                "category": species,                      # name
                "src_kps": src_kps,
                "trg_kps": trg_kps,
                "src_bndbox": a["bbox"],
                "trg_bndbox": b["bbox"]
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

def generate_cross_species_pairs(f, all_annotated_images, min_kps=3):
    fam_spec = {}
    for rec in all_annotated_images.values():
        fam = rec["supercategory"]
        spec = rec["category"]
        fam_spec.setdefault(fam, {}).setdefault(spec, []).append(rec)

    for fam, spec_map in fam_spec.items():
        specs = sorted(spec_map.keys())
        for s1, s2 in itertools.combinations(specs, 2):
            cat_label = "-".join(sorted([s1, s2]))  # name-name
            for a, b in itertools.product(spec_map[s1], spec_map[s2]):
                src_kps, trg_kps = match_keypoints(a, b)
                if len(src_kps) < min_kps:
                    continue
                pair = {
                    "src_imname": a["file_name"],
                    "trg_imname": b["file_name"],
                    "category": cat_label,                # name-name
                    "src_kps": src_kps,
                    "trg_kps": trg_kps,
                    "src_bndbox": a["bbox"],
                    "trg_bndbox": b["bbox"]
                }
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

def generate_cross_family_pairs(f, all_annotated_images, min_kps=3):
    by_family = {}
    for rec in all_annotated_images.values():
        by_family.setdefault(rec["supercategory"], []).append(rec)

    for f1, f2 in itertools.combinations(by_family, 2):
        fam_label = "-".join(sorted([f1, f2]))  # supercat-supercat
        for a, b in itertools.product(by_family[f1], by_family[f2]):
            src_kps, trg_kps = match_keypoints(a, b)
            if len(src_kps) < min_kps:
                continue
            pair = {
                "src_imname": a["file_name"],
                "trg_imname": b["file_name"],
                "category": fam_label,                   # supercategory-supercategory
                "src_kps": src_kps,
                "trg_kps": trg_kps,
                "src_bndbox": a["bbox"],
                "trg_bndbox": b["bbox"]
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


class AP10KDataset(CorrespondenceDataset):
    def __init__(self, dataset_dir: str, datatype: str, transform=None, min_kps=3):
        self.ap10kdir = os.path.join(dataset_dir, 'ap-10k')
        super().__init__(dataset='ap10k', datatype=datatype, transform=transform)
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
            data = load_data(split1, split2, split3)
            data = remove_duplicate_annotations(data)

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
            all_annotated_images = load_jsonl_as_dict(merged_json)

            with open(processed_json, "w", encoding="utf-8") as f:
                generate_intra_species_pairs(f, all_annotated_images)

                print("Intra-species pairs generated.")
                generate_cross_species_pairs(f, all_annotated_images)
                print("Cross-species pairs generated.")
                generate_cross_family_pairs(f, all_annotated_images)
                print("Cross-family pairs generated.")

            print(f"Processed annotation file created: {processed_json}")

        else:
            print(f"Processed annotation file already exists: {processed_json}")

        # load processed pairs
        with open(processed_json, "r", encoding="utf-8") as f:
            self.ann_files = f.read().splitlines()

    def load_annotation(self, idx):
        ann = json.loads(self.ann_files[idx])
        ann["pair_id"] = idx
        return ann

    def build_image_path(self, imname, category=None):
        return os.path.join(self.ap10kdir, "data", imname)









