from pathlib import Path
from collections import OrderedDict

import torch


class PreComputedFeaturemaps:
    def __init__(self, save_dir: Path, cache_size: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache_size = cache_size

        self.current_cat = None
        self.output_dict = {}
        self.cat_cache = OrderedDict()

    def __enter__(self):
        return self

    def save_featuremaps(self, emb: torch.Tensor, category: str, imname: str):
        category = str(category)
        imname = str(imname)

        if self.current_cat is None:
            self.current_cat = category

        if category != self.current_cat:
            # Save previous category
            if len(self.output_dict) > 0:
                torch.save(self.output_dict, self.save_dir / f"{self.current_cat}.pth")
            self.output_dict = {}
            self.current_cat = category

        self.output_dict[imname] = emb.detach().cpu()

    def flush(self):
        # Save last category
        if self.current_cat is not None and len(self.output_dict) > 0:
            torch.save(self.output_dict, self.save_dir / f"{self.current_cat}.pth")
            self.output_dict = {}

    def load_featuremaps(self, category: str, imname: str, device: torch.device) -> torch.Tensor:
        # cache hit
        if category in self.cat_cache:
            self.cat_cache.move_to_end(category)
            return self.cat_cache[category][imname].to(device)

        # cache miss
        pth = torch.load(self.save_dir / f"{category}.pth", map_location="cpu")
        self.cat_cache[category] = pth
        self.cat_cache.move_to_end(category)

        # evict LRU
        while len(self.cat_cache) > self.cache_size:
            self.cat_cache.popitem(last=False)

        return pth[imname].to(device)

    def __exit__(self, exc_type, exc, tb):
        self.flush()  # salva l’ultima categoria
        return False  # non “nasconde” eventuali errori
