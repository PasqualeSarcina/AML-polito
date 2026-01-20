from pathlib import Path
from collections import OrderedDict
import gc

import torch

def save_featuremap(featmap, imname: str, path: Path):
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(featmap, dict):
        # salva un dict: {layer_idx: tensor}
        out = {k: v.detach().cpu() for k, v in featmap.items()}
        torch.save(out, path / f"{imname}.pt")
    else:
        torch.save(featmap.detach().cpu(), path / f"{imname}.pt")


def load_featuremap(imname: str, path: Path, device: torch.device = torch.device("cpu")):
    obj = torch.load(path / f"{imname}.pt", map_location="cpu")

    if isinstance(obj, dict):
        return {k: v.to(device) for k, v in obj.items()}
    return obj.to(device)