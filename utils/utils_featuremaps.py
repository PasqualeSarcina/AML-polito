from pathlib import Path
from collections import OrderedDict
import gc

import torch

def save_featuremap(featmap: torch.Tensor, imname: str, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(featmap.detach().cpu(), path / f"{imname}.pt")

def load_featuremap(imname: str, path: Path, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    return torch.load(path / f"{imname}.pt", map_location=device)