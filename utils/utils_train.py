import os
import torch
import random
import numpy as np

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_single(batch_list):
    return batch_list[0]

def collate_batch(batch_list):
    batch_size = len(batch_list)

    batch = {}

    batch["src_img"] = torch.stack([b["src_img"] for b in batch_list], dim=0)
    batch["trg_img"] = torch.stack([b["trg_img"] for b in batch_list], dim=0)

    batch["src_bndbox"] = torch.stack(
    [torch.as_tensor(b["src_bndbox"], dtype=torch.float32) for b in batch_list],
    dim=0,
    )

    batch["trg_bndbox"] = torch.stack(
        [torch.as_tensor(b["trg_bndbox"], dtype=torch.float32) for b in batch_list],
        dim=0,
    )

    max_k = max(b["src_kps"].shape[0] for b in batch_list)

    src_kps = torch.zeros(batch_size, max_k, 2)
    trg_kps = torch.zeros(batch_size, max_k, 2)
    valid_mask = torch.zeros(batch_size, max_k, dtype=torch.bool)

    for i, b in enumerate(batch_list):
        k = b["src_kps"].shape[0]

        src_kps[i, :k] = b["src_kps"]
        trg_kps[i, :k] = b["trg_kps"]
        valid_mask[i, :k] = True

    batch["src_kps"] = src_kps
    batch["trg_kps"] = trg_kps
    batch["valid_mask"] = valid_mask

    return batch