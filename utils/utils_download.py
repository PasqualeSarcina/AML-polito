from pathlib import Path

import requests
from tqdm import tqdm


def download(url: str, out_path: Path, chunk_size: int = 1024 * 1024):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(out_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"Downloading {out_path.name}"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))