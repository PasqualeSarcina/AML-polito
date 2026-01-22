import os
import tarfile
import zipfile

import requests
from tqdm import tqdm
import tempfile
from pathlib import Path


def _download(url: str, out_path: Path, chunk_size: int = 1024 * 1024):
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


def _download_drive_file(file_id: str, out_path: Path, chunk_size: int = 1024 * 1024):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"

    # prima richiesta
    r = session.get(url, params={"id": file_id}, stream=True)
    r.raise_for_status()

    # cerca token di conferma nei cookie
    confirm = None
    for k, v in session.cookies.items():
        if k.startswith("download_warning"):
            confirm = v
            break

    # seconda richiesta con conferma, se necessaria
    if confirm:
        r = session.get(url, params={"id": file_id, "confirm": confirm}, stream=True)
        r.raise_for_status()

    with open(out_path, "wb") as f, tqdm(
            total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True, desc=f"Downloading {out_path.name}"
    ) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def _extract_zip(zip_path: Path, dest_dir: Path, chunk_size: int = 1024 * 1024):
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        infos = z.infolist()
        total_bytes = sum(i.file_size for i in infos)

        with tqdm(total=total_bytes, desc="Extracting", unit="B", unit_scale=True) as pbar:
            for info in infos:
                name = info.filename

                #if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                #    continue

                out_path = dest_dir / name

                with z.open(info, "r") as src, open(out_path, "wb") as dst:
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        dst.write(chunk)
                        pbar.update(len(chunk))


def download_spair(dataset_folder_path):
    print("Downloading SPair-71k dataset...")
    dest_dir = Path(dataset_folder_path)
    url = "https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"
    with tempfile.TemporaryDirectory(prefix="dl_") as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_zip = tmpdir / "SPair-71k.tar.gz"
        _download(url, tmp_zip)
        dest_dir = dest_dir.resolve()
        with tarfile.open(tmp_zip, "r:gz") as tar:
            members = tar.getmembers()

            with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
                for m in members:
                    try:
                        tar.extract(m, path=dest_dir, filter="data")  # Py>=3.12
                    except TypeError:
                        tar.extract(m, path=dest_dir)
                    pbar.update(1)


def download_pfpascal(dataset_folder_path):
    print("Downloading PF-Pascal dataset...")
    dest_dir = Path(dataset_folder_path) / "pf-pascal"
    dest_dir.mkdir(parents=True, exist_ok=True)
    data_url = "https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip"
    pairs_url = "https://www.robots.ox.ac.uk/~xinghui/sd4match/pf-pascal_image_pairs.zip"
    with tempfile.TemporaryDirectory(prefix="dl_") as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_zip_data = tmpdir / "PF-Pascal-data.zip"
        tmp_zip_pairs = tmpdir / "PF-Pascal-pairs.zip"
        _download(data_url, tmp_zip_data)
        _download(pairs_url, tmp_zip_pairs)
        _extract_zip(tmp_zip_data, dest_dir)
        _extract_zip(tmp_zip_pairs, dest_dir)


def download_pfwillow(dataset_folder_path):
    print("Downloading PF-Willow dataset...")
    dest_dir = Path(dataset_folder_path) / "pf-willow"
    dest_dir.mkdir(parents=True, exist_ok=True)
    data_url = "https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip"
    pairs_url = "https://www.robots.ox.ac.uk/~xinghui/sd4match/test_pairs.csv"
    with tempfile.TemporaryDirectory(prefix="dl_") as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_zip_data = tmpdir / "PF-dataset.zip"
        pairs = dest_dir / "test_pairs.csv"
        _download(data_url, tmp_zip_data)
        _download(pairs_url, pairs)
        _extract_zip(tmp_zip_data, dest_dir)


def download_ap10k(dataset_folder_path):
    print("Downloading AP-10K dataset...")
    dest_dir = Path(dataset_folder_path) / "ap-10k"
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_id = "1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad"  # Example file ID
    with tempfile.TemporaryDirectory(prefix="dl_") as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_zip = tmpdir / "ap-10k.zip"
        _download_drive_file(file_id, tmp_zip)
        _extract_zip(tmp_zip, dest_dir)

