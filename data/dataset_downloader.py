import os
import tarfile
import zipfile

import requests
from tqdm import tqdm
import tempfile
from pathlib import Path

from utils.utils_download import download


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

                # if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
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
        download(url, tmp_zip)
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
        download(data_url, tmp_zip_data)
        download(pairs_url, tmp_zip_pairs)
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
        download(data_url, tmp_zip_data)
        download(pairs_url, pairs)
        _extract_zip(tmp_zip_data, dest_dir)


def download_ap10k(dataset_folder_path):
    print("Downloading AP-10K dataset...")
    dest_dir = Path(dataset_folder_path) / "ap-10k"
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dl_") as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_zip = tmpdir / "ap-10k.zip"
        print("downloading in ", tmp_zip)
        # _download_drive_file(file_id, tmp_zip)
        download(
            "https://drive.usercontent.google.com/download?id=1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad&export=download&authuser=0&confirm=t&",
            tmp_zip)
        _extract_zip(tmp_zip, dest_dir)
