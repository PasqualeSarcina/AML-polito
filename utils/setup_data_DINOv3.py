import os
from pathlib import Path
import zipfile
import tarfile

def setup_data():
    # Project root (repo root)
    project_root = Path(__file__).resolve().parents[1]

    dataset_dir = project_root / "dataset"
    extract_dir = dataset_dir / "SPair-71k_extracted"

    zip_path = dataset_dir / "SPair-71k.zip"
    tar_path = dataset_dir / "SPair-71k.tar"
    targz_path = dataset_dir / "SPair-71k.tar.gz"
    tgz_path = dataset_dir / "SPair-71k.tgz"

    extract_dir.mkdir(parents=True, exist_ok=True)

    # If already extracted
    if (extract_dir / "SPair-71k").exists():
        print(f"Data already extracted in {extract_dir}")
        return extract_dir

    # ZIP
    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        return extract_dir

    # TAR / TAR.GZ / TGZ
    for tar_file in [targz_path, tgz_path, tar_path]:
        if tar_file.exists():
            print(f"Extracting {tar_file}...")
            with tarfile.open(tar_file, "r:*") as t:
                t.extractall(extract_dir, filter="data")
            return extract_dir

    print(
        "Dataset archive not found.\n"
        "Expected one of:\n"
        "  - dataset/SPair-71k.zip\n"
        "  - dataset/SPair-71k.tar\n"
        "  - dataset/SPair-71k.tar.gz\n"
        "  - dataset/SPair-71k.tgz"
    )
    return None

if __name__ == "__main__":
    setup_data()
