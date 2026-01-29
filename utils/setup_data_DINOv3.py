import os
import zipfile
import tarfile
from pathlib import Path


def _looks_like_spair_root(spair_root: Path) -> bool:
    """True se spair_root contiene la struttura attesa di SPair-71k."""
    return (
        (spair_root / "PairAnnotation").is_dir()
        and (spair_root / "Layout").is_dir()
        and (spair_root / "JPEGImages").is_dir()
    )


def _find_spair_dir(search_root: Path) -> Path | None:
    """
    Cerca una cartella 'SPair-71k' valida dentro search_root.
    Ritorna il path della cartella SPair-71k (non il parent) se trovata.
    """
    # Caso 1: search_root è già SPair-71k
    if _looks_like_spair_root(search_root):
        return search_root

    # Caso 2: search_root/SPair-71k
    candidate = search_root / "SPair-71k"
    if _looks_like_spair_root(candidate):
        return candidate

    # Caso 3: cerca ricorsivamente una cartella che sembra SPair-71k
    # (limitato a depth ragionevole)
    for p in search_root.rglob("SPair-71k"):
        if p.is_dir() and _looks_like_spair_root(p):
            return p

    return None


def setup_data(
    dataset_dir: str | Path | None = None,
    extracted_name: str = "SPair-71k_extracted",
    prefer: tuple[str, ...] = ("folder", "zip", "tar"),
    verbose: bool = True,
) -> Path | None:
    """
    Setup SPair-71k in modo compatibile con:
      - cartella già estratta
      - .zip
      - .tar / .tar.gz / .tgz

    Ritorna: Path alla directory che contiene SPair-71k (extracted_dir),
             così che tu possa fare: base_dir = extracted_dir / "SPair-71k"
    """

    # --- Resolve dataset directory ---
    if dataset_dir is None:
        # Assume: utils/setup_data.py -> project_root/dataset/...
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        dataset_dir = project_root / "dataset"
    else:
        dataset_dir = Path(dataset_dir).expanduser().resolve()

    extracted_dir = dataset_dir / extracted_name
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Se già estratto, usa quello ---
    if "folder" in prefer:
        found = _find_spair_dir(extracted_dir)
        if found is not None:
            if verbose:
                print(f"Data already extracted at: {found}")
            return extracted_dir

        # supporta anche dataset_dir/SPair-71k (utente che ha messo la cartella direttamente)
        found2 = _find_spair_dir(dataset_dir)
        if found2 is not None:
            # se SPair-71k è direttamente in dataset_dir, non forziamo a spostarlo:
            # ritorniamo il parent corretto per mantenere base_dir = data_root/'SPair-71k'
            if verbose:
                print(f"Found extracted dataset at: {found2}")
            return found2.parent

    # --- 2) Trova un archivio disponibile ---
    zip_path = dataset_dir / "SPair-71k.zip"
    tar_candidates = [
        dataset_dir / "SPair-71k.tar",
        dataset_dir / "SPair-71k.tar.gz",
        dataset_dir / "SPair-71k.tgz",
    ]

    archive_to_use: Path | None = None
    archive_kind: str | None = None

    for kind in prefer:
        if kind == "zip" and zip_path.is_file():
            archive_to_use = zip_path
            archive_kind = "zip"
            break
        if kind == "tar":
            for t in tar_candidates:
                if t.is_file():
                    archive_to_use = t
                    archive_kind = "tar"
                    break
            if archive_to_use is not None:
                break

    if archive_to_use is None:
        if verbose:
            print("ERROR: Dataset not found.")
            print("Expected one of:")
            print(f" - extracted folder: {extracted_dir}/SPair-71k (with PairAnnotation/Layout/JPEGImages)")
            print(f" - extracted folder: {dataset_dir}/SPair-71k (with PairAnnotation/Layout/JPEGImages)")
            print(f" - zip: {zip_path}")
            for t in tar_candidates:
                print(f" - tar: {t}")
        return None

    # --- 3) Estrazione ---
    if verbose:
        print(f"Extracting {archive_to_use.name} into: {extracted_dir}")

    try:
        if archive_kind == "zip":
            with zipfile.ZipFile(archive_to_use, "r") as z:
                z.extractall(extracted_dir)
        elif archive_kind == "tar":
            # tarfile gestisce tar e tar.gz/tgz in automatico con mode="r:*"
            with tarfile.open(archive_to_use, mode="r:*") as t:
                t.extractall(extracted_dir)
        else:
            raise RuntimeError(f"Unknown archive kind: {archive_kind}")
    except Exception as e:
        if verbose:
            print(f"ERROR during extraction: {e}")
        return None

    # --- 4) Verifica post-estrazione ---
    found = _find_spair_dir(extracted_dir)
    if found is None:
        if verbose:
            print("ERROR: Extraction finished but SPair-71k structure not found.")
            print(f"Looked inside: {extracted_dir}")
            print("Expected a folder containing: PairAnnotation/, Layout/, JPEGImages/")
        return None

    if verbose:
        print(f"Extraction OK. Found dataset at: {found}")

    # ritorna sempre il parent che contiene SPair-71k (coerente col tuo codice)
    return found.parent


if __name__ == "__main__":
    data_root = setup_data(verbose=True)
    if data_root is None:
        raise SystemExit(1)

    # esempio coerente con i tuoi script:
    base_dir = data_root / "SPair-71k"
    print("base_dir:", base_dir)
    print("has PairAnnotation:", (base_dir / "PairAnnotation").exists())
