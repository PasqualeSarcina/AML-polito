import os
import requests
import tarfile
from tqdm import tqdm

def download_spair71k(target_dir='dataset'):
    # URL ufficiale del dataset SPair-71k
    url = "http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"
    
    # Percorso finale
    dataset_path = os.path.join(target_dir, 'SPair-71k')
    tar_path = os.path.join(target_dir, 'SPair-71k.tar.gz')
    
    if os.path.exists(dataset_path):
        print(f"✅ Dataset già presente in: {dataset_path}")
        return

    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Download
    print(f"⬇️ Scarico SPair-71k da {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    
    with open(tar_path, "wb") as file, tqdm(
        total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
    # 2. Estrazione
    print("📦 Estrazione in corso... (può richiedere qualche minuto)")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
        
    # 3. Pulizia (rimuovi il file .tar.gz per risparmiare spazio)
    os.remove(tar_path)
    print("✅ Download ed estrazione completati!")

if __name__ == "__main__":
    download_spair71k()