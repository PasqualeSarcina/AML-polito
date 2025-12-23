#PRIMA CELLA
import os
import requests
from tqdm import tqdm

def download_sam_model(models_dir):
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    checkpoint_name = "sam_vit_b_01ec64.pth"
    checkpoint_path = os.path.join(models_dir, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        print(f"⬇️ Scarico il modello SAM ViT-B in: {checkpoint_path}")

        response = requests.get(checkpoint_url, stream=True)
        total = int(response.headers.get("content-length", 0))

        with open(checkpoint_path, "wb") as file, tqdm(
            total=total, unit="iB", unit_scale=True, unit_divisor=1024, desc="Download SAM ViT-B"
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print("✅ Download completato!")
    else:
        print(f"✅ Modello base già presente: {checkpoint_path}")
    
    return checkpoint_path