#PRIMA CELLA
import os
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def plot_training_results(train_losses, val_losses, save_dir, n_layers, run_id):
    plt.figure(figsize=(10, 6))
    
    # Crea asse X basato sul numero di epoche
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Training e Validation
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    
    # Decorazioni grafico
    plt.title(f'SAM Fine-tuning: {n_layers} Layers (Run: {run_id})')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvataggio su file
    filename = f"plot_{n_layers}layers_{run_id}.png"
    plt.savefig(os.path.join(save_dir, filename))
    print(f"📈 Grafico salvato come: {filename}")
    plt.close()