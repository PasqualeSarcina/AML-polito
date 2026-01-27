import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TASK_1_SAM.eval import evaluate_pck
from data.dataset import SPairDataset
from utils.common import download_sam_model

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_root = 'dataset/SPair-71k'
    checkpoint_dir = 'checkpoints'
    model_name = 'lora_adapter_final' # Il modello da testare

    lora_path = os.path.join(checkpoint_dir, model_name)

    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    test_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    base_ckpt = download_sam_model(checkpoint_dir)
    sam = sam_model_registry["vit_b"](checkpoint=base_ckpt)

    # Caricamento pesi custom se esistono
    if os.path.exists(lora_path):
        print(f"Caricamento adattatori LoRA da: {lora_path}")
        sam.image_encoder = PeftModel.from_pretrained(
            sam.image_encoder, 
            lora_path
        )
    else:
        print(f"ATTENZIONE: Cartella LoRA non trovata in {lora_path}. Procedo con SAM base.")
        
    sam.to(device)
    evaluate_pck(sam, test_dataloader, device)