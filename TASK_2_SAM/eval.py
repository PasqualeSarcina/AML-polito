import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm # Per la barra di caricamento
import gc
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import SPairDataset
from utils.geometry import extract_features, compute_correspondence
from segment_anything import sam_model_registry
from utils.common import download_sam_model

@torch.no_grad()
def evaluate_pck(model, dataloader, device, alpha=0.1):
    model.eval()
    
    #PULIZIA MEMORIA
    torch.cuda.empty_cache()
    gc.collect()

    #ACCUMULATORI GLOBALI (per keypoint)
    total_kps_global = 0
    correct_005_global = 0
    correct_010_global = 0
    correct_020_global = 0

    #ACCUMULATORI PER IMMAGINE (per calcolare PCK medio per immagine)
    #liste che conterranno il PCK di ogni singola coppia
    pck_005_per_img = []
    pck_010_per_img = []
    pck_020_per_img = []

    #ACCUMULATORI PER CATEGORIA
    category_stats = defaultdict(lambda: {'total': 0, 'c_05': 0, 'c_10': 0, 'c_20': 0})

    print(f"Inizio valutazione su {len(dataloader)} coppie...")

    error_count = 0

    for i, batch in enumerate(tqdm(dataloader)):
        src_img = batch['src_img'].to(device)
        trg_img = batch['trg_img'].to(device)
        src_kps = batch['src_kps'][0] 
        trg_kps = batch['trg_kps'][0].to(device) 
        pck_threshold_base = batch['pck_threshold'][0].item()

        category = batch['category'][0]

        img_H, img_W = src_img.shape[2], src_img.shape[3]

        kps_mask = batch['kps_valid'][0].to(device)  #vettore di booleani [True, True, False, ...]

        # 1. Estrai Features
        src_feats = extract_features(model, src_img, model_type='sam')
        trg_feats = extract_features(model, trg_img, model_type='sam')

        # 2. Calcola Corrispondenze
        pred_kps = compute_correspondence(src_feats, trg_feats, src_kps, (img_H, img_W), softmax_flag=False)
        pred_kps = pred_kps.to(device) 
        pred_kps_valid = pred_kps[kps_mask]
        trg_kps_valid = trg_kps[kps_mask]

        if len(trg_kps_valid) == 0:
            continue
        # 3. Valuta (PCK)
        # Calcola distanza Euclidea tra predizione e ground truth
        l2_dist = torch.norm(pred_kps_valid - trg_kps_valid, dim=1)

        # Numero keypoints in questa immagine
        n_kps = len(l2_dist)
        
        if n_kps > 0: # Skip se non ci sono keypoint (raro ma possibile)
            # 4. Calcolo Corretti per diverse soglie
            thr_01 = pck_threshold_base
            thr_05 = pck_threshold_base * 0.5
            thr_20 = pck_threshold_base * 2.0

            # Bool tensors dei corretti
            corr_01 = (l2_dist <= thr_01).sum().item()
            corr_05 = (l2_dist <= thr_05).sum().item()
            corr_20 = (l2_dist <= thr_20).sum().item()

            # --- AGGIORNAMENTO GLOBALE (Per Keypoint) ---
            correct_010_global += corr_01
            correct_005_global += corr_05
            correct_020_global += corr_20
            total_kps_global += n_kps

            # --- AGGIORNAMENTO PER IMMAGINE (Per Image) ---
            # Calcoliamo la % per questa specifica immagine e la aggiungiamo alla lista
            pck_010_per_img.append(corr_01 / n_kps)
            pck_005_per_img.append(corr_05 / n_kps)
            pck_020_per_img.append(corr_20 / n_kps)

            # --- AGGIORNAMENTO PER CATEGORIA ---
            category_stats[category]['total'] += n_kps
            category_stats[category]['c_10'] += corr_01
            category_stats[category]['c_05'] += corr_05
            category_stats[category]['c_20'] += corr_20

        #PULIZIA MEMORIA
        del src_img, trg_img, src_feats, trg_feats, pred_kps, l2_dist
        
    if total_kps_global == 0:
        print("Nessun keypoint trovato nel dataset!")
        return
    
    # --- CALCOLO RISULTATI FINALI ---
    
    # 1. Per Keypoint (Total Correct / Total Kps)
    res_kps_005 = (correct_005_global / total_kps_global) * 100
    res_kps_010 = (correct_010_global / total_kps_global) * 100
    res_kps_020 = (correct_020_global / total_kps_global) * 100

    # 2. Per Image (Media delle % di ogni immagine)
    res_img_005 = np.mean(pck_005_per_img) * 100
    res_img_010 = np.mean(pck_010_per_img) * 100
    res_img_020 = np.mean(pck_020_per_img) * 100

    print("\n" + "="*50)
    print(f"📊 REPORT VALUTAZIONE: SAM ViT-B su SPair-71k")
    print("="*50)
    
    print(f"{'METRICA':<15} | {'PER KEYPOINT':<15} | {'PER IMAGE (Mean)':<15}")
    print("-" * 50)
    print(f"{'PCK@0.05':<15} | {res_kps_005:6.2f}%         | {res_img_005:6.2f}%")
    print(f"{'PCK@0.10':<15} | {res_kps_010:6.2f}%         | {res_img_010:6.2f}%")
    print(f"{'PCK@0.20':<15} | {res_kps_020:6.2f}%         | {res_img_020:6.2f}%")
    print("="*50)
    
    print("\n" + "="*75)
    print(f"📂 DETTAGLIO PER CATEGORIA")
    print("="*75)
    # Intestazione con tutte e tre le metriche
    print(f"{'CATEGORIA':<15} | {'PCK@0.10':<10} | {'PCK@0.05':<10} | {'PCK@0.20':<10} | {'# KPS':<6}")
    print("-" * 75)

    for cat in sorted(category_stats.keys()): #ordine alfabetico delle categorie
        stats = category_stats[cat]
        tot = stats['total']
        if tot > 0:
            pck_10 = (stats['c_10'] / tot) * 100
            pck_05 = (stats['c_05'] / tot) * 100
            pck_20 = (stats['c_20'] / tot) * 100

            print(f"{cat:<15} | {pck_10:6.2f}%    | {pck_05:6.2f}%    | {pck_20:6.2f}%    | {tot:<6}")
        else:
            print(f"{cat:<15} |   N/A      |   N/A      |   N/A      | 0")
    print("="*75)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_root = 'dataset/SPair-71k'
    checkpoint_dir = 'checkpoints'
    model_name = 'sam_tuned_best.pth' # Il modello da testare

    pair_ann_path = os.path.join(dataset_root, 'PairAnnotation')
    layout_path = os.path.join(dataset_root, 'Layout')
    image_path = os.path.join(dataset_root, 'JPEGImages')
    
    test_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size='large', pck_alpha=0.1, datatype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    base_ckpt = download_sam_model(checkpoint_dir)
    sam = sam_model_registry["vit_b"](checkpoint=base_ckpt)

    # Caricamento pesi custom se esistono
    tuned_path = os.path.join(checkpoint_dir, model_name)
    if os.path.exists(tuned_path):
        print(f"Caricamento pesi custom: {tuned_path}")
        sam.image_encoder.load_state_dict(torch.load(tuned_path, map_location=device), strict=True)
    else:
        print(f"ERRORE: Il file {tuned_path} non esiste.")

    sam.to(device)
    
    evaluate_pck(sam, test_dataloader, device)
    
