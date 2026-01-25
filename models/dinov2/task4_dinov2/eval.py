import os
import math 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
# NUOVI IMPORT NECESSARI
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.task3_DINOv2_dataset import SPairDataset  
from utils.setup_data import setup_data
from task3_dinov2.soft_argmax_windows import soft_argmax_window

if __name__ == '__main__':

    print("--- 1. Checking Data Availability ---")
    data_root = setup_data() 

    if data_root is None:
        print("CRITICAL ERROR: Data could not be set up. Exiting.")
        exit()

    base_dir = os.path.join(data_root, 'SPair-71k','Spair-71k') 
    pair_ann_path = os.path.join(base_dir, 'PairAnnotation')
    layout_path = os.path.join(base_dir, 'Layout')
    image_path = os.path.join(base_dir, 'JPEGImages')

    print("\n--- 2. Loading Test Dataset ---")
    test_dataset = SPairDataset(
        pair_ann_path, layout_path, image_path, 
        dataset_size='large', datatype='test', pck_alpha=0.5
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 3. Loading Custom DINOv2 from .safetensors ---")

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    print("--> Applying LoRA architecture...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["qkv"], 
        lora_dropout=0.1, # serve per la config
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    checkpoint_path = 'C:/Users/nicol/Documents/PoliTo/AdvancedML/project/AML-polito/task4_dinov2/checkpoints/best_model_run1/adapter_model_run1.safetensors'
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from: {checkpoint_path}")
        
        state_dict = load_file(checkpoint_path)

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Status: {msg}")
        
    else:
        print(f"WARNING: Checkpoint '{checkpoint_path}' not found! Using random initialized LoRA.")

    model.to(device)
    model.eval() # Disattiva Dropout
    print(f"Model loaded on: {device}")

    # Initialize counters
    class_pck_data = {}
    class_pck_image = {}

    with torch.no_grad(): # Disable gradients
        for i, data in enumerate(tqdm(test_dataloader, desc="Evaluation")):
            
            category = data['category'][0]
            if category not in class_pck_data:
                class_pck_data[category] = {
                    'total_keypoints': 0,
                    'correct_kps_0_05': 0,
                    'correct_kps_0_1': 0,
                    'correct_kps_0_2': 0

                }
            if category not in class_pck_image:
                class_pck_image[category] = {
                    'total_image': 0,
                    'image_value_sum_0_05': 0, # Accumulatore per le medie delle singole immagini
                    'image_value_sum_0_1': 0,
                    'image_value_sum_0_2': 0
                }
            img_tot_keypoints = 0
            img_correct_keypoints_0_05 = 0
            img_correct_keypoints_0_1 = 0
            img_correct_keypoints_0_2 = 0

            src_img = data['src_img'].to(device) # torch.Size([1, 3, 518, 518])
            trg_img = data['trg_img'].to(device)
            
            # We pass the PADDED images
            dict_src = model.forward_features(src_img) # Python dictionary. 
            dict_trg = model.forward_features(trg_img)
            
            feats_src = dict_src["x_norm_patchtokens"] # [Batch_Size, Num_Patches, Dimension]
            feats_trg = dict_trg["x_norm_patchtokens"]  
            
            # We keep ORIGINAL dimensions for valid boundary checks
            _, _, H_orig, W_orig = data['src_img'].shape

            patch_size = 14
            w_grid = 37
            h_grid = 37

            kps_list_src = data['src_kps'][0] 
            trg_kps_gt = data['trg_kps'][0] 
            valid_mask = data['valid_mask'][0]
        
            bbox = data['trg_bbox'][0] 

            # Estraiamo i 4 valori scalari per l'immagine corrente (indice batch 0)
            x_min = bbox[0].item()
            y_min = bbox[1].item()
            x_max = bbox[2].item()
            y_max = bbox[3].item()

            w_bbox = x_max - x_min
            h_bbox = y_max - y_min
            # La dimensione di riferimento è il lato massimo della BBox
            max_side = max(w_bbox, h_bbox)
            
            # Calcoliamo le 3 soglie in pixel
            thr_05 = max_side * 0.05
            thr_10 = max_side * 0.10
            thr_20 = max_side * 0.20
            # Get threshold value
            #         
            for n_keypoint, keypoint_src in enumerate(kps_list_src):

                if valid_mask[n_keypoint] == 0:
                    continue

                x_src_val = keypoint_src[0].item()
                y_src_val = keypoint_src[1].item()

                x_pixel_src = int(x_src_val)
                y_pixel_src = int(y_src_val)

                # Grid Clamp
                x_patch_src = min(max(0, x_pixel_src // patch_size), w_grid - 1)
                y_patch_src = min(max(0, y_pixel_src // patch_size), h_grid - 1)

                # 3. INDEX CALCULATION
                patch_index_src = (y_patch_src * w_grid) + x_patch_src

                # Extract Vector
                source_vec = feats_src[0, patch_index_src, :]

                # Cosine Similarity shape [1369]
                similarity_map = torch.cosine_similarity(source_vec, feats_trg[0], dim=-1)

                sim_2d = similarity_map.view(h_grid, w_grid)

                y_col, x_row = soft_argmax_window(sim_map_2d=sim_2d)
                                
                # The logic remains: Coordinate * stride + offset
                x_pred_pixel = x_row * patch_size + (patch_size // 2)
                y_pred_pixel = y_col * patch_size + (patch_size // 2)

                # Ground Truth Check
                gt_x = trg_kps_gt[n_keypoint, 0].item()
                gt_y = trg_kps_gt[n_keypoint, 1].item()

                # Distance & Update
                distance = math.sqrt((x_pred_pixel - gt_x)**2 + (y_pred_pixel - gt_y)**2)

            
                is_correct_05 = distance <= thr_05
                is_correct_10 = distance <= thr_10
                is_correct_20 = distance <= thr_20

                # Update Category Data
                class_pck_data[category]['total_keypoints'] += 1
                if is_correct_05: class_pck_data[category]['correct_kps_0_05'] += 1
                if is_correct_10: class_pck_data[category]['correct_kps_0_1'] += 1
                if is_correct_20: class_pck_data[category]['correct_kps_0_2'] += 1

                # Update Image Data
                img_tot_keypoints += 1
                if is_correct_05: img_correct_keypoints_0_05 += 1
                if is_correct_10: img_correct_keypoints_0_1 += 1
                if is_correct_20: img_correct_keypoints_0_2 += 1
            
            # AGGIORNAMENTO DATI CATEGORIA (PCK PER IMAGE)
            if img_tot_keypoints > 0:
                image_accuracy_0_05 = img_correct_keypoints_0_05 / img_tot_keypoints
                image_accuracy_0_1 = img_correct_keypoints_0_1 / img_tot_keypoints
                image_accuracy_0_2 = img_correct_keypoints_0_2 / img_tot_keypoints

                
                class_pck_image[category]['total_image'] += 1
                class_pck_image[category]['image_value_sum_0_05'] += image_accuracy_0_05
                class_pck_image[category]['image_value_sum_0_1'] += image_accuracy_0_1
                class_pck_image[category]['image_value_sum_0_2'] += image_accuracy_0_2
        


    # ==========================================
    # FINAL REPORTING
    # ==========================================

    # --- 1. PCK PER POINT (Keypoint Accuracy) ---
    print("\n" + "="*50)
    print("PCK PER POINT (Keypoint Level)")
    print("="*50)

    global_kps_total = 0
    global_kps_correct_05 = 0
    global_kps_correct_10 = 0
    global_kps_correct_20 = 0

    print(f"{'Category':<15} | {'PCK@0.05':<10} | {'PCK@0.10':<10} | {'PCK@0.20':<10}")
    print("-" * 55)

    for category, data in class_pck_data.items():
        tot = data['total_keypoints']
        c05 = data['correct_kps_0_05']
        c10 = data['correct_kps_0_1']
        c20 = data['correct_kps_0_2']
        
        # Calculate Class Accuracy
        p05 = (c05 / tot) * 100 if tot > 0 else 0
        p10 = (c10 / tot) * 100 if tot > 0 else 0
        p20 = (c20 / tot) * 100 if tot > 0 else 0
        
        print(f"{category:<15} | {p05:<9.2f}% | {p10:<9.2f}% | {p20:<9.2f}%")
        
        # Add to Global Totals
        global_kps_total += tot
        global_kps_correct_05 += c05
        global_kps_correct_10 += c10
        global_kps_correct_20 += c20

    # Calculate Micro-Averages
    if global_kps_total > 0:
        micro_05 = (global_kps_correct_05 / global_kps_total) * 100
        micro_10 = (global_kps_correct_10 / global_kps_total) * 100
        micro_20 = (global_kps_correct_20 / global_kps_total) * 100

        print("-" * 55)
        print(f"{'OVERALL':<15} | {micro_05:<9.2f}% | {micro_10:<9.2f}% | {micro_20:<9.2f}%")
        print("="*55)
    else:
        print("No keypoints found.")


    # --- 2. PCK PER IMAGE (Image Accuracy) ---
    print("\n" + "="*50)
    print("PCK PER IMAGE (Image Level)")
    print("="*50)

    global_img_total = 0
    global_img_sum_acc_05 = 0
    global_img_sum_acc_10 = 0
    global_img_sum_acc_20 = 0

    print(f"{'Category':<15} | {'PCK@0.05':<10} | {'PCK@0.10':<10} | {'PCK@0.20':<10}")
    print("-" * 55)

    for category, data in class_pck_image.items():
        tot = data['total_image']
        s05 = data['image_value_sum_0_05']
        s10 = data['image_value_sum_0_1']
        s20 = data['image_value_sum_0_2']
        
        # Calculate Class Accuracy (Mean of image accuracies)
        p05 = (s05 / tot) * 100 if tot > 0 else 0
        p10 = (s10 / tot) * 100 if tot > 0 else 0
        p20 = (s20 / tot) * 100 if tot > 0 else 0
        
        print(f"{category:<15} | {p05:<9.2f}% | {p10:<9.2f}% | {p20:<9.2f}%")
        
        # Add to Global Totals
        global_img_total += tot
        global_img_sum_acc_05 += s05
        global_img_sum_acc_10 += s10
        global_img_sum_acc_20 += s20

    # Calculate Micro-Averages
    if global_img_total > 0:
        micro_05 = (global_img_sum_acc_05 / global_img_total) * 100
        micro_10 = (global_img_sum_acc_10 / global_img_total) * 100
        micro_20 = (global_img_sum_acc_20 / global_img_total) * 100

        print("-" * 55)
        print(f"{'OVERALL':<15} | {micro_05:<9.2f}% | {micro_10:<9.2f}% | {micro_20:<9.2f}%")
        print("="*55)



    