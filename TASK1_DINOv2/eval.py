import os
import math 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.Task1_DINOv2_dataset import SPairDataset  
from utils.setup_data import setup_data

# --- Helper for Model Inference ---
def pad_to_multiple(x, k=14):
    """
    Pads the image (bottom and right) so H and W are multiples of k (patch size).
    Crucial for DINOv2 to avoid shape mismatch errors.
    """
    h, w = x.shape[-2:]
    new_h = math.ceil(h / k) * k
    new_w = math.ceil(w / k) * k
    pad_bottom, pad_right = new_h - h, new_w - w
    if pad_bottom == 0 and pad_right == 0: return x
    return F.pad(x, (0, pad_right, 0, pad_bottom), value=0)

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

    if not os.path.exists(pair_ann_path):
        print(f"Error: Paths look wrong. Checked inside: {base_dir}")
        print(f"Contents of data root: {os.listdir(data_root)}")
        exit()

    print("\n--- 2. Loading Test Dataset ---")

    test_dataset = SPairDataset(
        pair_ann_path, 
        layout_path, 
        image_path, 
        dataset_size='large', 
        datatype='test'
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)
    print(f"Test Set Loaded: {len(test_dataset)} images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 3. Loading DINOv2 on {device} ---")

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    model.eval()

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
            # Counters specific for THIS image
        img_tot_keypoints = 0
        img_correct_keypoints_0_05 = 0
        img_correct_keypoints_0_1 = 0
        img_correct_keypoints_0_2 = 0

        src_img = data['src_img'].to(device) # torch.Size([1, 3, 333, 500])
       
        trg_img = data['trg_img'].to(device)

        # --- FIX: APPLY PADDING ---
        # Ensure dimensions are multiples of 14 to avoid AssertionError
        src_img_padded = pad_to_multiple(src_img, 14)
        trg_img_padded = pad_to_multiple(trg_img, 14)

        
        # We pass the PADDED images
        dict_src = model.forward_features(src_img_padded) # Python dictionary. 
        dict_trg = model.forward_features(trg_img_padded)
        
        feats_src = dict_src["x_norm_patchtokens"] # [Batch_Size, Num_Patches, Dimension]
        feats_trg = dict_trg["x_norm_patchtokens"]
        
        # --- IMPORTANT: GRID CALCULATION ---
        # We must use PADDED dimensions for the grid, otherwise indices will drift
        _, _, H_padded, W_padded = src_img_padded.shape 
        
        # We keep ORIGINAL dimensions for valid boundary checks
        _, _, H_orig, W_orig = data['src_img'].shape

        patch_size = 14
        w_grid = W_padded // patch_size 
        h_grid = H_padded // patch_size

        kps_list_src = data['src_kps'][0] 
        trg_kps_gt = data['trg_kps'][0] 
        
        bbox_list = data['trg_bbox'] 

        # Estraiamo i 4 valori scalari per l'immagine corrente (indice batch 0)
        x_min = bbox_list[0][0].item()
        y_min = bbox_list[1][0].item()
        x_max = bbox_list[2][0].item()
        y_max = bbox_list[3][0].item()

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

            x_src_val = keypoint_src[0].item()
            y_src_val = keypoint_src[1].item()

            # NaN Check
            if math.isnan(x_src_val) or math.isnan(y_src_val):
                continue
            
            x_pixel_src = int(x_src_val)
            y_pixel_src = int(y_src_val)

            # Boundary Check on ORIGINAL image (ignore points in padded area if any)
            if not (0 <= x_pixel_src < W_orig and 0 <= y_pixel_src < H_orig):
                continue

            # Grid Clamp
            x_patch_src = min(x_pixel_src // patch_size, w_grid - 1)
            y_patch_src = min(y_pixel_src // patch_size, h_grid - 1)

            # 3. INDEX CALCULATION
            patch_index_src = (y_patch_src * w_grid) + x_patch_src

            # Extract Vector
            source_vec = feats_src[0, patch_index_src, :]

            # Cosine Similarity
            similarity_map = torch.cosine_similarity(source_vec, feats_trg[0], dim=-1)

            # Prediction
            patch_idx_spatial = torch.argmax(similarity_map).item()

            # Convert Index -> Grid -> Pixel
            x_col_pred = patch_idx_spatial % w_grid
            y_row_pred = patch_idx_spatial // w_grid

            x_pred_pixel = x_col_pred * patch_size + (patch_size // 2)
            y_pred_pixel = y_row_pred * patch_size + (patch_size // 2)

            # Ground Truth Check
            gt_x = trg_kps_gt[n_keypoint, 0].item()
            gt_y = trg_kps_gt[n_keypoint, 1].item()

            if math.isnan(gt_x) or math.isnan(gt_y):
                continue
            if not (0 <= gt_x < W_orig and 0 <= gt_y < H_orig):
                continue

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