# DINOv3 Filtered Overall Results — SPair-71k

This file reports the selected overall results for the DINOv3 experiments evaluated on SPair-71k.

Common evaluation setup:

- Dataset: SPair-71k
- Backbone: DINOv3 ViT-B/16
- Input resolution: 512 × 512
- Patch size: 16
- Main selection metric: PCK@0.10 at keypoint level

## Overall Results

| Config ID | Task | Role | Model | Batch Size | Accumulation Steps | Learning Rate | Weight Decay | Layers | LoRA r | LoRA alpha | LoRA dropout | LoRA target | Refinement | PCK Keypoint @0.05 (%) | PCK Keypoint @0.10 (%) | PCK Keypoint @0.20 (%) | PCK Image @0.05 (%) | PCK Image @0.10 (%) | PCK Image @0.20 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0-ZS-HARD | Task 1 Zero-shot baseline | DINOv3 zero-shot baseline | DINOv3 ViT-B/16 | - | - | - | - | - | - | - | - | - | Hard argmax | 34.77 | 51.98 | 66.84 | 31.73 | 48.00 | 62.49 |
| C1-T2-HARD | Task 2 Fine-tuning | Best Task 2 model without window refinement | DINOv3 ViT-B/16 | 8 | 1 | 2e-06 | 1e-2 | 3 | - | - | - | - | Hard argmax | 59.33 | 75.25 | 85.65 | 57.23 | 73.32 | 83.89 |
| T2-L1-BS1 | Task 2 Fine-tuning | Selected best lr for layer/batch setting | DINOv3 ViT-B/16 | 1 | 8 | 1e-05 | 1e-2 | 1 | - | - | - | - | WSA r=3, temp=0.05 | 58.45 | 74.79 | 84.50 | 56.91 | 73.13 | 83.07 |
| T2-L1-BS5 | Task 2 Fine-tuning | Selected best lr for layer/batch setting | DINOv3 ViT-B/16 | 5 | 1 | 1e-05 | 1e-2 | 1 | - | - | - | - | WSA r=3, temp=0.05 | 58.66 | 75.32 | 85.00 | 56.92 | 73.48 | 83.46 |
| T2-L1-BS8 | Task 2 Fine-tuning | Selected best lr for layer/batch setting | DINOv3 ViT-B/16 | 8 | 1 | 1e-05 | 1e-2 | 1 | - | - | - | - | WSA r=3, temp=0.05 | 59.67 | 75.41 | 84.98 | 58.01 | 73.65 | 83.46 |
| T2-L2-BS1 | Task 2 Fine-tuning | Selected best lr for layer/batch setting | DINOv3 ViT-B/16 | 1 | 8 | 2e-06 | 1e-2 | 2 | - | - | - | - | WSA r=3, temp=0.05 | 60.22 | 76.43 | 85.76 | 58.48 | 74.65 | 84.34 |
| T2-L2-BS5 | Task 2 Fine-tuning | Selected best lr for layer/batch setting | DINOv3 ViT-B/16 | 5 | 1 | 2e-06 | 1e-2 | 2 | - | - | - | - | WSA r=3, temp=0.05 | 60.56 | 76.89 | 86.16 | 58.59 | 74.85 | 84.50 |
| T2-L2-BS8 | Task 2 Fine-tuning | Selected best lr for layer/batch setting | DINOv3 ViT-B/16 | 8 | 1 | 2e-06 | 1e-2 | 2 | - | - | - | - | WSA r=3, temp=0.05 | 60.72 | 76.92 | 86.12 | 58.79 | 74.92 | 84.49 |
| T2-L3-BS1 | Task 2 Fine-tuning | Selected best lr for layer/batch setting | DINOv3 ViT-B/16 | 1 | 8 | 2e-06 | 1e-2 | 3 | - | - | - | - | WSA r=3, temp=0.05 | 61.03 | 76.97 | 86.10 | 59.21 | 75.06 | 84.52 |
| T2-L3-BS5 | Task 2 Fine-tuning | Selected best lr for layer/batch setting | DINOv3 ViT-B/16 | 5 | 1 | 2e-06 | 1e-2 | 3 | - | - | - | - | WSA r=3, temp=0.05 | 60.24 | 76.76 | 86.13 | 58.28 | 74.59 | 84.32 |
| T2-L3-BS8 | Task 2 Fine-tuning | Best Task 2 model with window refinement | DINOv3 ViT-B/16 | 8 | 1 | 2e-06 | 1e-2 | 3 | - | - | - | - | WSA r=3, temp=0.05 | 61.88 | 77.51 | 86.45 | 59.76 | 75.42 | 84.77 |
| T4-LoRA-BS1 | Task 4 LoRA | LoRA r8 alpha16, selected best lr for batch setting | DINOv3 ViT-B/16 | 1 | 8 | 5e-05 | 1e-2 | - | 8 | 16 | 0.1 | qkv | WSA r=3, temp=0.05 | 55.80 | 71.79 | 82.00 | 53.69 | 69.34 | 79.82 |
| T4-LoRA-BS5 | Task 4 LoRA | LoRA r8 alpha16, selected best lr for batch setting | DINOv3 ViT-B/16 | 5 | 1 | 5e-05 | 1e-2 | - | 8 | 16 | 0.1 | qkv | WSA r=3, temp=0.05 | 56.13 | 71.42 | 81.50 | 54.04 | 68.92 | 79.25 |
| T4-LoRA-BS8 | Task 4 LoRA | LoRA r8 alpha16, selected best lr for batch setting | DINOv3 ViT-B/16 | 8 | 1 | 5e-05 | 1e-2 | - | 8 | 16 | 0.1 | qkv | WSA r=3, temp=0.05 | 56.24 | 71.88 | 81.82 | 54.05 | 69.26 | 79.46 |

