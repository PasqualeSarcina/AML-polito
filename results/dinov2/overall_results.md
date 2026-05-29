# Training Test Results

## 1 Training Configuration

| Model         | Scheduler | Batch Size | Learning Rate | Weight Decay | Layers | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :------------ | :-------- | :--------- | :------------ | :----------- | :----- | :--------------- | :----------- | :----------- | :----------- |
| Dinov2 vitb14 | None      | 5          | 1e-4          | 1e-2         | 2      | Keypoint Level   | 50.41        | 66.08        | 76.45        |
| Dinov2 vitb14 | None      | 5          | 1e-4          | 1e-2         | 2      | Image Level      | 48.45        | 63.74        | 74.34        |

## 2 Training Configuration

| Model         | Scheduler | Batch Size | Learning Rate | Weight Decay | Layers | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :------------ | :-------- | :--------- | :------------ | :----------- | :----- | :--------------- | :----------- | :----------- | :----------- |
| Dinov2 vitb14 | Yes       | 5          | 1e-5          | 1e-2         | 2      | Keypoint Level   | 52.07        | 67.34        | 77.52        |
| Dinov2 vitb14 | Yes       | 5          | 1e-5          | 1e-2         | 2      | Image Level      | 50.75        | 65.68        | 75.95        |

## 3 Training Configuration - BEST MODEL WITH WINDOW-SOFT ARGMAX

| Model         | Scheduler | Batch Size | Learning Rate | Weight Decay | Layers | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :------------ | :-------- | :--------- | :------------ | :----------- | :----- | :--------------- | :----------- | :----------- | :----------- |
| Dinov2 vitb14 | Yes       | 5          | 1e-5          | 1e-2         | 1      | Keypoint Level   | 53.73        | 69.69        | 79.72        |
| Dinov2 vitb14 | Yes       | 5          | 1e-5          | 1e-2         | 1      | Image Level      | 52.61        | 68.55        | 78.82        |

## 4 Training Configuration - BEST MODEL WITH HARD ARGMAX

| Model         | Scheduler | Batch Size | Learning Rate | Weight Decay | Layers | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :------------ | :-------- | :--------- | :------------ | :----------- | :----- | :--------------- | :----------- | :----------- | :----------- |
| Dinov2 vitb14 | Yes       | 5          | 1e-5          | 1e-2         | 1      | Keypoint Level   | 50.82        | 66.48        | 78.82        |
| Dinov2 vitb14 | Yes       | 5          | 1e-5          | 1e-2         | 1      | Image Level      | 49.76        | 65.54        | 77.88        |

## 5 Training Configuration

| Model         | Scheduler | Batch Size | Learning Rate | Weight Decay | Layers | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :------------ | :-------- | :--------- | :------------ | :----------- | :----- | :--------------- | :----------- | :----------- | :----------- |
| Dinov2 vitb14 | Yes       | 8          | 1e-5          | 1e-2         | 1      | Keypoint Level   | 52.97        | 68.54        | 78.83        |
| Dinov2 vitb14 | Yes       | 8          | 1e-5          | 1e-2         | 1      | Image Level      | 51.73        | 67.26        | 77.84        |

## 6 Training Configuration (LoRA)

| Model         | Scheduler | Batch Size | Learning Rate | Weight Decay | Dropout | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :------------ | :-------- | :--------- | :------------ | :----------- | :------ | :--------------- | :----------- | :----------- | :----------- |
| Dinov2 vitb14 | Yes       | 5          | 1e-4          | 1e-2         | 0.1     | Keypoint Level   | 37.9         | 56.5         | 71.1         |
| Dinov2 vitb14 | Yes       | 5          | 1e-4          | 1e-2         | 0.1     | Image Level      | 35.4         | 53.6         | 68.3         |
