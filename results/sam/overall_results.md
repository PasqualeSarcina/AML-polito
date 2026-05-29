# Training Test Results

## 1 Training Configuration

| Model      | Scheduler | Batch Size | Learning Rate | Weight Decay | Layers | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :--------- | :-------- | :--------- | :------------ | :----------- | :----- | :--------------- | :----------- | :----------- | :----------- |
| SAM vitb16 | Yes       | 8          | 1e-5          | 1e-2         | 1      | Keypoint Level   | 32.41        | 42.68        | 55.40        |
| SAM vitb16 | Yes       | 8          | 1e-5          | 1e-2         | 1      | Image Level      | 30.73        | 40.60        | 53.13        |

## 2 Training Configuration

| Model      | Scheduler | Batch Size | Learning Rate | Weight Decay | Layers | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :--------- | :-------- | :--------- | :------------ | :----------- | :----- | :--------------- | :----------- | :----------- | :----------- |
| SAM vitb16 | Yes       | 8          | 1e-5          | 1e-2         | 2      | Keypoint Level   | 33.13        | 43.48        | 56.20        |
| SAM vitb16 | Yes       | 8          | 1e-5          | 1e-2         | 2      | Image Level      | 31.43        | 41.40        | 53.91        |

## 3 Training Configuration - BEST MODEL WITH WINDOW SOFT-ARGMAX

| Model      | Scheduler | Batch Size | Learning Rate | Weight Decay | Layers | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :--------- | :-------- | :--------- | :------------ | :----------- | :----- | :--------------- | :----------- | :----------- | :----------- |
| SAM vitb16 | Yes       | 8          | 1e-5          | 1e-2         | 3      | Keypoint Level   | 33.60        | 44.00        | 56.62        |
| SAM vitb16 | Yes       | 8          | 1e-5          | 1e-2         | 3      | Image Level      | 31.88        | 41.82        | 54.14        |

## 4 Training Configuration - BEST MODEL WITH HARD ARGMAX

| Model      | Scheduler | Batch Size | Learning Rate | Weight Decay | Layers | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :--------- | :-------- | :--------- | :------------ | :----------- | :----- | :--------------- | :----------- | :----------- | :----------- |
| SAM vitb16 | Yes       | 8          | 1e-5          | 1e-2         | 3      | Keypoint Level   | 32.34        | 42.82        | 56.25        |
| SAM vitb16 | Yes       | 8          | 1e-5          | 1e-2         | 3      | Image Level      | 30.68        | 40.71        | 53.76        |

## 5 Training Configuration 

| Model      | Scheduler | Batch Size | Learning Rate | Weight Decay | Layers | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :--------- | :-------- | :--------- | :------------ | :----------- | :----- | :--------------- | :----------- | :----------- | :----------- |
| SAM vitb16 | Yes       | 8          | 1e-6          | 1e-2         | 3      | Keypoint Level   | 27.76        | 38.93        | 52.55        |
| SAM vitb16 | Yes       | 8          | 1e-6          | 1e-2         | 3      | Image Level      | 25.70        | 36.36        | 49.77        |

## 6 Training Configuration (LoRA) - BEST MODEL WITH WINDOW SOFT-ARGMAX

| Model      | Scheduler | Batch Size | Learning Rate | Weight Decay | Dropout | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :--------- | :-------- | :--------- | :------------ | :----------- | :------ | :--------------- | :----------- | :----------- | :----------- |
| SAM vitb16 | Yes       | 8          | 1e-4          | 1e-2         | 0.1     | Keypoint Level   | 29.28        | 40.65        | 54.38        |
| SAM vitb16 | Yes       | 8          | 1e-4          | 1e-2         | 0.1     | Image Level      | 27.15        | 37.89        | 51.34        |

## 7 Training Configuration (LoRA)

| Model      | Scheduler | Batch Size | Learning Rate | Weight Decay | Dropout | Evaluation Level | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.20 (%) |
| :--------- | :-------- | :--------- | :------------ | :----------- | :------ | :--------------- | :----------- | :----------- | :----------- |
| SAM vitb16 | Yes       | 8          | 1e-5          | 1e-2         | 0.1     | Keypoint Level   | 23.46        | 34.81        | 48.70        |
| SAM vitb16 | Yes       | 8          | 1e-5          | 1e-2         | 0.1     | Image Level      | 21.30        | 32.03        | 45.58        |
