## Authors

- [Nicolò Sanna](https://github.com/Nicolo99-sys)
- [Alexandra Elena Holota](https://github.com/AlexandraElena-Holota)
- [Pasquale Sarcina](https://github.com/PasqualeSarcina)
- [Antonella Sarcuni](https://github.com/s334047)

# MLDL25 - Semantic correspondence

This repository contains the code and resources for the MLDL25 project on Semantic Correspondence.

## Project Overview

The goal of this project is to evaluate and finetune state-of-the-art models for establishing semantic correspondences
between images.
We focus on the following models:

- DINOv2
- DINOv3
- SAM (Segment Anything Model)
- Diffusion features on UNets

## Repository Structure

- `checkpoints/`: Contains pre-trained and finetuned model checkpoints. Pretrained model are automatically downloaded,
  except for DINOv3 due to licensing restrictions.
- `data/`: Contains datasets used for training and evaluation. Dataset are automatically downloaded and pre-processed.
  The included dataset are:
    * SPair-71k
    * PF-PASCAL
    * PF-WILLOW
    * AP-10K
- `datasets/`: Dataset will be stored here after downloading.
- `models/`: Contains implementations of the code
- `results/`: Contains results on semantic correspondence tasks and training configurations details
- `utils/`: Utility functions for data processing, evaluation, and visualization.
- `eval.py`: Script for evaluating models on semantic correspondence tasks.


## Environment and Setup

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. For DINOv3, follow the additional setup instructions below.

### DINOv3 Setup

DINOv3 requires an additional manual setup because the pretrained weights are not distributed with this repository.

DINOv3 is not fully distributed with this repository because the official pretrained weights must be downloaded manually after accepting the corresponding license terms. The code expects the official DINOv3 repository to be available locally and the pretrained checkpoint to be placed in the project checkpoint directory.

#### 1. Clone the official DINOv3 repository

From the root of this project, clone the official DINOv3 repository inside `third_party/dinov3`:

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
```

Then install the additional dependencies required by DINOv3:

```bash
cd third_party/dinov3
pip install einops timm opencv-python torchmetrics fvcore iopath
cd ../..
```

The expected directory structure is:

```text
third_party/
└── dinov3/
    └── ... official DINOv3 repository files ...
```

#### 2. Download the pretrained checkpoint

The pretrained DINOv3 checkpoint is not included in this repository. It must be downloaded manually from the official DINOv3 release page after accepting the corresponding license terms.

For the experiments in this project, we use DINOv3 ViT-B/16:

```text
dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

Place the checkpoint in:

```text
checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

The expected directory structure is:

```text
checkpoints/
└── dinov3/
    └── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

The code also checks the fallback path below, but the recommended location is `checkpoints/dinov3/`:

```text
checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```
## Dataset Preparation

The datasets are automatically downloaded and pre-processed the first time you run an evaluation or training script. Once downloaded, everything is placed inside the `dataset/` directory.

## Training

Currently, we support finetuning for models like DINOv2 using standard gradient descent or LoRA. The training scripts use command-line arguments for easy hyperparameter tuning.

### Fine-Tuning DINOv2

**Standard Fine-Tuning**
```bash
python models/dinov2/train.py --epochs 5 --lr 1e-4 --w_decay 1e-2 --n_layers 1 --accumulation_steps 5
```

**LoRA Fine-Tuning**
```bash
python models/dinov2/train_LoRA.py --epochs 5 --lr 1e-4 --w_decay 1e-2 --accumulation_steps 8  --lora_r 16 --lora_alpha 32  --lora_dropout 0.1
```

### Fine-Tuning SAM

**Standard Fine-Tuning**
```bash
python models/sam/train.py --epochs 5 --lr 1e-5 --w_decay 1e-2 --n_layers 3 --accumulation_steps 8
```

**LoRA Fine-Tuning**
```bash
python models/sam/train_LoRA.py --epochs 5 --lr 1e-4 --w_decay 1e-2 --accumulation_steps 8 --lora_r 16 --lora_alpha 32  --lora_dropout 0.1 
```

### Fine-Tuning DINOv3

DINOv3 supports both light fine-tuning of the last transformer layers and LoRA fine-tuning.

**Standard Fine-Tuning**

DINOv3 light fine-tuning is implemented in:

```text
models/dinov3/Dinov3Train.py
```

The training script freezes the pretrained backbone and unfreezes only the last `n` transformer blocks. If the model has a final normalization layer, it is also made trainable.

Example command:

```bash
python -m models.dinov3.Dinov3Train \
  --dataset spair-71k \
  --epochs 5 \
  --batch-size 8 \
  --accumulation-steps 1 \
  --lr 2e-6 \
  --weight-decay 1e-2 \
  --n-layers 3 \
  --temperature 0.07 \
  --input-size 512 \
  --patch-size 16 \
  --num-workers 2
```

Useful arguments:

```text
--dataset                 Dataset name: spair-71k, pf-pascal, pf-willow, ap-10k
--epochs                  Number of training epochs
--batch-size              Batch size used by the dataloader
--accumulation-steps      Gradient accumulation steps
--lr                      Learning rate
--weight-decay            AdamW weight decay
--n-layers                Number of last transformer blocks to fine-tune
--temperature             Temperature used by the InfoNCE loss
--input-size              Input image resolution
--patch-size              ViT patch size
--num-workers             Number of dataloader workers
--seed                    Random seed
```

**LoRA Fine-Tuning**

DINOv3 LoRA fine-tuning is implemented in:

```text
models/dinov3/Dinov3LoRA.py
```

The LoRA implementation freezes the pretrained DINOv3 backbone and injects trainable low-rank adapters into selected linear layers. In the experiments of this project, LoRA is applied to the fused attention `qkv` projection.

Example command:

```bash
python -m models.dinov3.Dinov3LoRA \
  --dataset spair-71k \
  --epochs 5 \
  --batch-size 8 \
  --accumulation-steps 1 \
  --lr 5e-5 \
  --weight-decay 1e-2 \
  --temperature 0.07 \
  --input-size 512 \
  --patch-size 16 \
  --num-workers 2 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --lora-target-modules qkv \
  --save-merged-state-dict
```

Useful LoRA arguments:

```text
--lora-r                  LoRA rank
--lora-alpha              LoRA scaling factor
--lora-dropout            Dropout applied inside LoRA adapters
--lora-target-modules     Target linear modules, e.g. qkv
--train-norm              Also train normalization layers
--save-merged-state-dict  Save a standard merged DINOv3 state dict for evaluation
```

When using `--save-merged-state-dict`, the saved checkpoint can be evaluated directly with:

```bash
python eval.py --dataset spair-71k --wsam-win-radius 3 --wsam-temp 0.05 dinov3 --custom-weights checkpoints/dinov3/<merged_lora_checkpoint>.pth
```


## Evaluation

To evaluate a model, use the `eval.py` script. For example, to evaluate the DINOv2 model on the SPair-71k dataset with default weights, run:

```bash
python eval.py dinov2
```

By default, the script evaluates on the SPair-71k dataset using pretrained weights.

You can specify different datasets using the `--dataset` argument. For example, to evaluate SAM on the PF-PASCAL dataset with custom weights, run:

```bash
python eval.py --dataset pf-pascal sam --custom-weights path/to/custom_weights.pth
```

Use the `--help` flag to see all available options.

The supported model options are: `dinov2`, `dinov3`, `sam`, and `dift`.

You can also specify whether to use the window soft-argmax optimization with the command-line arguments `--wsam-win-radius` and `--wsam-temp`. The first argument controls the window radius, while the second controls the soft-argmax temperature. The default temperature is `0.05`.

For example, to evaluate SAM on the SPair-71k dataset with custom weights and window soft-argmax, run:

```bash
python eval.py --wsam-win-radius 3 sam --custom-weights path/to/custom_weights.pth
```

To disable window soft-argmax and use hard argmax matching, set the window radius to `0`:

```bash
python eval.py --wsam-win-radius 0 sam --custom-weights path/to/custom_weights.pth
```

### Evaluating DINOv3

To evaluate the frozen pretrained DINOv3 backbone on SPair-71k, run:

```bash
python eval.py --dataset spair-71k dinov3
```

For the reported DINOv3 experiments, the common evaluation setting uses window soft-argmax with radius `3` and temperature `0.05`:

```bash
python eval.py --dataset spair-71k --wsam-win-radius 3 --wsam-temp 0.05 dinov3
```

To disable window soft-argmax and use hard argmax matching with DINOv3, run:

```bash
python eval.py --dataset spair-71k --wsam-win-radius 0 dinov3
```

Fine-tuned DINOv3 checkpoints can be evaluated with `--custom-weights`:

```bash
python eval.py --dataset spair-71k --wsam-win-radius 3 --wsam-temp 0.05 dinov3 --custom-weights checkpoints/dinov3/<checkpoint_name>.pth
```

Example:

```bash
python eval.py --dataset spair-71k --wsam-win-radius 3 --wsam-temp 0.05 dinov3 --custom-weights checkpoints/dinov3/task2_bs8_acc1_lr2e-06_wd1e-02_nlayers3_best_model_state_dict.pth
```

When loading custom weights, the evaluation script reports missing and unexpected keys. For standard fine-tuned DINOv3 checkpoints saved as state dictionaries, both values should normally be zero.

When evaluating a model with LoRA adaptation, you can use the `--lora` flag. For example, to evaluate SAM on the SPair-71k dataset with custom weights via LoRA, run:

```bash
python eval.py --wsam-win-radius 3 sam --lora path/to/lora_folder
```

When evaluating DIFT, you can also choose to fuse DIFT features with DINOv2 features using the `--fuse-dino` flag. To use custom DINOv2 weights when fusing, use the `--custom-weights` argument.

To improve inference time, you can reduce the number of ensemble steps with the `--ensemble-size` argument. The default value is `4`.









