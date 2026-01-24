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
- `models/`: Contains implementations of the code
- `data/`: Contains datasets used for training and evaluation. Dataset are automatically downloaded and pre-processed.
  The included dataset are:
    * SPair-71k
    * PF-PASCAL
    * PF-WILLOW
    * AP-10K
- `datasets/`: Dayaset will be stored here after downloading.
- `utils/`: Utility functions for data processing, evaluation, and visualization.
- `eval.py`: Script for evaluating models on semantic correspondence tasks.


## Evaluation
To evaluate a model, use the `eval.py` script. For example, to evaluate the DINOv2 model on the SPair-71k dataset with
default weights, run:

```bash
python eval.py dinov2
```
By default, the script evaluates on the SPair-71k dataset using pre-trained weights.
You can specify different datasets and model checkpoints using command-line arguments. Use the `--help` flag to see all.

The supported models options are: `dinov2`, `dinov3`, `sam`, `dift`.

You can also specify wether to use or not the window soft argmax optimization with the `--win-soft-argmax` flag (active by deafult)
and its size and temperature. To disable it, use the `--no-win-soft-argmax` flag.

When evaluating dift, you can also choose to fuse the dift features with DINOv2 features using the `--fuse-dino` flag.
To use custom dino weights when fusing, use the `--custom-weights` argument.
To improve inference time, you can reduce the number of ensemble steps with the `---ensemble-size` argument (default is 4).








