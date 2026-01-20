import argparse
import os

import torch
from IPython import get_ipython

from models import sam, dift, dinov2
from utils.utils_results import compute_correct_per_category, compute_pckt_keypoints, compute_pckt_images


def build_parser():
    parser = argparse.ArgumentParser()

    model = parser.add_subparsers(dest='model', required=True)
    dinov2 = model.add_parser("dinov2")
    dinov2.add_argument("--custom-weights", type=str, required=False, help="path to custom weights")

    dinov3 = model.add_parser("dinov3")
    dinov3.add_argument("--custom-weights", type=str, required=False, help="path to custom weights")

    sam = model.add_parser("sam")
    sam.add_argument("--custom-weights", type=str, required=False, help="path to custom weights")

    dift = model.add_parser("dift")
    dift.add_argument("--fuse-dino", type=bool, required=False, help="path to custom weights")
    dift.add_argument("--ensemble-size", type=int, default=4,
                      help="Number of augmentations for DIFT feature extraction", required=False)

    win_soft_argmax = parser.add_argument_group('win_soft_argmax')
    win_soft_argmax.add_argument(
        '--win-soft-argmax',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use windowed soft argmax for correspondence regression',
    )
    win_soft_argmax.add_argument('--wsam-win-size', type=int, default=3, help='Window size for windowed soft argmax')
    win_soft_argmax.add_argument('--wsam-beta', type=float, default=20.0,
                                 help='Inverse temperature for windowed soft argmax')

    dataset = parser.add_argument('--dataset', type=str, default='spair-71k',
                                  choices=['pf-pascal', 'pf-willow', 'spair-71k', 'ap-10k'])

    return parser


def main():
    args = build_parser().parse_args()
    print("Starting evaluation...")
    using_colab = 'google.colab' in str(get_ipython())
    args.using_colab = using_colab

    if using_colab:
        base_dir = os.path.join(os.path.abspath(os.path.curdir), 'AML-polito')
    else:
        base_dir = os.path.abspath(os.path.curdir)
    args.base_dir = base_dir

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match args.model:
        case 'dinov2':
            model = dinov2.Dinov2Eval(args)

        case 'dinov3':
            raise NotImplementedError

        case 'sam':
            model = sam.SamEval(args)

        case 'dift':
            if hasattr(args, 'fuse_dino') and args.fuse_dino:
                from models.SdFuseDino import SdFuseDino
                model = SdFuseDino(args)
            else:
                model = dift.DiftEval(args)

        case _:
            raise ValueError(f"Unknown model: {args.model}")

    results = model.evaluate()
    correct = compute_correct_per_category(results)
    compute_pckt_keypoints(correct)
    compute_pckt_images(correct)


if __name__ == "__main__":
    main()
