from typing import TypedDict, List
import pandas as pd
import torch
import numpy as np


class CorrespondenceResult(TypedDict):
    r""" TypedDict for correspondence results """
    category: str
    pck_threshold_0_05: float
    pck_threshold_0_1: float
    pck_threshold_0_2: float
    distances: list[float]


class CategoryResult(TypedDict):
    r""" TypedDict for per-category results """
    correct_0_05: int
    correct_0_1: int
    correct_0_2: int
    num_keypoints: int


def compute_correct_per_category(results: List[CorrespondenceResult]) -> dict[str, List[CategoryResult]]:
    r""" Computes the number of correct keypoints per category
         Args:
            results: List of CorrespondenceResult
         Returns:
            category_results: dict mapping category to list of CategoryResult
     """
    category_results = {}

    for res in results:
        cat = res["category"]
        if cat not in category_results:
            category_results[cat] = []

        dists_list = res["distances"]
        num_keypoints = len(dists_list)
        dists = torch.tensor(dists_list)

        thr_0_05 = res["pck_threshold_0_05"]
        thr_0_1 = res["pck_threshold_0_1"]
        thr_0_2 = res["pck_threshold_0_2"]

        correct_0_05 = (dists <= thr_0_05).sum().item()
        correct_0_1 = (dists <= thr_0_1).sum().item()
        correct_0_2 = (dists <= thr_0_2).sum().item()

        category_results[cat].append(
            CategoryResult(
                correct_0_05=correct_0_05,
                correct_0_1=correct_0_1,
                correct_0_2=correct_0_2,
                num_keypoints=num_keypoints
            )
        )
    return category_results


def compute_pckt_keypoints(category_results: dict[str, List[CategoryResult]]):
    r"""
    Computes and prints PCK per keypoints
    Args:
        category_results: dict mapping category to list of CategoryResult
    """
    rows_keypoints = []

    for cat, stats_list in category_results.items():
        tot_keypoints = sum(s["num_keypoints"] for s in stats_list)
        tot_0_05 = sum(s["correct_0_05"] for s in stats_list)
        tot_0_1 = sum(s["correct_0_1"] for s in stats_list)
        tot_0_2 = sum(s["correct_0_2"] for s in stats_list)

        pck_0_05 = tot_0_05 / tot_keypoints if tot_keypoints > 0 else np.nan
        pck_0_1 = tot_0_1 / tot_keypoints if tot_keypoints > 0 else np.nan
        pck_0_2 = tot_0_2 / tot_keypoints if tot_keypoints > 0 else np.nan

        rows_keypoints.append({
            "Category": cat,
            "PCK 0.05": pck_0_05 * 100,
            "PCK 0.1": pck_0_1 * 100,
            "PCK 0.2": pck_0_2 * 100,
        })

    df_keypoints = pd.DataFrame(rows_keypoints).sort_values("Category")

    #  "All" = macro-average on categories
    mean_row_kp = {
        "Category": "All",
        "PCK 0.05": df_keypoints["PCK 0.05"].mean(skipna=True),
        "PCK 0.1": df_keypoints["PCK 0.1"].mean(skipna=True),
        "PCK 0.2": df_keypoints["PCK 0.2"].mean(skipna=True),
    }

    df_keypoints = pd.concat(
        [df_keypoints, pd.DataFrame([mean_row_kp])],
        ignore_index=True
    )

    print("PCK Results per keypoints (%):")
    print(df_keypoints)


def compute_pckt_images(category_results: dict[str, List[CategoryResult]]):
    r"""
    Computes and prints PCK per images
    Args:
        category_results: dict mapping category to list of CategoryResult
    """
    rows_images = []

    for cat, stats_list in category_results.items():

        pck_imgs_0_05 = []
        pck_imgs_0_1 = []
        pck_imgs_0_2 = []

        for s in stats_list:
            if s["num_keypoints"] == 0:
                continue

            pck_imgs_0_05.append(s["correct_0_05"] / s["num_keypoints"])
            pck_imgs_0_1.append(s["correct_0_1"] / s["num_keypoints"])
            pck_imgs_0_2.append(s["correct_0_2"] / s["num_keypoints"])

        rows_images.append({
            "Category": cat,
            "PCK 0.05": np.mean(pck_imgs_0_05) * 100 if pck_imgs_0_05 else np.nan,
            "PCK 0.1": np.mean(pck_imgs_0_1) * 100 if pck_imgs_0_1 else np.nan,
            "PCK 0.2": np.mean(pck_imgs_0_2) * 100 if pck_imgs_0_2 else np.nan,
        })

    df_image = pd.DataFrame(rows_images).sort_values("Category")

    #  "All" = macro-average on categories
    all_row = {
        "Category": "All",
        "PCK 0.05": df_image["PCK 0.05"].mean(skipna=True),
        "PCK 0.1": df_image["PCK 0.1"].mean(skipna=True),
        "PCK 0.2": df_image["PCK 0.2"].mean(skipna=True),
    }

    df_image = pd.concat([df_image, pd.DataFrame([all_row])], ignore_index=True)

    print("PCK per-image (%):")
    print(df_image)