
"""
Helper functions used internally for segmentation tasks.
"""
from typing import Optional, List

import numpy as np


def _get_valid_optional_params(
    batch_size: Optional[int] = None,
    n_jobs: Optional[int] = None,
):
    """Takes in optional args and returns good values for them if they are None."""
    if batch_size is None:
        batch_size = 10000

    if batch_size <= 0:
        raise ValueError(f"Batch size must be greater than 0, got {batch_size}")
    return batch_size, n_jobs


def _get_summary_optional_params(
    class_names: Optional[List[str]] = None,
    exclude: Optional[List[int]] = None,
    top: Optional[int] = None,
):
    """Takes in optional args and returns good values for them if they are None for summary functions."""
    if exclude is None:
        exclude = []
    if top is None:
        top = 20
    return class_names, exclude, top


def _check_input(labels: np.ndarray, pred_probs: np.ndarray) -> None:
    """
    Checks that the input labels and predicted probabilities are valid.

    Parameters
    ----------
    labels:
        Array of shape ``(N, H, W)`` of integer labels, where `N` is the number of images in the dataset and `H` and `W` are the height and width of the images.

    pred_probs:
        Array of shape ``(N, K, H, W)`` of predicted probabilities, where `N` is the number of images in the dataset, `K` is the number of classes, and `H` and `W` are the height and width of the images.
    """
    if len(labels.shape) != 3:
        raise ValueError("labels must have a shape of (N, H, W)")

    if len(pred_probs.shape) != 4:
        raise ValueError("pred_probs must have a shape of (N, K, H, W)")

    num_images, height, width = labels.shape
    num_images_pred, num_classes, height_pred, width_pred = pred_probs.shape

    if num_images != num_images_pred or height != height_pred or width != width_pred:
        raise ValueError("labels and pred_probs must have matching dimensions for N, H, and W")
