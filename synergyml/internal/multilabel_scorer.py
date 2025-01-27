
"""
Helper classes and functions used internally to compute label quality scores in multi-label classification.
"""
from enum import Enum
from typing import Callable, Dict, Optional, Union

import numpy as np
from sklearn.model_selection import cross_val_predict
from scipy.special import softmax


def softmin(
    s: np.ndarray,
    *,
    temperature: float = 0.1,
    axis: int = 1,
    **_,
) -> np.ndarray:
    """Softmin score aggregation function.

    Parameters
    ----------
    s :
        Input array.

    temperature :
        Temperature parameter. Too small values may cause numerical underflow and NaN scores.

    axis :
        Axis along which to apply the function.

    Returns
    -------
        Softmin score.
    """

    return np.einsum(
        "ij,ij->i", s, softmax(x=1 - s, temperature=temperature, axis=axis, shift=True)
    )

