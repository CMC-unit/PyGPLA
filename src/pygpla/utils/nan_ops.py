"""Utilities to handle NaNs in matrices."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = ["remove_nan_entries"]


def remove_nan_entries(
    matrix: np.ndarray,
    *,
    distribution: str = "sparse",
    remove: str = "both",
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Remove NaNs from a matrix similarly to the MATLAB helper in PyGPLA_dev.
    """

    raise NotImplementedError("NaN removal helper not ported yet.")
