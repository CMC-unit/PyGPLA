"""Whitening utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["apply_whitening"]


def apply_whitening(
    lfp_signal: np.ndarray,
    *,
    method: int = 0,
    variance_proportion: float | int | None = None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Apply PCA whitening to LFP data, returning the whitened signal and unwhitening operator.
    """

    raise NotImplementedError("Whitening routines not ported yet.")
