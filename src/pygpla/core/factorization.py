"""SVD factorization for GPLA coupling matrices."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["factorize_coupling_matrix"]


def factorize_coupling_matrix(
    coupling_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform SVD on the coupling matrix, handling NaNs consistently with MATLAB port.
    """

    raise NotImplementedError("Coupling matrix factorization not ported yet.")
