"""SVD factorization for GPLA coupling matrices."""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

__all__ = ["factorize_coupling_matrix"]


def factorize_coupling_matrix(
    coupling_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform SVD on the coupling matrix, handling NaNs consistently with MATLAB port.

    Parameters
    ----------
    coupling_matrix :
        Complex matrix of shape (channels, units).

    Returns
    -------
    singular_lfp_vecs : np.ndarray
        Left singular vectors (channels, k).
    singular_spike_vecs : np.ndarray
        Right singular vectors (units, k), NaN-padded when NaNs were present.
    singular_values : np.ndarray
        Singular values (length k), where k = min(channels, units) or valid columns.
    """

    M = np.asarray(coupling_matrix, dtype=complex)

    if np.isnan(M).any():
        warnings.warn(
            "Units with NaNs (often due to zero spikes) were excluded from SVD.",
            RuntimeWarning,
        )
        valid_cols = ~np.isnan(M).any(axis=0)
        M_clean = M[:, valid_cols]
        U, s, Vh_clean = np.linalg.svd(M_clean, full_matrices=False)
        V_clean = Vh_clean.conj().T

        V = np.full((M.shape[1], V_clean.shape[1]), np.nan, dtype=complex)
        V[valid_cols, :] = V_clean
        return U, V, s

    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    V = Vh.conj().T
    return U, V, s
