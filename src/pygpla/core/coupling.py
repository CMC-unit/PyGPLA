"""Coupling matrix construction."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..typing import ArrayLike

__all__ = ["compute_coupling_matrix"]


def compute_coupling_matrix(
    spike_trains: np.ndarray,
    lfp_signal: np.ndarray,
    *,
    normalization_method: str = "nSpk-square-root",
    same_electrode_info: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the complex coupling matrix between LFP phases and spike trains.

    Returns
    -------
    coupling_matrix, spike_counts
        Placeholder output matching the MATLAB-derived function signature.
    """

    raise NotImplementedError("Coupling matrix computation not ported yet.")
