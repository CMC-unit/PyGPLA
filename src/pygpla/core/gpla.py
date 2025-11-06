"""Core GPLA routine assembling coupling, factorization, and rotations."""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np

__all__ = ["run_gpla_core"]


def run_gpla_core(
    spike_trains: np.ndarray,
    lfp_phases: np.ndarray,
    *,
    normalize_gplv: int = 0,
    sv_index: Union[int, Iterable[int], str] = 1,
    same_electrode_info=None,
    normalization_method: str = "nSpk-square-root",
    unwhiten_operator: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, complex, np.ndarray, np.ndarray]:
    """
    Placeholder for the MATLAB-equivalent core GPLA routine.
    """

    raise NotImplementedError("Core GPLA routine not ported yet.")
