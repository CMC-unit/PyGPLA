"""High-level convenience interface for GPLA analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .typing import ArrayLike

__all__ = ["GPLAResult", "gpla"]


@dataclass(slots=True)
class GPLAResult:
    """Container for GPLA outputs."""

    lfp_vector: np.ndarray
    spike_vector: np.ndarray
    gplv: float
    stats: Dict[str, Any]
    metadata: Dict[str, Any]


def gpla(
    spike_trains: Sequence[np.ndarray],
    lfp_signal: ArrayLike,
    *,
    config: Optional["PreprocessingConfig"] = None,
    stats_config: Optional[Dict[str, Any]] = None,
    return_metadata: bool = True,
) -> GPLAResult:
    """
    Compute GPLA metrics for the provided spike and LFP data.

    Parameters
    ----------
    spike_trains:
        Collection of per-trial spike trains (units × samples) in 0/1 format.
    lfp_signal:
        Complex analytic LFP signal (channels × samples × trials) or compatible array-like.
    config:
        Optional preprocessing configuration. Will default to sensible choices once the
        preprocessing module is wired in.
    stats_config:
        Optional configuration dictionary to enable statistical testing (e.g., jittering,
        RMT heuristic). A structured configuration object will replace this dictionary in
        later iterations.
    return_metadata:
        If ``True`` (default) include auxiliary metadata about selections, whitening, etc.

    Notes
    -----
    The implementation is currently a placeholder. The full logic will be ported from
    ``PyGPLA_dev`` once the package skeleton is ready for module migration.
    """

    raise NotImplementedError(
        "pygpla.gpla is not implemented yet; port core routines from PyGPLA_dev."
    )
