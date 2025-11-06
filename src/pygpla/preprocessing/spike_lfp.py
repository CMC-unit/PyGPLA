"""Spike/LFP preprocessing pipeline."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import PreprocessingConfig, validate_lfp_signal, validate_spike_trains

__all__ = ["prepare_spike_lfp_data"]


def prepare_spike_lfp_data(
    spike_trains: Sequence[np.ndarray],
    lfp_signal: np.ndarray,
    config: PreprocessingConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray, np.ndarray | None]:
    """
    Prepare spike/LFP arrays for GPLA computation (whitening, selection, concatenation).
    """

    raise NotImplementedError("Preprocessing pipeline not ported yet.")
