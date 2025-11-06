"""Spike train surrogate generators."""

from __future__ import annotations

import numpy as np

__all__ = [
    "interval_jitter",
    "isi_preserved_jitter",
    "group_preserved_jitter",
    "population_jitter",
]


def interval_jitter(spike_trains: np.ndarray, window_width: float, sampling_frequency: float) -> np.ndarray:
    """Interval jittering placeholder."""

    raise NotImplementedError("Interval jittering not ported yet.")


def isi_preserved_jitter(spike_trains: np.ndarray, window_width: float, sampling_frequency: float) -> np.ndarray:
    """ISI-preserved jittering placeholder."""

    raise NotImplementedError("ISI-preserved jittering not ported yet.")


def group_preserved_jitter(spike_trains: np.ndarray, window_width: float, sampling_frequency: float) -> np.ndarray:
    """Group-preserved jittering placeholder."""

    raise NotImplementedError("Group-preserved jittering not ported yet.")


def population_jitter(
    spike_trains: np.ndarray,
    window_width_range,
    sampling_frequency: float,
) -> np.ndarray:
    """Population-level jittering placeholder."""

    raise NotImplementedError("Population jittering not ported yet.")
