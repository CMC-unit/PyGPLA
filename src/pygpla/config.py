"""Configuration objects for PyGPLA workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .typing import ArrayLike

__all__ = ["WhiteningConfig", "PreprocessingConfig", "StatTestConfig"]


@dataclass(slots=True)
class WhiteningConfig:
    """Parameters controlling PCA whitening."""

    enabled: bool = False
    method: int = 0  # mirrors MATLAB flag (0, 1, or 2)
    variance_proportion: Union[float, int, None] = None


@dataclass(slots=True)
class PreprocessingConfig:
    """Options for spike/LFP preprocessing prior to GPLA."""

    gplv_normalization: int = 0
    spike_threshold: Optional[int] = None
    unit_subset: Optional[Sequence[int]] = None
    temporal_window: Optional[Union[Tuple[int, int], Sequence[ArrayLike]]] = None
    same_electrode_table: Optional[ArrayLike] = None
    whitening: WhiteningConfig = field(default_factory=WhiteningConfig)
    lfp_normalization: bool = False
    plv_normalization_method: str = "nSpk-square-root"


@dataclass(slots=True)
class StatTestConfig:
    """Statistical testing controls for the GPLA pipeline.

    ``additional_params`` accepts specialized legacy runner options not represented
    by named fields, such as ``rngSeed`` or ``SVspectrumStatsType``. Named fields
    take precedence if the same legacy key is also present in ``additional_params``.
    """

    test_type: str = "RMT-based"
    n_surrogates: int = 0
    jitter_window_width: float = 0.05
    sampling_frequency: float = 1.0
    jitter_type: str = "interval-jittering"
    alpha: float = 0.05
    additional_params: Dict[str, Any] = field(default_factory=dict)


def validate_spike_trains(spike_trains: Sequence[np.ndarray]) -> None:
    """
    Validate that spike trains share dimensions across trials.

    Expects a non-empty sequence of 2D arrays shaped (units, samples); raises ValueError
    if any trial has mismatched unit counts or wrong dimensionality.
    """

    if not spike_trains:
        raise ValueError("At least one spike train trial must be provided.")

    n_units = spike_trains[0].shape[0]
    for idx, trial in enumerate(spike_trains):
        if trial.ndim != 2:
            raise ValueError(f"Spike trial {idx} must be a 2D array (units × samples).")
        if trial.shape[0] != n_units:
            raise ValueError("All trials must have the same number of units.")


def validate_lfp_signal(lfp_signal: ArrayLike) -> np.ndarray:
    """
    Validate the dimensions of an analytic-LFP or phase-array input.

    Requires a 3D array shaped (channels, samples, trials); raises ValueError otherwise.
    Complex arrays represent analytic LFP signals. Real arrays are interpreted downstream
    as phase angles in radians, not as raw LFP voltage. Returns the input as an ndarray.
    """

    arr = np.asarray(lfp_signal)
    if arr.ndim != 3:
        raise ValueError("LFP signal must be a 3D array (channels × samples × trials).")
    return arr
