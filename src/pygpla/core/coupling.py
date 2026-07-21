"""Coupling matrix construction."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["compute_coupling_matrix"]


def _analytic_representation(lfp_signal: np.ndarray, normalization_method: str) -> np.ndarray:
    """Return complex-valued representation used for coupling matrix computation."""

    if normalization_method == "var1_theoretical":
        if np.isrealobj(lfp_signal):
            return np.exp(1j * lfp_signal)
        return np.exp(1j * np.angle(lfp_signal))

    if np.isrealobj(lfp_signal):
        return np.exp(1j * lfp_signal)
    amplitude = np.abs(lfp_signal)
    phase = np.exp(1j * np.angle(lfp_signal))
    return amplitude * phase


def _normalization_factors(spike_counts: np.ndarray, method: str) -> np.ndarray:
    """Compute normalization factors for each unit."""

    # Zero-spike units intentionally produce non-finite factors (handled below),
    # so silence the expected divide-by-zero from the power operation.
    with np.errstate(divide="ignore"):
        if method == "nSpk":
            nf = np.power(spike_counts, -1.0)
        elif method in ("nSpk-square-root", "var1_theoretical"):
            nf = np.power(spike_counts, -0.5)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    nf[~np.isfinite(nf)] = np.nan
    return nf


def compute_coupling_matrix(
    spike_trains: np.ndarray,
    lfp_signal: np.ndarray,
    *,
    normalization_method: str = "nSpk-square-root",
    same_electrode_info: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the complex coupling matrix between analytic LFP and spike trains.

    Parameters
    ----------
    spike_trains:
        Array of shape (units, samples) with binary or count spikes.
    lfp_signal:
        Real phase-angle or complex analytic-LFP array of shape (channels, samples).
        Real inputs are interpreted as radians and wrapped to unit-magnitude complex
        exponentials; they are not treated as raw LFP voltage. Complex inputs preserve
        amplitude.
    normalization_method:
        One of ``{"nSpk", "nSpk-square-root", "var1_theoretical"}`` controlling spike-count
        scaling in the coupling matrix.
    same_electrode_info:
        Optional dict for same-electrode correction. If it includes
        ``spkU_lfpCh_cnvrtTabel``, those entries are replaced by coupling values computed
        from interval-jittered spikes.

    Returns
    -------
    coupling_matrix:
        Complex matrix of shape (channels, units).
    spike_counts:
        Total spike counts per unit.
    """

    spikes = np.asarray(spike_trains)
    lfp = np.asarray(lfp_signal)

    if spikes.ndim != 2:
        raise ValueError("Spike trains must be a 2D array (units × time).")
    if lfp.ndim != 2:
        raise ValueError("LFP signal must be a 2D array (channels × time).")
    if spikes.shape[1] != lfp.shape[1]:
        raise ValueError("Spike trains and LFP signal must have matching time samples.")

    analytic_lfp = _analytic_representation(lfp, normalization_method)
    spikes_float = spikes.astype(float, copy=False)
    spike_counts = spikes_float.sum(axis=1).astype(float)

    coupling_raw = analytic_lfp @ spikes_float.T
    nf = _normalization_factors(spike_counts, normalization_method)
    coupling_matrix = (nf[np.newaxis, :] * np.abs(coupling_raw)) * np.exp(
        1j * np.angle(coupling_raw)
    )

    if same_electrode_info is not None:
        # Lazy import to avoid circular dependency during module import time
        from ..stats.jitter import interval_jitter

        jitter_window = float(same_electrode_info.get("jitterWinWidth", 0.05))
        sampling_rate = float(same_electrode_info.get("spkSF", 1.0))
        table = same_electrode_info.get("spkU_lfpCh_cnvrtTabel", None)

        if table is not None:
            table = np.asarray(table)
            jittered = interval_jitter(
                spikes.astype(int, copy=False), jitter_window, sampling_rate
            )
            jittered_coupling, _ = compute_coupling_matrix(
                jittered, lfp, normalization_method=normalization_method, same_electrode_info=None
            )

            if table.ndim == 2 and table.shape[1] == 2:
                limit = min(table.shape[0], coupling_matrix.shape[1])
                for unit_idx in range(limit):
                    ch_index = int(table[unit_idx, 1]) - 1  # MATLAB → Python index
                    if 0 <= ch_index < coupling_matrix.shape[0]:
                        coupling_matrix[ch_index, unit_idx] = jittered_coupling[ch_index, unit_idx]

    return coupling_matrix, spike_counts
