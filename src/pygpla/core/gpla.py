"""Core GPLA routine assembling coupling, factorization, and rotations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple, Union

import numpy as np

from .coupling import compute_coupling_matrix
from .factorization import factorize_coupling_matrix

__all__ = ["run_gpla_core"]


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Normalize columns of a complex matrix while guarding against zeros."""

    denom = np.sqrt(np.nansum(np.abs(matrix) ** 2, axis=0, keepdims=True))
    denom[denom == 0] = 1.0
    return matrix / denom


def _select_sv_indices(
    singular_values: np.ndarray, sv_index: Union[int, Iterable[int], str]
) -> list[int]:
    """Resolve requested singular value indices into zero-based positions."""

    if isinstance(sv_index, str):
        return list(range(singular_values.shape[0]))
    if isinstance(sv_index, (int, np.integer)):
        return [int(sv_index) - 1]
    if isinstance(sv_index, Iterable):
        return [int(idx) - 1 for idx in sv_index]
    raise TypeError("sv_index must be an int, iterable of ints, or 'all'.")


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
    Compute the core GPLA decomposition: coupling matrix → SVD → phase-rotated vectors.

    Parameters
    ----------
    spike_trains :
        Array of shape (units, samples) containing binary or count spike trains.
    lfp_phases :
        Complex analytic LFP array of shape (channels, samples).
    normalize_gplv :
        If nonzero, scale gPLV by (n_channels * n_units)^-0.5 (legacy MATLAB option).
    sv_index :
        Singular value(s) to return (1-based int, iterable of ints, or "all").
    same_electrode_info :
        Optional dict mapping spike units to LFP channels for same-electrode correction.
    normalization_method :
        Normalization for coupling matrix: {"nSpk", "nSpk-square-root", "var1_theoretical"}.
    unwhiten_operator :
        Optional unwhitening matrix applied to LFP singular vectors before normalization.

    Returns
    -------
    lfp_vectors : np.ndarray
        Complex array (channels, n_components) of phase-rotated LFP singular vectors.
    spike_vectors : np.ndarray
        Complex array (units, n_components) of phase-rotated spike singular vectors.
    gplv : float
        Dominant singular value (optionally normalized).
    complex_gplv : complex
        Complex-valued gPLV carrying phase from the last selected component.
    coupling_matrix : np.ndarray
        Complex coupling matrix of shape (channels, units).
    singular_values : np.ndarray
        Singular values from the SVD (1D array).
    """

    coupling_matrix, spike_counts = compute_coupling_matrix(
        spike_trains,
        lfp_phases,
        normalization_method=normalization_method,
        same_electrode_info=same_electrode_info,
    )

    singular_lfp_vecs_raw, singular_spike_vecs_raw, singular_values = factorize_coupling_matrix(
        coupling_matrix
    )

    if unwhiten_operator is not None:
        unwhitened = unwhiten_operator @ singular_lfp_vecs_raw
        singular_lfp_vecs = _normalize_columns(unwhitened)
    else:
        singular_lfp_vecs = singular_lfp_vecs_raw

    if normalization_method == "nSpk-square-root":
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_spikes = singular_spike_vecs_raw / np.power(spike_counts[:, None], 0.5)
        singular_spike_vecs = _normalize_columns(normalized_spikes)
    else:
        singular_spike_vecs = singular_spike_vecs_raw

    indices = _select_sv_indices(singular_values, sv_index)

    lfp_vec_list = []
    spk_vec_list = []
    last_mean_phase = 0.0

    for idx in indices:
        if idx < 0 or idx >= singular_lfp_vecs.shape[1]:
            continue
        lfp_col = singular_lfp_vecs[:, idx]
        spk_col = singular_spike_vecs[:, idx]
        mean_phase = np.angle(np.nanmean(lfp_col))
        phase_rotator = np.exp(-1j * mean_phase)
        lfp_vec_list.append(lfp_col * phase_rotator)
        spk_vec_list.append(spk_col * phase_rotator)
        last_mean_phase = mean_phase

    if not lfp_vec_list:
        raise ValueError("No singular vectors selected; check sv_index parameter.")

    lfp_vectors = np.column_stack(lfp_vec_list)
    spike_vectors = np.column_stack(spk_vec_list)

    gplv = float(singular_values[0])
    if normalize_gplv:
        gplv *= (coupling_matrix.shape[0] * coupling_matrix.shape[1]) ** -0.5

    complex_gplv = gplv * np.exp(2j * last_mean_phase)

    return (
        lfp_vectors,
        spike_vectors,
        gplv,
        complex_gplv,
        coupling_matrix,
        singular_values,
    )
