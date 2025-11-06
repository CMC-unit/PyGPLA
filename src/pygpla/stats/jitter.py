"""Spike train surrogate generators."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "interval_jitter",
    "isi_preserved_jitter",
    "group_preserved_jitter",
    "population_jitter",
]


def interval_jitter(
    spike_trains: np.ndarray, window_width: float, sampling_frequency: float
) -> np.ndarray:
    """
    Interval jittering: jitter each spike uniformly inside a 2·window_width window.
    """

    spikes = np.asarray(spike_trains, dtype=int)
    n_unit, n_sample = spikes.shape

    row_idx, col_idx = np.where(spikes == 1)
    spike_times = col_idx / float(sampling_frequency)

    win_len = 2.0 * window_width
    jittered_times = win_len * np.floor(spike_times / win_len) + win_len * np.random.rand(
        spike_times.size
    )
    jittered_idx = np.ceil(jittered_times * sampling_frequency).astype(int)

    out_of_range = (jittered_idx < 1) | (jittered_idx > n_sample)
    if np.any(out_of_range):
        jittered_idx[out_of_range] = np.random.randint(1, n_sample + 1, size=out_of_range.sum())

    jittered = np.zeros((n_unit, n_sample), dtype=int)
    jittered[row_idx, jittered_idx - 1] = 1
    return jittered


def isi_preserved_jitter(
    spike_trains: np.ndarray, window_width: float, sampling_frequency: float
) -> np.ndarray:
    """
    ISI-preserved interval jittering (per unit).
    """

    spikes = np.asarray(spike_trains, dtype=int)
    n_unit, n_sample = spikes.shape
    jittered = np.zeros((n_unit, n_sample), dtype=int)

    win_len = 2.0 * window_width
    for unit_idx in range(n_unit):
        spike_cols = np.where(spikes[unit_idx, :] == 1)[0]
        if spike_cols.size == 0:
            continue

        spike_times = spike_cols / float(sampling_frequency)
        jitter_win_idx = np.floor(spike_times / win_len).astype(int)

        for win_id in np.unique(jitter_win_idx):
            spike_idx = np.where(jitter_win_idx == win_id)[0]
            if spike_idx.size == 0:
                continue

            times = spike_times[spike_idx]
            if times.size <= 1:
                relative_times = np.array([0.0])
            else:
                inter_spike = np.diff(times)
                permuted = np.random.permutation(inter_spike.size)
                relative_times = np.concatenate([[0.0], np.cumsum(inter_spike[permuted])])

            init_range = win_len - relative_times[-1]
            init = init_range * np.random.rand()
            new_times = relative_times + init + win_len * win_id
            new_times = new_times + (1.0 / sampling_frequency) * np.random.rand(*new_times.shape)
            new_cols = np.ceil(new_times * sampling_frequency).astype(int)

            out_of_range = (new_cols < 1) | (new_cols > n_sample)
            if np.any(out_of_range):
                new_cols[out_of_range] = np.random.randint(1, n_sample + 1, size=out_of_range.sum())
            jittered[unit_idx, new_cols - 1] = 1

    return jittered


def group_preserved_jitter(
    spike_trains: np.ndarray, window_width: float, sampling_frequency: float
) -> np.ndarray:
    """
    Group-preserved interval jittering: shared circular shift per window across units.
    """

    spikes = np.asarray(spike_trains, dtype=int)
    n_unit, n_sample = spikes.shape
    window_len = int(np.floor(2.0 * window_width * sampling_frequency))

    if window_len <= 0:
        return spikes.copy()

    jittered = np.zeros((n_unit, n_sample), dtype=int)
    n_windows = n_sample // window_len

    for win_idx in range(n_windows):
        start = win_idx * window_len
        end = start + window_len
        shift = np.random.randint(0, window_len)
        jittered[:, start:end] = np.roll(spikes[:, start:end], shift=shift, axis=1)

    if n_sample > window_len * n_windows:
        start = n_windows * window_len
        shift = np.random.randint(0, window_len)
        jittered[:, start:] = np.roll(spikes[:, start:], shift=shift, axis=1)

    return jittered


def population_jitter(
    spike_trains: np.ndarray,
    window_width_range: Tuple[float, float],
    sampling_frequency: float,
) -> np.ndarray:
    """
    Population-level (synchronous) jittering with window width drawn from a range.
    """

    spikes = np.asarray(spike_trains, dtype=int)
    n_unit, n_sample = spikes.shape

    win_min, win_max = window_width_range
    jitter_width = win_min + (win_max - win_min) * np.random.rand()

    row_idx, col_idx = np.where(spikes == 1)
    if col_idx.size == 0:
        return np.zeros_like(spikes)

    pop_cols, _, inverse = np.unique(col_idx, return_index=True, return_inverse=True)
    pop_times = pop_cols / float(sampling_frequency)

    win_len = 2.0 * jitter_width
    jittered_pop_times = win_len * np.floor(pop_times / win_len) + win_len * np.random.rand(
        pop_times.size
    )
    jittered_pop_cols = np.round(jittered_pop_times * sampling_frequency).astype(int)

    jittered_cols = jittered_pop_cols[inverse]
    jittered = np.zeros((n_unit, n_sample), dtype=int)
    valid = (jittered_cols >= 1) & (jittered_cols <= n_sample)
    if np.any(valid):
        jittered[row_idx[valid], jittered_cols[valid] - 1] = 1
    return jittered
