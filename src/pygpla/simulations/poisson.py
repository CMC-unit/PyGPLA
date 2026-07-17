"""Poisson spike train simulators."""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["generate_homogeneous_poisson", "generate_inhomogeneous_poisson"]


def generate_homogeneous_poisson(
    firing_rates,
    duration_s: float,
    sf: float,
    n_tr: Optional[int] = None,
    n_unit: Optional[int] = None,
    method: str = "based_spikingProbality",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate homogeneous Poisson spike trains.

    Parameters replicate the MATLAB/Python port: firing_rates can be scalar,
    per-unit, or per-trial.
    Returns an array of shape (units, n_bins, trials) with 0/1 spikes.

    Pass ``rng`` (a ``numpy.random.Generator``) for reproducible output; when it
    is None a fresh, unseeded generator is used.
    """

    if rng is None:
        rng = np.random.default_rng()

    n_bins = int(round(duration_s * sf))

    fr = np.asarray(firing_rates)
    if fr.ndim == 0:
        fr = fr.reshape(1, 1)

    if n_unit is None:
        n_unit = fr.shape[0]
    if n_tr is None:
        n_tr = fr.shape[1] if fr.ndim > 1 else 1

    if method != "based_spikingProbality":
        raise ValueError("Only 'based_spikingProbality' method is supported.")

    if fr.size == 1:
        prob = np.full((n_unit, n_bins, n_tr), float(fr.item()) / sf)
    elif fr.shape[1] == 1:
        prob = np.repeat(fr, n_bins, axis=1)[:, :, None] / sf
        prob = np.repeat(prob, n_tr, axis=2)
    else:
        tmp = np.zeros((n_unit, 1, n_tr), dtype=float)
        tmp[:, 0, :] = fr
        prob = np.repeat(tmp, n_bins, axis=1) / sf

    spikes = (rng.random((n_unit, n_bins, n_tr)) < prob).astype(int)
    return spikes


def generate_inhomogeneous_poisson(
    fr_modulators: np.ndarray,
    duration_s: float,
    sf: float,
    n_tr: Optional[int] = None,
    n_unit: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate inhomogeneous Poisson spike trains via thinning.

    `fr_modulators` has shape (units, n_bins[, trials]) containing rate profiles (Hz).

    Pass ``rng`` (a ``numpy.random.Generator``) for reproducible output; when it
    is None a fresh, unseeded generator is used.
    """

    if rng is None:
        rng = np.random.default_rng()

    fr = np.asarray(fr_modulators)
    if fr.ndim == 2:
        fr = fr[:, :, None]

    n_unit = n_unit or fr.shape[0]
    n_tr = n_tr or fr.shape[2]

    min_val = fr.min()
    if min_val < 0:
        fr = fr - min_val

    r_max = fr.max(axis=1, keepdims=True)
    base_fr = np.squeeze(r_max, axis=1)
    hp = generate_homogeneous_poisson(
        base_fr, duration_s, sf, n_tr=n_tr, n_unit=n_unit, rng=rng
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = fr / r_max
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    reject = ratio > rng.random(fr.shape)

    spikes = (reject & (hp == 1)).astype(int)
    return spikes
