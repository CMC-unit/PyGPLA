"""Phase-locked spike train simulators."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .poisson import generate_inhomogeneous_poisson

__all__ = ["generate_phase_locked_spikes", "_i0", "_i1"]


def _i0(x: np.ndarray) -> np.ndarray:
    """Modified Bessel function I0 approximation (Abramowitz & Stegun)."""

    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    y = (ax / 3.75) ** 2
    out = np.empty_like(ax)
    mask = ax <= 3.75
    out[mask] = (
        1.0
        + y[mask]
        * (
            3.5156229
            + y[mask]
            * (
                3.0899424
                + y[mask]
                * (
                    1.2067492
                    + y[mask] * (0.2659732 + y[mask] * (0.0360768 + y[mask] * 0.0045813))
                )
            )
        )
    )
    z = np.divide(3.75, ax, out=np.zeros_like(ax), where=ax > 0)
    out[~mask] = (
        (np.exp(ax[~mask]) / np.sqrt(ax[~mask]))
        * (
            0.39894228
            + z[~mask]
            * (
                0.01328592
                + z[~mask]
                * (
                    0.00225319
                    + z[~mask]
                    * (
                        -0.00157565
                        + z[~mask]
                        * (
                            0.00916281
                            + z[~mask]
                            * (
                                -0.02057706
                                + z[~mask]
                                * (
                                    0.02635537
                                    + z[~mask] * (-0.01647633 + z[~mask] * 0.00392377)
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return out


def _i1(x: np.ndarray) -> np.ndarray:
    """Modified Bessel function I1 approximation (Abramowitz & Stegun)."""

    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    y = (ax / 3.75) ** 2
    out = np.empty_like(ax)
    mask = ax <= 3.75
    out[mask] = ax[mask] * (
        0.5
        + y[mask]
        * (
            0.87890594
            + y[mask]
            * (
                0.51498869
                + y[mask]
                * (
                    0.15084934
                    + y[mask] * (0.02658733 + y[mask] * (0.00301532 + y[mask] * 0.00032411))
                )
            )
        )
    )
    z = np.divide(3.75, ax, out=np.zeros_like(ax), where=ax > 0)
    out[~mask] = (
        (np.exp(ax[~mask]) / np.sqrt(ax[~mask]))
        * (
            0.39894228
            + z[~mask]
            * (
                -0.03988024
                + z[~mask]
                * (
                    -0.00362018
                    + z[~mask]
                    * (
                        0.00163801
                        + z[~mask]
                        * (
                            -0.01031555
                            + z[~mask]
                            * (
                                0.02282967
                                + z[~mask]
                                * (
                                    -0.02895312 + z[~mask] * (0.01787654 - z[~mask] * 0.00420059)
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    out = np.where(x < 0, -out, out)
    return out


def _repeat_params(arr, n_unit, n_tr):
    a = np.asarray(arr)
    if a.ndim == 0:
        return np.full((n_unit, n_tr), float(a))
    if a.shape == (1, 1):
        return np.full((n_unit, n_tr), float(a[0, 0]))
    if a.shape == (n_unit, 1):
        return np.repeat(a, n_tr, axis=1)
    if a.shape == (n_unit, n_tr):
        return a
    if a.shape == (1, n_tr):
        return np.repeat(a, n_unit, axis=0)
    raise ValueError("Unexpected param shape")


def generate_phase_locked_spikes(
    spike_params: Dict,
    signal_params: Dict,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate phase-locked Poisson spike trains using von Mises modulation.

    Mirrors the MATLAB/PyGPLA_dev implementation.

    Pass ``rng`` (a ``numpy.random.Generator``) for reproducible output; when it
    is None a fresh, unseeded generator is used.
    """

    a = np.asarray(spike_params.get("lockingPhase", 0.0))
    kappa = np.asarray(spike_params["kappa"])
    avFR = np.asarray(spike_params["avefiringRate"])
    freq = np.asarray(spike_params["lockingFreq"])

    n_unit = int(signal_params["nUnit"])
    n_tr = int(signal_params["nTr"])
    sf = float(signal_params["SF"])
    duration_s = float(signal_params["signalLength"])
    n_bins = int(round(duration_s * sf))

    t = np.linspace(0.0, duration_s, n_bins, endpoint=False)

    A = _repeat_params(a, n_unit, n_tr)
    K = _repeat_params(kappa, n_unit, n_tr)
    R = _repeat_params(avFR, n_unit, n_tr)
    F = _repeat_params(freq, n_unit, n_tr)

    theoPLV = _i1(K) / _i0(K)

    FRmodulators = np.zeros((n_unit, n_bins, n_tr), dtype=float)
    for u in range(n_unit):
        for tr in range(n_tr):
            phase = 2 * np.pi * F[u, tr] * t - A[u, tr]
            FRmodulators[u, :, tr] = R[u, tr] * np.exp(K[u, tr] * np.cos(phase)) / _i0(K[u, tr])

    spikes = generate_inhomogeneous_poisson(
        FRmodulators, duration_s, sf, n_tr=n_tr, n_unit=n_unit, rng=rng
    )
    return spikes, theoPLV, FRmodulators
