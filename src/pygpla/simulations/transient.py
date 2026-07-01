"""Transient coupling simulators."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .phase_locked import _i0
from .poisson import generate_inhomogeneous_poisson

__all__ = ["simulate_transient_locked"]


def _gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    N = n - 1
    n_idx = np.arange(0, n)
    return np.exp(-0.5 * (alpha * (n_idx - N / 2.0) / (N / 2.0)) ** 2)


def _repeat_params(arr, n_unit, n_tr):
    a = np.asarray(arr)
    if a.ndim == 0:
        return np.full((n_unit, n_tr), float(a))
    if a.shape == (1, 1):
        return np.full((n_unit, n_tr), float(a[0, 0]))
    if a.ndim == 1 and a.shape[0] == n_unit:
        return np.repeat(a[:, None], n_tr, axis=1)
    if a.ndim == 1 and a.shape[0] == n_tr:
        return np.repeat(a[None, :], n_unit, axis=0)
    if a.shape == (n_unit, 1):
        return np.repeat(a, n_tr, axis=1)
    if a.shape == (n_unit, n_tr):
        return a
    if a.shape == (1, n_tr):
        return np.repeat(a, n_unit, axis=0)
    if a.size == 1:
        return np.full((n_unit, n_tr), float(a))
    raise ValueError("Unexpected param shape for repeating: %r" % (a.shape,))


def _make_event_trains(
    duration_s: float,
    sf: float,
    osc_freq: float,
    n_cycl: int,
    sync_prop: float,
    n_tr: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    trn_dur = n_cycl / float(osc_freq)
    events: List[np.ndarray] = []
    for _ in range(n_tr):
        event_count = (sync_prop * duration_s) / trn_dur
        rate = max(event_count / duration_s, 1e-12)
        ts = []
        t = -np.log(rng.random()) / rate
        while t < duration_s:
            if (len(ts) == 0) or (t - ts[-1] >= trn_dur):
                ts.append(t)
            t = t + (-np.log(rng.random()) / rate)
        ts = np.array([x for x in ts if x <= duration_s - trn_dur])
        events.append(ts)
    return events


def simulate_transient_locked(
    global_params: Dict,
    spike_params: Dict,
    coupling_params: Dict,
    signal_params: Dict,
    *,
    return_analytic: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Dict]:
    """
    Simulate transiently phase-locked spikes and LFP.
    """

    if rng is None:
        rng = np.random.default_rng()

    n_ch = int(signal_params["nCh"])
    n_unit = int(signal_params["nUnit"])
    n_tr = int(signal_params["nTr"])
    sf = float(signal_params["SF"])
    duration_s = float(signal_params["signalLength"])

    n_bins = int(round(duration_s * sf))
    t = np.linspace(0.0, duration_s, n_bins, endpoint=False)

    osc_freq = float(global_params["oscFreq"])
    n_cycl = int(global_params["nCycl"])
    sync_prop = float(global_params["syncSigProportion"])
    kappa_lfp = float(global_params["lfpPhaseNoise_kappa"])
    sigma = float(global_params["whiteNoise_sigma"])

    event_trains = _make_event_trains(duration_s, sf, osc_freq, n_cycl, sync_prop, n_tr, rng)
    trn_len_s = n_cycl / osc_freq
    trn_len_n = int(round(trn_len_s * sf))

    lfp_real = sigma * rng.standard_normal(size=(n_ch, n_bins, n_tr))
    base_osc = np.exp(1j * (2 * np.pi * osc_freq * t))
    gw = _gausswin(trn_len_n)

    for tr in range(n_tr):
        for ev in event_trains[tr]:
            start = int(round(ev * sf))
            stop = start + trn_len_n
            if stop > n_bins:
                continue
            ph_noise = rng.vonmises(mu=0.0, kappa=kappa_lfp, size=(n_ch, trn_len_n))
            osc_win = base_osc[start:stop]
            cmp = np.exp(1j * ph_noise) * osc_win
            lfp_real[:, start:stop, tr] = (np.real(cmp) * gw)[None, :]

    avFR = _repeat_params(spike_params["avefiringRate"], n_unit, n_tr)
    kappa_spk = _repeat_params(coupling_params["lockingStrength_kappa"], n_unit, n_tr)
    lock_phase = _repeat_params(coupling_params["lockingPhase"], n_unit, n_tr)

    FR = np.zeros((n_unit, n_bins, n_tr), dtype=float)
    I0_k = _i0(kappa_spk)
    for tr in range(n_tr):
        for ev in event_trains[tr]:
            start = int(round(ev * sf))
            stop = start + trn_len_n
            if stop > n_bins:
                continue
            phase = 2 * np.pi * osc_freq * t[start:stop][None, :] - lock_phase[:, tr][:, None]
            num = np.exp(kappa_spk[:, tr][:, None] * np.cos(phase))
            FR[:, start:stop, tr] = avFR[:, tr][:, None] * (num / I0_k[:, tr][:, None])

    spikes_3d = generate_inhomogeneous_poisson(FR, duration_s, sf, n_tr=n_tr, n_unit=n_unit)
    spikes: List[np.ndarray] = [spikes_3d[:, :, tr] for tr in range(n_tr)]

    if return_analytic:
        lfp_phase = np.angle(np.exp(1j * (2 * np.pi * osc_freq * t))[None, :, None])
        lfp_amp = np.maximum(np.abs(lfp_real), 1e-12)
        lfp_analytic = lfp_amp * np.exp(1j * lfp_phase)
    else:
        lfp_analytic = np.nan + 1j * np.nan

    meta = {
        "event_trains": event_trains,
        "transient_length_samples": trn_len_n,
        "transient_length_seconds": trn_len_s,
    }
    return lfp_real, lfp_analytic, spikes, meta
