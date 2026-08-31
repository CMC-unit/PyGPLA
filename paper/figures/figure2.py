#!/usr/bin/env python
"""
Reproduce the transient-coupling illustration (Figure 2) using the new pygpla API.

This is a near-direct port of `visualizations/figure2/figure2.py` from PyGPLA_dev.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

FIGURES_DIR = Path(__file__).resolve().parent
PAPER_DIR = FIGURES_DIR.parent
REPO_ROOT = PAPER_DIR.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MPL_CACHE = PAPER_DIR / ".matplotlib_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from pygpla.api import gpla
from pygpla.simulations import simulate_transient_locked

DPI = 600
PANEL_LABEL_SIZE = 14
TITLE_SIZE = 9.5
AXIS_LABEL_SIZE = 8.5
TICK_LABEL_SIZE = 7.5
SAVE_FORMATS = ("png", "pdf", "svg", "eps")


@dataclass
class SignalParams:
    nCh: int = 1
    nUnit: int = 18
    SF: float = 1e3
    nTr: int = 10
    signalLength: float = 10.0


def _choose_event_window(
    meta: Dict, sf: float, tr_index: int, pad_samples: int = 100
) -> Tuple[slice, slice]:
    events = meta["event_trains"][tr_index]
    if len(events) == 0:
        mid = int(round(meta["transient_length_samples"]))
        return slice(0, 2 * mid), slice(0, 2 * mid)
    ev = events[min(1, len(events) - 1)]
    start = int(round(ev * sf))
    mid = start + meta["transient_length_samples"] // 2
    rng = slice(max(0, mid - pad_samples), mid + pad_samples)
    full = slice(
        max(0, start - pad_samples), start + meta["transient_length_samples"] + pad_samples
    )
    return rng, full


def _polar_spkvec(ax, spk_vec: np.ndarray, title: str = "", color="red"):
    ax.set_theta_zero_location("E")
    th = np.angle(spk_vec)
    r = np.abs(spk_vec)
    ax.scatter(th, r, s=14, c=color, alpha=1.0, linewidths=0)
    max_r = max(0.5, np.nanmax(r) if r.size else 0.5)
    ax.set_rmax(max_r)
    ax.set_rticks([0, max_r / 2, max_r])
    ax.set_rgrids(
        [max_r / 2, max_r],
        labels=[f"{max_r/2:.1f}", f"{max_r:.1f}"],
        fontsize=TICK_LABEL_SIZE,
    )
    ax.set_thetagrids(np.arange(0, 360, 45))
    ax.grid(True, color="0.82", linewidth=0.45)
    ax.tick_params(labelsize=TICK_LABEL_SIZE, pad=1)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=4)


def _bandpass_and_analytic(
    lfp_real: np.ndarray, sf: float, band: Tuple[float, float]
) -> np.ndarray:
    n_ch, n_bins, n_tr = lfp_real.shape
    out = np.zeros((n_ch, n_bins, n_tr), dtype=np.complex128)
    nyq = sf / 2.0
    wn = [band[0] / nyq, band[1] / nyq]
    b, a = butter(2, wn, btype="bandpass")
    for tr in range(n_tr):
        for ch in range(n_ch):
            x = lfp_real[ch, :, tr]
            xf = filtfilt(b, a, x)
            out[ch, :, tr] = hilbert(xf)
    return out


def _plot_model_schematic(ax, model_type: int, n_units: int = 6):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    circle = plt.Circle((5, 8), 0.8, color="black", fill=False, linewidth=1.2)
    ax.add_patch(circle)

    unit_positions = np.linspace(1.5, 8.5, n_units)
    if model_type == 1:
        colors = ["cyan"] * n_units
        line_widths = [2] * n_units
    elif model_type == 2:
        colors = plt.cm.hsv(np.linspace(0, 0.33, n_units))
        line_widths = [2] * n_units
    elif model_type == 3:
        cluster_colors = ["red", "green", "blue"]
        colors = []
        for i in range(n_units):
            cluster_idx = i // (n_units // 3)
            cluster_idx = min(cluster_idx, 2)
            colors.append(cluster_colors[cluster_idx])
        line_widths = [2] * n_units
    else:
        colors = ["gray"] * n_units
        line_widths = [0.5] * n_units

    for i, (x_pos, color, lw) in enumerate(zip(unit_positions, colors, line_widths)):
        triangle = plt.Polygon(
            [(x_pos - 0.3, 2), (x_pos + 0.3, 2), (x_pos, 2.6)], color=color, alpha=1.0
        )
        ax.add_patch(triangle)
        if lw > 0.5:
            ax.plot([x_pos, 5], [2.6, 7.2], color=color, linewidth=1.1, alpha=1.0)

    ax.set_aspect("equal")
    ax.axis("off")


def main():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "axes.linewidth": 0.8,
            "axes.titleweight": "regular",
            "axes.labelpad": 2.5,
            "axes.titlepad": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    signal_params = SignalParams()
    global_params = dict(
        oscFreq=22.5,
        nCycl=20,
        syncSigProportion=0.8,
        lfpPhaseNoise_kappa=10.0,
        whiteNoise_sigma=0.05,
    )

    nU = signal_params.nUnit
    phase_linear = np.linspace(0, np.pi, nU)
    n_chunk = 3
    chunk_sz = nU // n_chunk
    cluster = np.concatenate(
        [
            np.full(chunk_sz, 0.0),
            np.full(chunk_sz, 2 * np.pi / 3),
            np.full(nU - 2 * chunk_sz, 4 * np.pi / 3),
        ]
    )
    rng = np.random.default_rng(123)
    rng.shuffle(cluster)

    cases = [
        dict(name="Model 1", kappa=10.0, phase=np.pi * np.ones(nU)),
        dict(name="Model 2", kappa=10.0, phase=phase_linear),
        dict(name="Model 3", kappa=10.0, phase=cluster),
        dict(name="Model 4", kappa=0.0, phase=np.pi / 2 * np.ones(nU)),
    ]

    gplv_vals: List[float] = []
    spk_vecs: List[np.ndarray] = []
    lfp_examples: List[np.ndarray] = []
    spike_rasters: List[np.ndarray] = []
    full_ranges: List[slice] = []
    win_ranges: List[slice] = []

    for idx, case in enumerate(cases):
        coupling_params = dict(
            lockingStrength_kappa=case["kappa"],
            lockingPhase=case["phase"],
        )
        spike_params = dict(avefiringRate=20.0)
        lfp_real, _lfp_analytic_sim, spikes, meta = simulate_transient_locked(
            global_params,
            spike_params,
            coupling_params,
            dict(
                nCh=signal_params.nCh,
                nUnit=signal_params.nUnit,
                SF=signal_params.SF,
                nTr=signal_params.nTr,
                signalLength=signal_params.signalLength,
            ),
            return_analytic=False,
            rng=np.random.default_rng(42 + idx),
        )

        band = (global_params["oscFreq"] - 2.5, global_params["oscFreq"] + 2.5)
        lfp_analytic = _bandpass_and_analytic(lfp_real, signal_params.SF, band)

        spikeTrains_raw = [spikes[tr] for tr in range(signal_params.nTr)]
        lfpPhases_raw = lfp_analytic

        statTestInfo = {
            "testType": "spike-jittering",
            "SVspectrumStatsType": "RMT-heuristic",
            "jitterType": "group-preserved-interval-jittering",
            "nJtr": 1,
            "alphaValue": 0.05,
            "spkSF": signal_params.SF,
            "jitterWinWidth": 1.0 / global_params["oscFreq"],
        }

        result = gpla(
            spikeTrains_raw,
            lfpPhases_raw,
            flag_gPLVnrmlz=0,
            nSpikeThreshold=None,
            unitSubset=None,
            temporalWindow=None,
            flag_origDimEigVec=0,
            stats_config=statTestInfo,
            iSV=1,
            sameElecCheckInfo_r=None,
            plvNrmlzMethed="var1_theoretical",
            flag_whitening=0,
            flag_lfpNrmlz=0,
        )

        lfpVec = result.lfp_vector.copy()
        spkVec = result.spike_vector.copy()
        gPLV = result.gplv

        if np.real(np.nansum(lfpVec[:, 0])) < 0:
            lfpVec[:, 0] = -lfpVec[:, 0]
            spkVec[:, 0] = -spkVec[:, 0]

        gplv_vals.append(gPLV)
        spk_vecs.append(spkVec[:, 0].copy())

        wr, fr = _choose_event_window(meta, signal_params.SF, 0, pad_samples=70)
        win_ranges.append(wr)
        full_ranges.append(fr)
        lfp_examples.append(lfp_real[0, :, 0])
        spike_rasters.append(spikeTrains_raw[0])

    fig = plt.figure(figsize=(7.2, 6.6))

    model_colors = ["#2E8B57", "#DAA520", "#CD853F", "#708090"]

    ax1 = plt.subplot(5, 3, (1, 2))
    sig = lfp_examples[0]
    fr = full_ranges[0]
    wr = win_ranges[0]
    extended_range = slice(max(0, fr.start - 1000), min(len(sig), fr.stop + 3000))
    x = np.arange(len(sig)) / signal_params.SF
    ax1.plot(x[extended_range], sig[extended_range], "k-", linewidth=0.75)
    win_start_time = x[wr.start] if wr.start < len(x) else x[-1]
    win_end_time = x[wr.stop - 1] if wr.stop <= len(x) else x[-1]
    ax1.axvspan(win_start_time, win_end_time, color="#d8e6ff", alpha=1.0, linewidth=0)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("LFP signal with analysis window highlighted")
    for spine in ("top", "right"):
        if spine in ax1.spines:
            ax1.spines[spine].set_visible(False)
    scale_x = x[extended_range][-1] - 0.5
    scale_y = np.max(sig[extended_range]) * 0.8
    ax1.plot([scale_x - 0.5, scale_x], [scale_y, scale_y], "k-", linewidth=1.2)
    ax1.text(scale_x - 0.25, scale_y + 0.1, "500 ms", ha="center", fontsize=TICK_LABEL_SIZE)

    ax2 = plt.subplot(5, 3, 3)
    ax2.bar(
        range(1, 5),
        gplv_vals,
        color=model_colors,
        alpha=1.0,
        edgecolor="black",
        linewidth=0.6,
    )
    ax2.set_xticks(range(1, 5))
    ax2.set_xticklabels(["M1", "M2", "M3", "M4"])
    ax2.set_ylabel("gPLV")
    ax2.set_title("Global PLV")
    ax2.set_ylim(0, max(gplv_vals) * 1.1)
    for spine in ("top", "right"):
        if spine in ax2.spines:
            ax2.spines[spine].set_visible(False)

    model_axes = []
    for i, case in enumerate(cases):
        row = i + 2

        ax_model = plt.subplot(5, 3, (row - 1) * 3 + 1)
        model_axes.append(ax_model)
        _plot_model_schematic(ax_model, i + 1)
        ax_model.set_title(f"{case['name']}", fontsize=TITLE_SIZE, fontweight="bold")

        ax_combined = plt.subplot(5, 3, (row - 1) * 3 + 2)
        sig = lfp_examples[i]
        wr = win_ranges[i]
        x_win = np.arange(len(sig)) / signal_params.SF
        ax_combined.plot(x_win[wr], sig[wr], color=model_colors[i], linewidth=1.0, label="LFP")

        lfp_min = np.min(sig[wr])
        lfp_max = np.max(sig[wr])
        lfp_range = lfp_max - lfp_min
        spk = spike_rasters[i][:, wr]
        u_idx, t_idx = np.where(spk == 1)
        if t_idx.size > 0:
            spike_times = x_win[wr][t_idx]
            unit_positions = lfp_min + (u_idx / signal_params.nUnit) * lfp_range
            colors = plt.cm.Set1(np.linspace(0, 1, signal_params.nUnit))
            spike_colors = [colors[u] for u in u_idx]
            ax_combined.scatter(
                spike_times,
                unit_positions,
                s=7,
                c=spike_colors,
                alpha=1.0,
                marker="|",
                linewidths=0.7,
            )

        ax_combined.set_ylabel("LFP Amplitude")
        ax_combined.set_xlabel("Time (s)")
        if i == 0:
            ax_combined.set_title("LFP + spike trains")
        for spine in ("top", "right"):
            if spine in ax_combined.spines:
                ax_combined.spines[spine].set_visible(False)

        ax_polar = plt.subplot(5, 3, (row - 1) * 3 + 3, projection="polar")
        _polar_spkvec(ax_polar, spk_vecs[i], color="red")

    if len(cases) >= 3:
        cax = fig.add_axes([0.15, 0.02, 0.3, 0.02])
        cax.set_in_layout(False)
        cmap = plt.cm.hsv
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=360))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Phase lag [deg]", fontsize=AXIS_LABEL_SIZE)
        cbar.set_ticks([0, 180, 360])
        cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, width=0.8, length=2.5)

    fig.subplots_adjust(
        left=0.08, right=0.97, top=0.94, bottom=0.10, hspace=0.62, wspace=0.60
    )

    for ax, label in zip([ax1, ax2, *model_axes], "ABCDEF"):
        position = ax.get_position()
        fig.text(
            max(0.01, position.x0 - 0.055),
            min(0.985, position.y1 + 0.018),
            label,
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            ha="left",
            va="top",
            color="black",
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    basepath = FIGURES_DIR / "figure2_python"
    for ext in SAVE_FORMATS:
        outpath = f"{basepath}.{ext}"
        plt.savefig(outpath, dpi=DPI, bbox_inches="tight")
        print(f"Saved figure: {outpath}")


if __name__ == "__main__":
    main()
