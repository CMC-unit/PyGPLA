 (quickstart)=
# Quickstart

This page gets you from zero to a first GPLA result in a few minutes.

You will:

- verify the installation
- run GPLA on a small synthetic dataset
- learn the expected input shapes
- (optionally) add a significance test

If you want the full Figure 2-style reproduction with transient models and multi-panel plots,
see {doc}`tutorials`.

## Install

From the repo root:

```bash
pip install -e .
```

Optional (recommended) extras:

```bash
pip install -e ".[sim]"
```

:::{note}
`.[sim]` installs SciPy. You don’t need it for the core GPLA computation itself, but you often
need it for real-data preprocessing (bandpass filtering + Hilbert transform).
:::

## Installation check

```{code-block} python
import pygpla
print(f"PyGPLA version: {pygpla.__version__}")
```

If this fails, confirm you installed from the repo root and that your environment is active.

## Minimal example (synthetic transient coupling)

This uses the built-in simulator `{py:func}`pygpla.simulations.transient.simulate_transient_locked``
to generate:

- `spikes`: a list of trials, each shaped `(n_units, n_samples)` with 0/1 spikes
- `lfp_analytic`: a complex array shaped `(n_channels, n_samples, n_trials)`

```{code-block} python
:linenos:
import numpy as np

from pygpla.api import gpla
from pygpla.simulations import simulate_transient_locked

rng = np.random.default_rng(123)

signal_params = dict(
    nCh=1,
    nUnit=12,
    SF=600.0,
    nTr=6,
    signalLength=6.0,
)

global_params = dict(
    oscFreq=20.0,
    nCycl=15,
    syncSigProportion=0.7,
    lfpPhaseNoise_kappa=8.0,
    whiteNoise_sigma=0.05,
)

spike_params = dict(avefiringRate=18.0)
coupling_params = dict(
    lockingStrength_kappa=10.0,  # 0.0 would be a “null” (no coupling) control
    lockingPhase=0.0,            # preferred phase (radians); can also be an array per unit
)

lfp_real, lfp_analytic, spikes, meta = simulate_transient_locked(
    global_params,
    spike_params,
    coupling_params,
    signal_params,
    return_analytic=True,
    rng=rng,
)

result = gpla(
    spikes,
    lfp_analytic,
    stats_config=None,              # or configure RMT / spike-jittering (see below)
    plvNrmlzMethed="var1_theoretical",
    flag_whitening=0,
    flag_lfpNrmlz=0,
)

print("gPLV:", result.gplv)
print("LFP vector shape:", result.lfp_vector.shape)
print("Spike vector shape:", result.spike_vector.shape)
print("Selected units:", result.metadata["selected_units"])
```

**Key points**

- `result.gplv` is the scalar coupling strength (the leading singular value of the coupling matrix).
- `result.lfp_vector[:, 0]` is the dominant LFP pattern across channels.
- `result.spike_vector[:, 0]` is the dominant spike pattern across units.
- The vectors are complex: magnitude ≈ contribution strength, angle ≈ relative phase.

## Understanding the inputs (shapes)

`gpla(spike_trains, lfp_signal, ...)` expects:

- `spike_trains`: list of length `n_trials`, each element a 2D array `(n_units, n_samples)`
- `lfp_signal`: preferably a 3D **complex analytic** array
  `(n_channels, n_samples, n_trials)`

:::{important}
`gpla()` does not interpret a real-valued array as raw LFP voltage. Real input is supported
only as phase angles in radians and produces a warning. For raw LFP data, first bandpass
filter around the frequency band of interest and apply a Hilbert transform to obtain the
complex analytic signal. The Figure 2 tutorial shows a concrete example: {doc}`tutorials`.
:::

## Interpreting the outputs (what you get back)

`result` is a `{py:class}`pygpla.api.GPLAResult``:

- `result.gplv`: the dominant singular value of the coupling matrix (global coupling strength)
- `result.lfp_vector`: complex LFP pattern (channels × components)
- `result.spike_vector`: complex spike pattern (units × components)
- `result.p_value`: p-value from the spike-jitter test (NaN when no test is requested, and NaN by design for the analytical RMT-based test — read its decision from `result.stats["gPLV_stats"]["nullHypoReject"]`)
- `result.metadata["raw_svd"]["couplingMatrix"]`: the complex coupling matrix used for SVD

A quick way to look at phases/magnitudes:

```{code-block} python
import numpy as np

spike_phases = np.angle(result.spike_vector[:, 0])
spike_magnitudes = np.abs(result.spike_vector[:, 0])
print("Spike phase (rad):", spike_phases)
print("Spike magnitude:", spike_magnitudes)
```

## Working with real data (building the analytic LFP)

PyGPLA expects an **analytic** (complex) LFP input. If you have a real LFP time series, a
common workflow is:

1. bandpass filter around a frequency band of interest
2. apply a Hilbert transform to get the complex analytic signal

```{code-block} python
:linenos:
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def bandpass_hilbert(x: np.ndarray, sf: float, band_hz: tuple[float, float]) -> np.ndarray:
    nyq = sf / 2.0
    b, a = butter(2, [band_hz[0] / nyq, band_hz[1] / nyq], btype="bandpass")
    xf = filtfilt(b, a, x)
    return hilbert(xf)

# Example shape convention:
# lfp_real: (n_channels, n_samples, n_trials)
# lfp_analytic: same shape, complex dtype
```

You can then pass `lfp_analytic` into `{py:func}`pygpla.api.gpla``.

If you already have phase angles in radians, you may pass that real-valued phase array
directly. PyGPLA emits a warning to distinguish this supported phase representation from
accidentally supplied raw LFP voltage.

## Adding significance testing (optional)

PyGPLA supports two main significance-testing modes:

### 1) Analytical RMT-based test (fast)

```{code-block} python
result = gpla(
    spikes,
    lfp_analytic,
    stats_config={"testType": "RMT-based"},
    plvNrmlzMethed="var1_theoretical",
)
print("RMT reject?:", result.stats["gPLV_stats"]["nullHypoReject"])
```

:::{note}
The RMT-based test is an analytical **threshold** test (the top singular value
against the Marchenko–Pastur edge). It returns a **reject/accept decision**, not a
continuous p-value, so `result.p_value` is `NaN` for this test *by design*. Read the
decision from `result.stats["gPLV_stats"]["nullHypoReject"]`. If you specifically
need a p-value, use the spike-jitter surrogate test below.
:::

### 2) Spike-jitter surrogates (flexible but slower)

```{code-block} python
stats_config = {
    "testType": "spike-jittering",
    "nJtr": 200,
    "alphaValue": 0.05,
    "jitterType": "interval-jittering",
    "jitterWinWidth": 0.05,        # seconds
    "spkSF": signal_params["SF"],  # Hz
}

result = gpla(
    spikes,
    lfp_analytic,
    stats_config=stats_config,
    plvNrmlzMethed="var1_theoretical",
)
print("p-value:", result.p_value)
```

:::{important}
Jitter p-values require enough surrogates (`nJtr`) to be meaningful; `nJtr=1` is only useful
to sanity-check the code path.
:::

## Common troubleshooting

:::{admonition} “LFP signal must be a 3D array…”
:class: warning
Ensure `lfp_signal` is shaped `(channels, samples, trials)` (not `(samples, channels)` and not
missing the trials dimension).
:::

:::{admonition} “Whitening requires analytic (complex) LFP signal”
:class: warning
Whitening is only defined for complex analytic LFPs in PyGPLA. Compute the analytic signal
first (bandpass + Hilbert), then set `flag_whitening=1` or `2`.
:::

:::{admonition} “Units with NaNs were excluded from SVD”
:class: note
This usually means some units had zero spikes, which creates NaNs under spike-count
normalization. Use `nSpikeThreshold` (or a `PreprocessingConfig`) to filter low-spike units.
:::

## What’s next?

- Full reproduction and deeper explanation: {doc}`tutorials`
- API reference: {doc}`api`
- Usage notes (normalization, whitening, statistics knobs): {doc}`usage`
