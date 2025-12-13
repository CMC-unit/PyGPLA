# Quickstart

This page shows the smallest end-to-end GPLA run in PyGPLA:

1. create (or load) spike trains and an **analytic** LFP signal
2. run `{py:func}`pygpla.api.gpla``
3. inspect the gPLV and the spike/LFP vectors

If you want a full “publication-style” reproduction with transient models and multi-panel
plots, see `docs/tutorials.md`.

## Install

From the repo root:

```bash
pip install -e .
```

Optional (recommended) extras:

```bash
pip install -e ".[sim]"
```

## Minimal example (synthetic transient coupling)

This uses the built-in simulator `{py:func}`pygpla.simulations.transient.simulate_transient_locked``
to generate:

- `spikes`: a list of trials, each shaped `(n_units, n_samples)` with 0/1 spikes
- `lfp_analytic`: a complex array shaped `(n_channels, n_samples, n_trials)`

```python
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

## Understanding the inputs (shapes)

`gpla(spike_trains, lfp_signal, ...)` expects:

- `spike_trains`: list of length `n_trials`, each element a 2D array `(n_units, n_samples)`
- `lfp_signal`: a 3D **complex** array `(n_channels, n_samples, n_trials)`

:::{note}
For real data, you typically build `lfp_signal` by bandpass filtering around a target
frequency and applying a Hilbert transform to get the complex analytic signal.
The Figure 2 tutorial shows a concrete example of that preprocessing.
:::

## Interpreting the outputs (what you get back)

`result` is a `{py:class}`pygpla.api.GPLAResult``:

- `result.gplv`: the dominant singular value of the coupling matrix (global coupling strength)
- `result.lfp_vector`: complex LFP pattern (channels × components)
- `result.spike_vector`: complex spike pattern (units × components)
- `result.p_value`: p-value when a statistical test is requested (else NaN)
- `result.metadata["raw_svd"]["couplingMatrix"]`: the complex coupling matrix used for SVD

A quick way to look at phases/magnitudes:

```python
import numpy as np

spike_phases = np.angle(result.spike_vector[:, 0])
spike_magnitudes = np.abs(result.spike_vector[:, 0])
print("Spike phase (rad):", spike_phases)
print("Spike magnitude:", spike_magnitudes)
```

## Adding significance testing (optional)

PyGPLA supports two main significance-testing modes:

### 1) Analytical RMT-based test (fast)

```python
result = gpla(
    spikes,
    lfp_analytic,
    stats_config={"testType": "RMT-based"},
    plvNrmlzMethed="var1_theoretical",
)
print("RMT reject?:", result.stats["gPLV_stats"]["nullHypoReject"])
```

### 2) Spike-jitter surrogates (flexible but slower)

```python
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

## Next steps

- Reproduce Figure 2 and learn the code structure: `docs/tutorials.md`
- Browse the API reference: `docs/api.md`
