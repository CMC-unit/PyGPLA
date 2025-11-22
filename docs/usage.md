# Usage

## Installation

- Editable dev install with simulation extras:

```bash
pip install -e .[sim,tests]
```

- Documentation build (Sphinx + MyST):

```bash
pip install -e .[docs]
```

## Requirements

- Python ≥ 3.10 (needed for dataclasses with `slots`).
- NumPy ≥ 1.21. Install extras as needed: `.[sim]` for simulations, `.[docs]` for docs.

## Data expectations

- **Spike trains**: list of arrays shaped `(units, samples)` per trial, binary or counts.
- **LFP analytic signal**: complex array shaped `(channels, samples, trials)`; use a bandpass + Hilbert transform upstream.
- **Sampling**: spike and LFP sample counts must match within each trial; supply `sampling_frequency` when using jitter surrogates.

## Minimal example

```python
import numpy as np
from pygpla.api import gpla
from pygpla.simulations import simulate_transient_locked

# Simulate transiently locked spikes and LFP
global_params = dict(oscFreq=20.0, nCycl=10, syncSigProportion=0.7, lfpPhaseNoise_kappa=8.0, whiteNoise_sigma=0.05)
spike_params = dict(avefiringRate=15.0)
coupling_params = dict(lockingStrength_kappa=10.0, lockingPhase=0.0)
signal_params = dict(nCh=1, nUnit=12, SF=600.0, nTr=4, signalLength=5.0)

lfp_real, lfp_analytic, spikes, _ = simulate_transient_locked(
    global_params, spike_params, coupling_params, signal_params, return_analytic=True, rng=np.random.default_rng(123)
)

result = gpla(
    spikes,
    lfp_analytic,
    stats_config=None,            # or pass StatTestConfig / dict for RMT or jitter tests
    plvNrmlzMethed="var1_theoretical",
    flag_whitening=0,
    flag_lfpNrmlz=0,
)

print("gPLV:", result.gplv)
print("LFP vector shape:", result.lfp_vector.shape)
print("Spike vector shape:", result.spike_vector.shape)
```

## Statistical testing

- **RMT analytical**: set `stats_config={"testType": "RMT-based"}`; uses Marchenko–Pastur bound on the top singular value.
- **Spike-jitter surrogates**: `stats_config={"testType": "spike-jittering", "nJtr": 200, "jitterType": "interval-jittering", "jitterWinWidth": 0.05, "alphaValue": 0.05, "spkSF": sampling_rate}`; supports interval, ISI-preserved, group-preserved, and population jittering.

## Preprocessing knobs

- `plvNrmlzMethed`: `nSpk`, `nSpk-square-root` (default), or `var1_theoretical`.
- `flag_whitening`: 0 (off), 1/2 for PCA whitening variants; optionally set `PreprocessingConfig.whitening.variance_proportion`.
- `flag_lfpNrmlz`: normalize analytic LFP amplitude if set.
- `sameElecCheckInfo_r`: provide a mapping to replace same-electrode entries with jitter-based estimates.

## Reproducibility tips

- Set RNG seeds for simulations and surrogate tests (`rngSeed` in `stats_config` or pass `np.random.default_rng`).
- Record parameter dictionaries alongside results; `GPLAResult.metadata` already stores selected units, dimensions, and unwhitening operator.
