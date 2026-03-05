# PyGPLA: Generalized Phase Locking Analysis in Python

[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD%202--Clause-blue.svg)](LICENSE)

<!-- DOCS_INDEX_START -->

PyGPLA is a Python package for analyzing **multichannel spike–field coupling** using
**Generalized Phase Locking Analysis (GPLA)** as introduced in:

Safavi et al. (2023), *Uncovering the organization of neural circuits with Generalized Phase
Locking Analysis*, PLOS Computational Biology.

GPLA summarizes high-dimensional **spike–[Local Field Potential (LFP)](https://en.wikipedia.org/wiki/Local_field_potential)** coupling by constructing a complex coupling matrix and computing its dominant low-rank structure via [Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition). The output includes:

- a scalar coupling strength (**gPLV**)
- an **LFP vector** and **spike vector** whose magnitudes/phases are interpretable at the
  population level
- optional significance testing (analytical RMT-based or spike-jitter surrogates)

## Features

- **GPLA core**: coupling matrix construction + SVD + phase convention
- **Statistical testing**:
  - fast **RMT-based** heuristic (Marchenko–Pastur edge)
  - **spike-jitter** surrogate tests (interval / ISI-preserved / group-preserved / population)
- **Preprocessing hooks**: trial concatenation, spike-count filtering, optional PCA whitening
- **Simulations**: phase-locked and transient coupling generators
- **Figure reproduction**: a Figure 2-style simulation and visualization script

## Installation

### From PyPI (placeholder)

PyPI publishing is planned; once available:

```bash
pip install pygpla
```

### From source (recommended for now)

```bash
git clone https://github.com/CMC-lab/PyGPLA.git
cd PyGPLA
pip install -e .
```

### Optional extras

```bash
# SciPy (for simulations and real-data style filtering/Hilbert workflows)
pip install -e ".[sim]"

# Documentation build (Sphinx + MyST)
pip install -e ".[docs]"

# Tests
pip install -e ".[tests]"
```

## Quick start

This minimal example simulates transiently coupled spikes and an analytic LFP signal, then
runs GPLA via `pygpla.api.gpla`.

```python
import numpy as np

from pygpla.api import gpla
from pygpla.simulations import simulate_transient_locked

rng = np.random.default_rng(123)

signal_params = dict(nCh=1, nUnit=12, SF=600.0, nTr=6, signalLength=6.0)
global_params = dict(
    oscFreq=20.0,
    nCycl=15,
    syncSigProportion=0.7,
    lfpPhaseNoise_kappa=8.0,
    whiteNoise_sigma=0.05,
)
spike_params = dict(avefiringRate=18.0)
coupling_params = dict(lockingStrength_kappa=10.0, lockingPhase=0.0)

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
    lfp_analytic,                     # shape: (channels, samples, trials), complex dtype
    stats_config=None,                # or {"testType": "RMT-based"} / spike-jitter config
    plvNrmlzMethed="var1_theoretical",
    flag_whitening=0,
    flag_lfpNrmlz=0,
)

print("gPLV:", result.gplv)
print("LFP vector shape:", result.lfp_vector.shape)
print("Spike vector shape:", result.spike_vector.shape)
```

Input conventions:

- spike trains: list of trials, each `(n_units, n_samples)`
- analytic LFP: `(n_channels, n_samples, n_trials)` complex array (bandpass + Hilbert in real data)

## Reproducing Figure 2 from the original paper (Safavi et al., 2023)

Run:

```bash
python paper/Figures/figure2.py
```

This generates `paper/Figures/figure2_python.png`.

<!-- DOCS_INDEX_END -->

## Documentation

Build the docs locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Start reading at `docs/_build/html/index.html`.

## Testing

```bash
pip install -e ".[tests]"
pytest
```

## Citing


## License

BSD 2-Clause License. See `LICENSE`.

## Acknowledgments

Developed at the [CMC-Lab](https://shervinsafavi.github.io/cmclab/).

## Contact

- Maintainer: Amir Khani — i.khani.amir@gmail.com
