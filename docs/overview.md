# Overview

PyGPLA implements Generalized Phase Locking Analysis (GPLA), a dimensionality-reduction method for multichannel spike–field coupling. GPLA builds a complex coupling matrix between LFP phases and spike trains, factors it with SVD, and reports:

- **gPLV**: dominant singular value summarizing global coupling strength.
- **LFP vector**: spatial pattern and phase of LFP channels contributing to coupling.
- **Spike vector**: unit-wise coupling pattern and relative timing.

The Python package mirrors the MATLAB workflow while adding clearer typing, preprocessing helpers, and statistical tests.

## Pipeline at a glance

1. **Prepare data**: concatenate trials, select units, window samples, optionally whiten LFP and normalize amplitudes (`pygpla.preprocessing`).
2. **Build coupling matrix**: analytic LFP × spike trains with configurable normalization and same-electrode correction (`pygpla.core.coupling`).
3. **Factorize**: SVD with column normalization and phase rotation; optional unwhitening of LFP vectors (`pygpla.core.gpla`).
4. **Statistics**: RMT-based analytical test or spike-jitter surrogates for p-values and null distributions (`pygpla.stats.tests`).

## Key components

- High-level API: `pygpla.api.gpla` returning a `GPLAResult` (vectors, gPLV, p-value, stats, metadata).
- Config dataclasses: `PreprocessingConfig`, `WhiteningConfig`, `StatTestConfig` to replace legacy flag soup.
- Simulations: steady and transient phase-locking generators to validate analyses (`pygpla.simulations`).
- Utilities: whitening, jitter generators, and summary helpers.

## When to use GPLA

- Multichannel spike + LFP recordings where pairwise PLV is hard to interpret.
- Hypothesis testing for global coupling (RMT heuristic) or condition comparisons via surrogate tests.
- Exploratory analysis of spatial phase structure (waves, gradients) across electrodes.

## Requirements

- Python ≥ 3.10 (dataclasses use `slots`), NumPy ≥ 1.21.
- For docs: Sphinx + MyST (`pip install -e ".[docs]"`).
