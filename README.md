# PyGPLA

PyGPLA is a Python package for analyzing multichannel spike–field coupling using
Generalized Phase Locking Analysis. It offers:

- Trial-wise preprocessing (unit selection, whitening, same-electrode checks)
- Coupling-matrix construction and low-rank GPLA factorization
- Statistical testing via random-matrix heuristics or spike-jitter surrogates
- Synthetic data generators for Poisson, phase-locked, and transient scenarios

The main entry point is `pygpla.api.gpla`, which returns LFP/Spike eigenvectors,
gPLV values, and optional statistics in one call.

## Quick demo

Run a basic transient-coupling simulation and analyze it with PyGPLA:

```bash
conda activate gpla
python paper/Figure/figure2.py
```

The script spins up the four canonical models (uniform, gradient, clustered,
and null coupling), feeds them through `pygpla.api.gpla`, and plots the resulting
LFP windows, spike rasters, and spike-vector polar maps.

## Roadmap (abridged)

1. Polish packaging metadata (`pyproject.toml`) and documentation.
2. Add regression/unit tests plus continuous integration.
3. Extend simulation/utils coverage where needed.
4. Expand this README with API docs, tutorials, and workflow guidance.

Until then, treat the package as experimental and report gaps or blockers via
issues/PRs.
