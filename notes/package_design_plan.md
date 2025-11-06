# PyGPLA Packaging Plan

## Goals
- Deliver a Python-native GPLA toolkit with a stable, documented public API covering data preparation, coupling analysis, statistics, and simulation utilities.
- Preserve numerical parity with the MATLAB port while improving readability, typing, and error handling.
- Provide a modern packaging baseline (PEP 621 metadata, src-layout, optional extras) that supports PyPI publication and downstream integration.

## Proposed Repo Layout
```
PyGPLA/
├── pyproject.toml              # PEP 621 metadata, dependency pins, build (hatchling)
├── README.md                   # high-level overview and quickstart
├── src/
│   └── pygpla/
│       ├── __init__.py         # version, public API exports
│       ├── api.py              # convenience facade (one-shot helpers)
│       ├── config.py           # dataclasses / TypedDicts describing configurable inputs
│       ├── core/               # core GPLA mathematics
│       │   ├── __init__.py
│       │   ├── coupling.py     # coupling matrix assembly, normalization
│       │   ├── factorization.py# SVD + complex rotations
│       │   ├── gpla.py         # orchestration (former gpla_core/tngpla without stats)
│       │   └── whitening.py    # PCA whitening utilities (vectorized + trial aware)
│       ├── preprocessing/
│       │   └── spike_lfp.py   # trial concatenation, unit selection, validation
│       ├── stats/
│       │   ├── __init__.py
│       │   ├── jitter.py       # surrogate generators
│       │   ├── tests.py        # statistical tests (RMT, jitter variants)
│       │   └── summaries.py    # helpers for packaging test results
│       ├── simulations/
│       │   ├── __init__.py
│       │   ├── poisson.py
│       │   ├── phase_locked.py
│       │   ├── transient.py
│       │   └── multifreq.py
│       ├── datasets/           # small synthetic fixtures (optional download helpers)
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── circular.py
│       │   └── nan_ops.py
│       ├── typing.py           # shared type aliases (ArrayLike, SpikeTrain, etc.)
│       └── _version.py         # generated via hatch-vcs or manual bump
├── tests/
│   ├── test_core.py
│   ├── test_stats.py
│   └── fixtures/               # deterministic spike/LFP samples
├── docs/
│   ├── index.md                # mkdocs or sphinx content
│   └── guides/                 # tutorials migrated from notebooks
└── tools/                      # optional scripts (formatting, lint, benchmarking)
```

## Public API Surface
- `pygpla.gpla` (`api.py`): high-level function returning dataclass `GPLAResult` (lfp_vec, spk_vec, gplv, stats, metadata). Wraps preprocessing + core compute.
- `pygpla.compute_coupling`, `pygpla.compute_gplv`, `pygpla.run_stat_test`: lower-level entry points mirroring MATLAB function parity for advanced users.
- Configuration dataclasses (e.g., `PreprocessingConfig`, `WhiteningConfig`, `StatTestConfig`) exposed via `pygpla.config`. Accept `numpy.ndarray` or array-like inputs with validation.
- Simulation utilities exported under `pygpla.simulations` for crafting tutorial datasets.
- Optional CLI entry point (`pygpla.__main__`) for batch processing (stretch goal).

## Configuration & Validation Strategy
- Centralize argument validation in `config.py` using dataclasses + `__post_init__` (or `pydantic` if dependency accepted) to enforce shapes, dtype, and semantics.
- Provide helper `ensure_complex_analytic(signal)` and `ensure_spike_matrix(...)` to sanitize MATLAB-style inputs.
- Standardize random seed handling (e.g., accept `np.random.Generator` instances or integer seeds across stats/simulations).
- Introduce consistent error hierarchy (`pygpla.errors`) if needed (e.g., `InvalidInputError`, `WhiteningError`).

## Dependency & Build Decisions
- Hard dependency: `numpy>=1.21` (align with supported Python versions, e.g., `>=3.9`).
- Optional extras:
  - `sim`: adds `scipy` if advanced simulations require Bessel or FFT utilities.
  - `docs`: `mkdocs-material` (or `sphinx`) for documentation builds.
  - `test`: `pytest`, `numpy.testing`, `hypothesis` (optional) for property tests.
- Build backend: `hatchling` (simple PEP 517 backend) + optional `hatch-vcs` for versioning.
- Formatting/QA: `ruff` (lint/format), `black` (if desired), `mypy` (once types added).

## Migration Mapping
- `src/methods/gpla_core.py` → `pygpla/core/gpla.py`
- `src/methods/tngpla.py` → `pygpla/api.py` (stats orchestration) + `pygpla/stats/tests.py`
- `src/methods/prep_SpkLfpData.py` → `pygpla/preprocessing/spike_lfp.py`
- `tncmpt_couplingMatrix.py` → `core/coupling.py`
- `fctrz_couplingMatrix.py` → `core/factorization.py`
- `tnstataliz_gPLV.py` + `jitter.py` → `stats/tests.py` + `stats/jitter.py`
- `whitenRed2.py` / `whitenRed4.py` → `core/whitening.py`
- `utilities/*` → `utils/*`
- `simulations/*` retained with simplified naming and parameter checking.

## Documentation Roadmap
- Quickstart: installation, minimal example using synthetic data.
- Guides: replicating MATLAB workflow, interpreting gPLV outputs, customization (whitening, normalization, stats).
- API reference: auto-generated from docstrings (Sphinx or MkDocs + `mkdocstrings`).
- Migration tips: mapping MATLAB script expectations to Python package usage.

## Open Questions / Follow-ups
- Decide whether to include real datasets or provide download helper pointing to external storage.
- Clarify whitening proportion default (`np.nan` or explicit float) and document behavior.
- Confirm naming conventions (retain MATLAB names for traceability vs. Pythonic renaming).
- Determine if CLI support is required for the initial release.
