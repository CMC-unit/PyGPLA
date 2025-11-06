# PyGPLA_dev Code Audit

## Repository Scope
- MATLAB-to-Python port for Generalized Phase Locking Analysis (GPLA) workflows.
- Core logic lives under `src/` with exploratory notebooks, reports, and datasets beside it.
- Temporary experiments (`tmp/`) are explicitly out of scope for packaging.

## Core Computational Modules (`src/methods`)
- `tngpla.py`: high-level orchestration (data prep → core GPLA → stats). Handles unit selection, whitening, same-electrode corrections, and optional surrogate tests. Returns GPLA vectors, gPLV, stats, and metadata.
- `prep_SpkLfpData.py`: concatenates spike trains/LFPs across trials, enforces unit and temporal selections, applies PCA whitening (`whitenRed2/4`), and optional LFP normalization.
- `tncmpt_couplingMatrix.py`: builds complex coupling matrix with normalization options (`nSpk`, `nSpk-square-root`, `var1_theoretical`) and optional jitter-based same-electrode correction.
- `gpla_core.py`: wraps coupling construction, SVD factorization (`fctrz_couplingMatrix`), column normalization, phase rotation, and gPLV computation.
- `tnstataliz_gPLV.py`: statistical testing layer. Implements RMT heuristic and multiple spike-jittering variants (interval, ISI-preserved, group, population).
- `jitter.py`: collection of surrogate generators mirroring MATLAB interval-jitter routines.
- `whitenRed2.py` / `whitenRed4.py`: PCA whitening utilities (channels × samples and per-trial variants).

## Supporting Packages
- `src/utilities`: numerical helpers (`circular_stats.ml_est_kappa_from_plv`, `nan_utils.remove_nan_in_matrix`).
- `src/simulations`: synthetic data generators (Poisson, sustained/transient phase locking, multifrequency coupling, 2D neural field) for validation and tutorials.
- `src/data`: placeholder for exploratory, processed, and simulation datasets (no Python code, but informs packaging of sample assets).
- `src/exptools`, `src/packages`, `src/misc`: contain README stubs or exploratory scaffolding; no Python modules yet.

## Dependencies & Environment
- Runtime dependency: `numpy` (used across all Python modules).
- No `pyproject.toml`, `setup.cfg`, or requirements files — packaging metadata and dependency pinning still need to be authored.
- No automated tests, CI configuration, or formatting/type-checking tools present.

## Known Gaps & TODOs (from inline notes)
- `tnstataliz_gPLV`: cumulative SV spectrum test type not implemented.
- `prep_SpkLfpData`: whitening proportion handling (`nan` default) flagged for clarification.
- General: expand documentation/comments and add unit/integration tests (not yet started).

## Implications for the Final Package
- Need to decide which simulation utilities and datasets ship with the library versus remaining as developer resources.
- Establish a clean `pygpla/` package namespace with clear API exports (likely wrapping `tngpla`, `gpla_core`, stats, and simulation helpers).
- Design packaging metadata (`pyproject.toml`), dependency management, testing strategy, and documentation structure before code migration.
