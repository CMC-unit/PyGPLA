# Software Architecture

This page documents the current software architecture of PyGPLA and the design choices behind
the package layout.

## Package Structure

PyGPLA is organized as a focused, modular library under `src/pygpla`, with a high-level API and
separable numerical components.

### Top-level API (`pygpla.api`)

- **`gpla`**: main orchestration entrypoint used by most users.
  It runs preprocessing, core GPLA decomposition, and optional statistical testing.
- **`GPLAResult`**: dataclass returned by `gpla`, containing vectors, gPLV, p-value, stats, and metadata.

**Design rationale:** keep one stable, user-facing entrypoint while exposing lower-level pieces for
advanced workflows.

### Configuration Module (`pygpla.config`)

- **`PreprocessingConfig`**, **`WhiteningConfig`**, **`StatTestConfig`**:
  dataclass-based configuration objects for structured parameter control.
- **`validate_spike_trains`**, **`validate_lfp_signal`**:
  input-shape validation helpers.

**Design rationale:** support MATLAB-style flags for backward compatibility while providing a
clearer typed config layer for Python users.

### Preprocessing Module (`pygpla.preprocessing`)

- **`prepare_spike_lfp_data`**:
  trial concatenation, unit/time selection, spike-threshold filtering, optional whitening, and
  optional LFP normalization.

**Design rationale:** isolate input shaping and preprocessing from the coupling/SVD math.

### Core Module (`pygpla.core`)

- **`compute_coupling_matrix`** (`core.coupling`):
  builds complex spike-LFP coupling matrix with normalization options
  (`nSpk`, `nSpk-square-root`, `var1_theoretical`) and same-electrode correction hook.
- **`factorize_coupling_matrix`** (`core.factorization`):
  SVD with handling for NaN columns (e.g., units with zero spikes).
- **`run_gpla_core`** (`core.gpla`):
  computes GPLA vectors, phase rotation convention, and gPLV.
- **`apply_whitening`**, **`whitenRed2`**, **`whitenRed4`** (`core.whitening`):
  whitening utilities ported from the MATLAB workflow.

**Design rationale:** keep GPLA numerical primitives composable and testable in isolation.

### Statistics Module (`pygpla.stats`)

- **`run_statistical_test`**:
  wraps GPLA with significance testing.
- **Test modes**:
  - `RMT-based` (Marchenko-Pastur edge heuristic)
  - `spike-jittering` (surrogate-based p-values)
- **Jitter generators**:
  interval, ISI-preserved interval, group-preserved interval, and population jitter.

**Design rationale:** separate inferential logic from deterministic decomposition.

### Simulations Module (`pygpla.simulations`)

- **`generate_homogeneous_poisson`**, **`generate_inhomogeneous_poisson`**
- **`generate_phase_locked_spikes`**
- **`simulate_transient_locked`**

**Design rationale:** provide reproducible synthetic data generators for validation, testing, and
tutorials without coupling them to core inference code.

### Shared Typing (`pygpla.typing`)

- Shared aliases such as `ArrayLike`, `ComplexArray`, and spike train type helpers.

## Dependencies

### Core Dependency

- **NumPy >= 1.21**:
  core array operations, linear algebra, and numerical computations across all modules.

### Optional Dependencies

- **SciPy >= 1.9** (extra: `sim`):
  used in simulation and tutorial workflows that need filtering/Hilbert-style preprocessing.
- **Sphinx ecosystem** (extra: `docs`):
  documentation build stack (`sphinx`, `myst-parser`, `sphinx-rtd-theme`, `linkify-it-py`).
- **Pytest** (extra: `tests`):
  test execution.

## Design Choices

### MATLAB-Parity First, Python API Second

Many parameter names and algorithmic branches preserve MATLAB compatibility for migration safety.
At the same time, a Pythonic configuration layer and dataclass-based results are provided to
improve readability and maintainability.

### Function-Centric Architecture

PyGPLA currently uses a function-oriented architecture (instead of a large class hierarchy).
This keeps the numerical path explicit and lightweight:

`gpla -> prepare_spike_lfp_data -> run_statistical_test -> run_gpla_core`.

### Explicit Data Contracts

The package assumes:

- spikes as trial list of `(units, samples)` arrays
- analytic LFP as `(channels, samples, trials)` complex array

These contracts are enforced in preprocessing and validation utilities.

### Deterministic and Stochastic Paths

- Deterministic path: coupling matrix + SVD decomposition.
- Stochastic path: jitter surrogates and simulation modules.

The split helps users reason about reproducibility and runtime costs.

## Testing and Quality Assurance

- Test framework: **pytest**
- Current tests include:
  - transient simulation sanity/integration checks
  - unit tests for the coupling matrix, SVD factorization, whitening, spike-jitter
    surrogates, and input validation
- Lint tooling configuration is present via **Ruff** in `pyproject.toml`.

As the package matures, parity/regression tests against MATLAB reference outputs are the next
important QA step.

## Performance Considerations

- Uses NumPy vectorized matrix operations for coupling construction and SVD workflows.
- Trial concatenation is performed once in preprocessing to simplify downstream compute.
- Surrogate-based jitter testing is intentionally optional due to runtime cost.

## Version Management

PyGPLA currently uses an explicit package version:

- `project.version = "0.0.1"` in `pyproject.toml`
- `__version__ = "0.0.1"` in `src/pygpla/_version.py`

The Hatch setting `tool.hatch.version.path = "src/pygpla/_version.py"` keeps the package's
runtime version source in one place. Version bumps are therefore currently a deliberate,
manual release action rather than git-tag-driven automation.
