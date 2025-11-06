# PyGPLA Migration & Verification Plan

## Migration Phases

1. **Core Math Port**
   - Move `gpla_core.py`, `tncmpt_couplingMatrix.py`, and `fctrz_couplingMatrix.py` into `pygpla/core/`.
   - Implement column normalization, phase rotation, and same-electrode correction.
   - Add precise docstrings with MATLAB parity notes and shape expectations.
   - Introduce lightweight input validation (`np.asarray` conversions, finite checks).

2. **Preprocessing Pipeline**
   - Port `prep_SpkLfpData.py`, `whitenRed2.py`, and `whitenRed4.py` into `preprocessing` and `core/whitening`.
   - Replace flag-based interfaces with structured configs (`PreprocessingConfig`, `WhiteningConfig`) but preserve compatibility shims.
   - Add unit tests using synthetic spike/LFP arrays to validate selection, whitening, and unit subsetting.

3. **Statistics Layer**
   - Port `tnstataliz_gPLV.py` and `jitter.py` into `stats/tests.py` and `stats/jitter.py`.
   - Normalize RNG seeding and expose unified `StatTestConfig`.
  - Validate jitter routines with deterministic seeds and shape assertions.
   - Implement RMT heuristic and jitter-based p-value computations with numpy broadcasting.

4. **Simulation Utilities**
   - Port `simulations/*.py` into `pygpla/simulations/`, ensuring parameter validation and optional RNG injection.
   - Provide small synthetic fixtures for tests and documentation examples.

5. **Utilities & Supporting Functions**
   - Port `utilities/nan_utils.py` and `utilities/circular_stats.py` into `pygpla/utils/`.
   - Integrate with stats/core where needed; add docstrings and tests.

6. **API Wiring**
   - Implement `pygpla.api.gpla` to orchestrate preprocessing → core → stats.
   - Expose low-level entry points (`compute_coupling_matrix`, `run_statistical_test`) at top level for advanced users.
   - Add configuration validation and helpful error messages.

## Testing Strategy

- **Unit Tests**
  - Per-module tests in `tests/` (e.g., `test_core.py`, `test_preprocessing.py`, `test_stats.py`).
  - Use deterministic synthetic data (from `simulations`) to test gPLV outputs against known MATLAB results (build JSON/NPY fixtures).
  - Cover edge cases: zero spikes, NaNs, same-electrode mapping, whitening variations.

- **Integration Tests**
  - End-to-end pipeline test using transient/multifrequency simulations to verify gPLV magnitude and vector alignment.
  - Statistical test regression: ensure jitter p-values and RMT rejection flag match reference outputs.

- **Numerical Regression**
  - Store reference outputs (npz) from MATLAB for critical scenarios; implement tolerance-based comparisons (`np.testing.assert_allclose`).

- **Static Analysis & QA**
  - Enable Ruff linting (style/import checks) and, optionally, `black` formatting.
  - Introduce `mypy` after adding type hints to core modules.

## Documentation Tasks

- Update `docs/index.md` with installation instructions and basic quickstart once API is stable.
- Create guides under `docs/guides/`:
  - MATLAB-to-Python migration walkthrough.
  - Understanding whitening/normalization options.
  - Significance testing (RMT vs jitter).
- Auto-generate API reference using `mkdocs` + `mkdocstrings`.
- Maintain changelog once versioning strategy is in place (consider `CHANGELOG.md`).

## Release & Maintenance Checklist

- Configure CI (GitHub Actions) to run lint + tests on Python 3.9–3.12.
- Decide on versioning (semantic versioning starting at `0.1.0` once ready).
- Prepare PyPI publishing workflow (`hatch build`, `hatch publish`).
- Document contribution guidelines and code style expectations.
