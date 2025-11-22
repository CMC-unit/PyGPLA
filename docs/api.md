# API reference

These entries are generated with Sphinx autodoc/autosummary. Rebuild the docs (`sphinx-build -b html docs docs/_build/html`) to refresh the stubs under `_autosummary/`.

## High-level analysis

```{autosummary}
:toctree: _autosummary
:caption: High-level API

pygpla.api.gpla
pygpla.api.GPLAResult
```

## Configuration

```{autosummary}
:toctree: _autosummary
:caption: Configuration

pygpla.config.PreprocessingConfig
pygpla.config.WhiteningConfig
pygpla.config.StatTestConfig
pygpla.config.validate_spike_trains
pygpla.config.validate_lfp_signal
```

## Core routines

```{autosummary}
:toctree: _autosummary
:caption: Core math

pygpla.core.coupling.compute_coupling_matrix
pygpla.core.factorization.factorize_coupling_matrix
pygpla.core.gpla.run_gpla_core
pygpla.core.whitening.apply_whitening
pygpla.core.whitening.whitenRed2
pygpla.core.whitening.whitenRed4
```

## Preprocessing

```{autosummary}
:toctree: _autosummary
:caption: Preprocessing

pygpla.preprocessing.spike_lfp.prepare_spike_lfp_data
```

## Statistics

```{autosummary}
:toctree: _autosummary
:caption: Statistical testing

pygpla.stats.tests.run_statistical_test
pygpla.stats.jitter.interval_jitter
pygpla.stats.jitter.isi_preserved_jitter
pygpla.stats.jitter.group_preserved_jitter
pygpla.stats.jitter.population_jitter
pygpla.stats.summaries.summarize_gplv_results
pygpla.stats.summaries.summarize_plv_matrix
```

## Simulations

```{autosummary}
:toctree: _autosummary
:caption: Simulations

pygpla.simulations.phase_locked.generate_phase_locked_spikes
pygpla.simulations.poisson.generate_homogeneous_poisson
pygpla.simulations.poisson.generate_inhomogeneous_poisson
pygpla.simulations.transient.simulate_transient_locked
```
