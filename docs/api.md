# API reference

These entries are generated with Sphinx autodoc/autosummary.

## High-level analysis

```{autosummary}

pygpla.api.gpla
pygpla.api.GPLAResult
```

## Configuration

```{autosummary}

pygpla.config.PreprocessingConfig
pygpla.config.WhiteningConfig
pygpla.config.StatTestConfig
pygpla.config.validate_spike_trains
pygpla.config.validate_lfp_signal
```

## Core routines

```{autosummary}

pygpla.core.coupling.compute_coupling_matrix
pygpla.core.factorization.factorize_coupling_matrix
pygpla.core.gpla.run_gpla_core
pygpla.core.whitening.apply_whitening
pygpla.core.whitening.whitenRed2
pygpla.core.whitening.whitenRed4
```

## Preprocessing

```{autosummary}

pygpla.preprocessing.spike_lfp.prepare_spike_lfp_data
```

## Statistics

```{autosummary}

pygpla.stats.tests.run_statistical_test
pygpla.stats.jitter.interval_jitter
pygpla.stats.jitter.isi_preserved_jitter
pygpla.stats.jitter.group_preserved_jitter
pygpla.stats.jitter.population_jitter
```

## Simulations

```{autosummary}

pygpla.simulations.phase_locked.generate_phase_locked_spikes
pygpla.simulations.poisson.generate_homogeneous_poisson
pygpla.simulations.poisson.generate_inhomogeneous_poisson
pygpla.simulations.transient.simulate_transient_locked
```
