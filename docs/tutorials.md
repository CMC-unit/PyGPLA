# Tutorials

This page provides a hands-on, end-to-end walkthrough of the simulations and analysis used to reproduce *Figure 2* from **Safavi et al. (2023)**. The implementation is available in `paper/Figures/figure2.py`.

The goals are twofold:

1. Reproduce the *Figure 2* output (`paper/Figures/figure2_python.png`).
2. Understand the key components of the GPLA pipeline:  
   simulation → analytic LFP construction → coupling matrix / SVD → (optional) statistics → plotting and interpretation.

:::{note}
This tutorial is written for Sphinx + MyST (`myst_parser`). It shows code as static snippets
and does **not** execute code during the docs build.
:::

## What you will build

`paper/Figures/figure2.py` generates four synthetic “transient coupling” scenarios:

- **Model 1 (global synchrony)**: all units lock to the same LFP phase during transient events.
- **Model 2 (phase gradient)**: units have systematically different preferred phases.
- **Model 3 (clusters)**: units form a few phase-locked subpopulations.
- **Model 4 (null)**: no spike–LFP coupling (control condition).

For each model, we:

- simulate an LFP trace and spike trains with *transient* coupling windows,
- compute an analytic LFP (bandpass + Hilbert),
- run GPLA via `{py:func}`pygpla.api.gpla``,
- visualize gPLV and the spike vector geometry.

## Prerequisites

### Install dependencies

This repository’s runtime dependency is only NumPy, but Figure 2 requires plotting and
signal processing libraries.

From the repo root:

```bash
pip install -e ".[sim,docs]"
pip install matplotlib
```

- `.[sim]` provides SciPy (needed for bandpass filtering + Hilbert transform).
- `matplotlib` is required to render and save the figure.

:::{note}
If you only want to **build the docs**, you typically only need `.[docs]`. The figure script
is separate and can be run in an environment that has `matplotlib` and `scipy`.
:::

### Quick reproduction (one command)

Run the script directly:

```bash
python paper/Figures/figure2.py
```

It saves:

- `paper/Figures/figure2_python.png`

The rest of this page explains *how* that script works.

## Script overview

At a high level, `paper/Figures/figure2.py` has four layers:

1. **Setup**: imports, path handling for `src/`, Matplotlib config.
2. **Helpers**: choose plotting windows, compute analytic LFP, and helper plots.
3. **Simulation + GPLA loop**: generate data for each model and run `{py:func}`pygpla.api.gpla``.
4. **Figure assembly**: build the multi-panel layout and save the PNG.

### Setup: imports, `sys.path`, and Matplotlib cache

The figure script is designed to be runnable from the repo even if the package is not
installed. It adds `src/` to the Python path and sets a local Matplotlib cache directory.

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 1-45
```

Key things to notice:

- `ROOT = Path(__file__).resolve().parents[1]` points to `paper/`.
- `SRC = ROOT / "src"` makes imports like `from pygpla.api import gpla` work.
- `MPLCONFIGDIR` is redirected to `paper/.matplotlib_cache` to avoid permission issues on
  shared filesystems.

## Parameters: what is being simulated?

The simulation is configured with two types of parameters:

- **Signal geometry**: number of channels/units/trials, sampling frequency, duration.
- **Dynamics**: oscillation frequency, transient window length, coupling strength, etc.

### Signal (recording) parameters

`SignalParams` defines the default “recording” shape used across all models:

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 48-63
```

In the default figure:

- `nCh=1` LFP channel (a single LFP trace),
- `nUnit=18` spiking units,
- `nTr=10` trials,
- `SF=1000 Hz` sampling frequency,
- `signalLength=10 s` per trial.

### Global simulation parameters (LFP + events)

Inside `main()`, `global_params` configures the transient oscillatory LFP:

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 129-149
```

The key fields:

- `oscFreq`: center oscillation frequency (used for bandpass and coupling timing).
- `nCycl`: transient duration in **cycles**; duration in seconds is `nCycl / oscFreq`.
- `syncSigProportion`: controls how many transient events occur per trial.
- `lfpPhaseNoise_kappa`: von Mises concentration controlling phase noise in the LFP.
- `whiteNoise_sigma`: additive background noise level.

### Coupling models (spike preferred phases)

The four models differ in two per-unit parameters:

- `lockingStrength_kappa` (coupling strength; `0` means no locking),
- `lockingPhase` (each unit’s preferred phase).

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 151-190
```

Interpretation:

- **Model 1**: identical phases \u2192 spike vector phases should cluster tightly.
- **Model 2**: linearly increasing phases \u2192 spike vector phases should spread out.
- **Model 3**: three phase clusters \u2192 spike vector phases should form clusters.
- **Model 4**: `kappa=0` \u2192 gPLV should drop relative to the coupled models.

## Helper functions: where the “important” work happens

### 1) Constructing an analytic LFP (bandpass + Hilbert)

GPLA expects a *complex analytic* LFP (or a phase representation that can be converted to
complex form). In real data, you usually:

1. bandpass filter around a target frequency,
2. apply a Hilbert transform to get the complex analytic signal.

That’s exactly what `_bandpass_and_analytic()` does:

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 88-108
```

Why this matters:

- The coupling matrix used by GPLA is (conceptually) a sum of analytic LFP values at spike
  times, so *phase* and *amplitude* handling depends on how you construct `lfp_analytic`.
- In the figure script, the simulator returns a real-valued LFP trace, and the analytic signal
  is derived “like real preprocessing” to make the tutorial realistic.

:::{tip}
If you already have an analytic LFP array shaped `(channels, samples, trials)`, you can skip
this entire step and pass it directly into `{py:func}`pygpla.api.gpla``.
:::

### 2) Choosing the plotting window around a transient event

The simulator returns `meta["event_trains"]` (a list of transient start times per trial). The
script uses that metadata to choose a short window for plotting:

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 66-86
```

This function is purely for visualization convenience. GPLA itself is run on the full trial
concatenation (unless you explicitly window the data).

### 3) Plotting helpers: schematic and spike-vector polar plot

Two small helpers are used to mimic the style of the paper figure:

- `_plot_model_schematic()` draws a cartoon of the coupling pattern.
- `_polar_spkvec()` plots each unit’s spike-vector coefficient in polar coordinates
  (angle = phase, radius = magnitude).

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 111-127
```

The polar plot is especially important for interpretation:

- **phase** of `spike_vector[u]` reflects timing of unit `u` relative to the global LFP pattern,
- **magnitude** reflects how strongly that unit participates in the dominant coupling mode.

## The core loop: simulate → preprocess → run GPLA

### Simulate transient data for one model

The transient simulation generator is `{py:func}`pygpla.simulations.transient.simulate_transient_locked``.
It returns:

- `lfp_real`: real-valued LFP trace `(nCh, nBins, nTr)`
- `spikes`: list of per-trial spike matrices, each `(nUnit, nBins)`
- `meta`: event timings and transient duration

The figure script calls it with `return_analytic=False` because it wants to compute the
analytic LFP via bandpass/Hilbert (previous section):

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 206-246
```

### Run GPLA: `pygpla.api.gpla`

The GPLA call in the script is:

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 248-292
```

The most important parameters here are:

- `spikeTrains_raw`: a list of trials, each shaped `(units, samples)`.
- `lfpPhases_raw`: analytic LFP shaped `(channels, samples, trials)`.
- `plvNrmlzMethed="var1_theoretical"`:
  uses a phase-only analytic representation internally (see
  `{py:func}`pygpla.core.coupling.compute_coupling_matrix``) and applies spike-count scaling.
- `iSV=1`: analyze the **first** singular mode (the dominant coupling pattern).
- `stats_config` (named `statTestInfo` in the script): optional surrogate testing settings.

#### What GPLA computes (in code terms)

Under the hood, `{py:func}`pygpla.api.gpla`` does:

1. `{py:func}`pygpla.preprocessing.spike_lfp.prepare_spike_lfp_data``:
   concatenates trials into long matrices (time axis), optionally whitens LFPs.
2. `{py:func}`pygpla.stats.tests.run_statistical_test``:
   runs the core GPLA (`{py:func}`pygpla.core.gpla.run_gpla_core``) and optionally computes
   significance via RMT or spike-jitter surrogates.

The **core math** is:

- build coupling matrix `C` via `{py:func}`pygpla.core.coupling.compute_coupling_matrix``,
- compute SVD via `{py:func}`pygpla.core.factorization.factorize_coupling_matrix``,
- set a phase convention by rotating vectors so the mean LFP-vector phase is zero.

### About the surrogate settings in Figure 2

The figure script sets:

```python
"testType": "spike-jittering",
"nJtr": 1,
```

This is intentionally *fast* and mainly exercises the code path; it is **not** a meaningful
p-value estimate.

:::{important}
For real analysis, use a larger `nJtr` (e.g. 200–2000+) or use `"testType": "RMT-based"`
when its assumptions are appropriate for your data.
:::

### Sign/phase ambiguity and the “flip”

SVD-based decompositions are only unique up to a global complex phase. The GPLA core
already sets a phase convention, but for plotting the script also flips the sign if the (real)
sum of the LFP vector is negative:

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 294-303
```

This is purely to keep the plotted vectors visually consistent across runs.

## Figure assembly: how the panels are built

After the script computes `gplv_vals`, `spk_vecs`, example LFP traces, and raster snippets, it
builds a multi-row layout:

- **Top row**: example LFP trace + analysis window, and a bar plot of gPLV for all four models
- **Rows 2–5**: model schematic, LFP+spikes snippet, spike-vector polar plot

The plotting code is long but straightforward: it pulls from lists built in the loop and plots
them into fixed subplot positions.

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 305-356
```

Finally, the figure is saved into `paper/Figures/`:

```{literalinclude} ../paper/Figures/figure2.py
:language: python
:linenos:
:lines: 357-365
```

## Most important takeaways

If you only remember a few things from this tutorial, make them these:

1. **Input shapes matter**:
   - spikes: list of `(units, samples)` arrays (one per trial),
   - analytic LFP: `(channels, samples, trials)` complex array.
2. **GPLA is a low-rank summary** of *all* spike–LFP couplings via SVD of a coupling matrix.
3. **gPLV is the leading singular value**; the spike/LFP vectors are the corresponding
   singular vectors (with a phase convention).
4. **Stats are optional** and depend on your null hypothesis:
   RMT-based is fast; spike-jittering is flexible but can be expensive.

## Where to go next

- Read the high-level API docs: `{py:func}`pygpla.api.gpla`` and `{py:class}`pygpla.api.GPLAResult``.
- Explore simulations: `{py:mod}`pygpla.simulations`` (especially transient and phase-locked).
- If you want a “real data” tutorial, tell me:
  - your spike/LFP data format,
  - your frequency bands and sampling rate,
  - whether you want sliding windows or full-trial coupling,
  and I can draft a dedicated guide under `docs/guides/`.
