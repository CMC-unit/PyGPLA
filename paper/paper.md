---
title: "PyGPLA: A Python Toolbox for Generalized Phase Locking Analysis"
tags:
  - Python
  - neuroscience
  - spike-field coupling
  - local field potential
  - multivariate analysis
  - singular value decomposition
authors:
  - name: Amir Khani
    orcid: 0009-0008-8042-9590
    corresponding: true
    affiliation: 1
  - name: Shervin Safavi
    orcid: 0000-0002-2868-530X
    corresponding: true
    affiliation: "2, 3"
affiliations:
  - name: Donders Institute for Brain, Cognition, and Behaviour, Radboud University, Nijmegen, Netherlands
    index: 1
    ror: "016xsfp80"
  - name: Computational Neuroscience, Department of Child and Adolescent Psychiatry, Faculty of Medicine, Technische Universität Dresden, Dresden 01307, Germany
    index: 2
    ror: "042aqky30"
  - name: Department of Computational Neuroscience, Max Planck Institute for Biological Cybernetics, Tübingen 72076, Germany
    index: 3
    ror: "026nmvv73"

date: "2026-03-22"
bibliography: paper.bib
repository: "https://github.com/shervinsafavi/pygpla"
crossref: true
url: "https://pygpla.readthedocs.io/en/latest/"

---

## Summary

PyGPLA is a Python implementation of Generalized Phase Locking Analysis (GPLA) for multivariate spike-LFP coupling analysis [@safavi2023uncovering]. For each frequency, GPLA constructs a complex coupling matrix $\hat{C}(f) \in \mathbb{C}^{N_c \times N_u}$ between LFP channels and spike units, then applies singular value decomposition (SVD). The leading singular value (gPLV) summarizes population-level coupling strength, while the corresponding singular vectors describe dominant LFP and spike coupling modes. PyGPLA provides a full analysis workflow including LFP preprocessing, coupling-matrix construction, SVD-based decomposition, and statistical significance testing across frequency bands.


## Statement of need

Neural recordings are becoming increasingly high-dimensional and multimodal, demanding more sophisticated analysis tools. Simultaneous analysis of spiking activity and local field potentials (LFP) is among the most informative multi-modal approaches in systems neuroscience, providing insight into the multi-scale mechanisms underlying cognitive functions such as attention and memory [@buzsaki2012origin; @einevoll2013modelling; @herreras2016local]. LFP oscillatory activity partly reflects subthreshold processes shared by neuronal ensembles, and the synchronization between this activity and spiking is hypothesized to coordinate neural populations during cognition [@buzsaki2012origin; @hagen2016hybrid].

However, commonly used spike-field coupling measures are pairwise, making them suboptimal for modern multichannel recording systems [@zeitler2006assessing; @vinck2010pairwise; @vinck2012improved; @jiang2015measuring; @li2016unbiased; @zarei2018introducing]. As contemporary electrophysiological techniques allow simultaneous recordings from hundreds or thousands of sites, pairwise analyses generate high-dimensional matrices whose size grows rapidly with the number of channels, limiting interpretable extraction of large-scale collective dynamics [@dickey2009single; @jun2017fully; @juavinett2019chronically; @buzsaki2004large].

GPLA was developed to address these challenges by providing an efficient multivariate framework together with statistical routines for characterizing spike-field coupling at the population level [@safavi2023uncovering]. However, the original algorithm was implemented in MATLAB, limiting its accessibility. PyGPLA bridges this gap as an open-source Python implementation, aligning with the extensive use of Python in neuroscience as solidified by libraries such as MNE-Python [@GramfortEtAl2013a], Nilearn [@Nilearn], and TranCIT [@Nouri2025].

Existing spike–field coupling methods - including the phase-locking value (PLV), pairwise phase consistency (PPC) [@vinck2010pairwise], and spike-field coherence - operate on individual spike–LFP channel pairs. While informative for small-scale recordings, these pairwise approaches face combinatorial scaling with modern high-density probes, and do not directly capture the population-level structure of coupling. GPLA addresses this gap by providing a single multivariate decomposition that summarizes all spike–LFP interactions simultaneously, analogous to how PCA summarizes covariance structure. To our knowledge, no other Python package implements this multivariate approach to spike–field coupling analysis.

## Functionality

### Core analysis pipeline

PyGPLA provides an integrated solution for multivariate spike–field coupling analysis, including:

- **LFP preprocessing:** Band-pass filtering, Hilbert transform for analytic signal extraction, and optional reduced-rank whitening to decorrelate channels while avoiding noise amplification [@chavez2006proper].
- **Coupling-matrix construction:** Assembly of the complex-valued coupling matrix $\widehat{\mathbf{C}}(f) \in \mathbb{C}^{N_c \times N_u}$, where each entry sums the analytic LFP evaluated at all spike times of a given unit.
- **SVD-based decomposition:** Extraction of the gPLV (leading singular value) and associated LFP and spike spatial vectors, with rotational phase alignment, unwhitening of LFP vectors, and spike-vector rescaling.
- **Statistical testing:** Significance assessment via two complementary approaches: (1) surrogate-based testing using multiple spike-jittering schemes, and (2) an analytical test based on Marchenko–Pastur Random Matrix Theory (RMT) [@anderson2010random].
- **Simulation tools:** Synthetic data generators for phase-locked and transient-coupling scenarios, supporting reproducibility and method validation.

### Normalization options

The package supports two normalization modes for the coupling matrix: PLV-type normalization ($1/N_m^{\mathrm{tot}}$) for phase-only analysis, and unit-variance normalization ($1/\sqrt{N_m^{\mathrm{tot}}}$) required for the analytical RMT-based significance test.

## Example

We illustrate PyGPLA on synthetic transient-coupling simulations generated with the script `paper/figures/figure2.py`. As shown in \autoref{fig:gpla_results}, the figure compares four coupling models and reports their recovered gPLV values, together with representative LFP and spike-train windows and the associated spike-vector structure.

![Illustrative PyGPLA simulation example produced by `paper/figures/figure2.py`. Panel A shows an LFP trace with the analysis window. Panel B shows gPLV values across four simulated coupling models (M1-M4). Panels C-F display, for each model, a schematic, LFP plus spike-train segment, and the recovered spike vector in polar coordinates.](figures/figure2_python.png){#fig:gpla_results}

Code snippets and detailed instructions for reproducing these results are available in the package documentation and example scripts in the repository.

## Implementation details

The `pygpla` package is distributed under the BSD-3-Clause license. PyGPLA follows a layered architecture separating preprocessing (`pygpla.preprocessing`), core computation (`pygpla.core`), statistical testing (`pygpla.stats`), simulation utilities (`pygpla.simulations`), and configuration (`pygpla.config`). The high-level API (`pygpla.api.gpla`) orchestrates the full pipeline in a single call, keeping default usage compact while preserving access to lower-level components for advanced analyses. The package is built on NumPy [@harris2020array] and SciPy [@virtanen2020fundamental], and includes a `pytest` test suite with continuous integration via GitHub Actions.

## Acknowledgments

We acknowledge contributions from collaborators who provided feedback during the development of PyGPLA. Shervin Safavi acknowledges the support from the Max Planck Society and an add-on fellowship from the Joachim Herz Foundation.

## References
