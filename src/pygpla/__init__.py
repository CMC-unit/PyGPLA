"""
PyGPLA: Generalized Phase Locking Analysis tools for Python.

GPLA summarizes multichannel spike-LFP coupling by building a complex coupling
matrix and extracting its dominant low-rank structure via SVD, yielding a scalar
coupling strength (gPLV) together with an LFP vector and a spike vector.

The main entry point is :func:`pygpla.api.gpla`::

    from pygpla.api import gpla

    result = gpla(spikes, lfp_analytic, stats_config={"testType": "RMT-based"})
    result.gplv, result.lfp_vector, result.spike_vector

Modules
-------
api
    High-level :func:`~pygpla.api.gpla` entry point and its
    :class:`~pygpla.api.GPLAResult` container.
core
    Coupling-matrix construction, SVD factorization, and PCA whitening.
preprocessing
    Trial concatenation, spike-count filtering, and related data preparation.
simulations
    Phase-locked, transient-coupling, and Poisson spike-train generators.
stats
    RMT-based and spike-jitter surrogate significance testing.
config
    Dataclasses describing whitening, preprocessing, and statistical-test options.

The method is described in Safavi et al. (2023), *Uncovering the organization of
neural circuits with Generalized Phase Locking Analysis*, PLOS Computational Biology.
"""

from ._version import __version__

__all__ = ["__version__"]
