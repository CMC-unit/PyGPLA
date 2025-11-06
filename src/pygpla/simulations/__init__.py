"""Synthetic data generators for PyGPLA."""

from .poisson import generate_homogeneous_poisson, generate_inhomogeneous_poisson
from .phase_locked import generate_phase_locked_dataset
from .transient import generate_transient_locked_dataset
from .multifreq import generate_multifrequency_dataset

__all__ = [
    "generate_homogeneous_poisson",
    "generate_inhomogeneous_poisson",
    "generate_phase_locked_dataset",
    "generate_transient_locked_dataset",
    "generate_multifrequency_dataset",
]
