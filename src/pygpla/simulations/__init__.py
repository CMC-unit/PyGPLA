"""Synthetic data generators for PyGPLA."""

from .poisson import generate_homogeneous_poisson, generate_inhomogeneous_poisson
from .phase_locked import generate_phase_locked_spikes
from .transient import simulate_transient_locked

__all__ = [
    "generate_homogeneous_poisson",
    "generate_inhomogeneous_poisson",
    "generate_phase_locked_spikes",
    "simulate_transient_locked",
]
