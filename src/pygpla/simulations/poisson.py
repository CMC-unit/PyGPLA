"""Poisson spike train simulators (placeholders)."""

from __future__ import annotations

import numpy as np

__all__ = ["generate_homogeneous_poisson", "generate_inhomogeneous_poisson"]


def generate_homogeneous_poisson(rate_hz: float, duration_s: float, sf: float, n_units: int, n_trials: int) -> np.ndarray:
    """Generate homogeneous Poisson spike trains."""

    raise NotImplementedError("Homogeneous Poisson simulator not ported yet.")


def generate_inhomogeneous_poisson(rate_profile, duration_s: float, sf: float, n_units: int, n_trials: int) -> np.ndarray:
    """Generate inhomogeneous Poisson spike trains."""

    raise NotImplementedError("Inhomogeneous Poisson simulator not ported yet.")
