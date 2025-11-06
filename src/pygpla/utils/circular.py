"""Circular statistics helpers."""

from __future__ import annotations

import numpy as np

__all__ = ["ml_estimate_kappa"]


def ml_estimate_kappa(plv: float, n: int) -> float:
    """
    Maximum-likelihood-inspired estimate of concentration parameter kappa.

    Mirrors the MATLAB logic; actual implementation will be ported from PyGPLA_dev.
    """

    raise NotImplementedError("Circular stats helper not ported yet.")
