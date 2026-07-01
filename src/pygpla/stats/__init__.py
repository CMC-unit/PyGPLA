"""Statistical testing and surrogate generation."""

from .jitter import (
    group_preserved_jitter,
    interval_jitter,
    isi_preserved_jitter,
    population_jitter,
)
from .tests import run_statistical_test

__all__ = [
    "interval_jitter",
    "isi_preserved_jitter",
    "group_preserved_jitter",
    "population_jitter",
    "run_statistical_test",
]
