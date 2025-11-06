"""Helpers to summarize GPLA statistical outputs."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

__all__ = ["summarize_gplv_results", "summarize_plv_matrix"]


def summarize_gplv_results(gplv_values: Iterable[float]) -> Dict[str, float]:
    """Compute summary statistics over GPLA values."""

    values = np.asarray(list(gplv_values), dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(values.mean()), "std": float(values.std())}


def summarize_plv_matrix(plv_matrix: np.ndarray) -> Dict[str, float]:
    """Summaries for PLV matrices (placeholder)."""

    if plv_matrix.size == 0:
        return {"max": float("nan"), "min": float("nan")}
    return {
        "max": float(np.nanmax(plv_matrix)),
        "min": float(np.nanmin(plv_matrix)),
    }
