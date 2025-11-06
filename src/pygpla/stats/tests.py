"""Statistical tests for GPLA outputs."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..typing import ArrayLike

__all__ = ["run_statistical_test"]


def run_statistical_test(
    spike_trains: np.ndarray,
    lfp_signal: np.ndarray,
    *,
    config: Dict[str, Any] | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run statistical significance tests on GPLA metrics.
    """

    raise NotImplementedError("Statistical testing not ported yet.")
