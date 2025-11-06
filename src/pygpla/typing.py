"""Shared typing helpers."""

from __future__ import annotations

from typing import NewType, Tuple

import numpy as np
from numpy.typing import ArrayLike as _ArrayLike

ArrayLike = _ArrayLike
ComplexArray = np.ndarray
SpikeTrain = NewType("SpikeTrain", np.ndarray)
SpikeTrainSet = Tuple[SpikeTrain, ...]

__all__ = ["ArrayLike", "ComplexArray", "SpikeTrain", "SpikeTrainSet"]
