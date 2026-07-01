"""Unit tests for input validation helpers (pygpla.config)."""

import numpy as np
import pytest

from pygpla.config import validate_lfp_signal, validate_spike_trains


def test_validate_spike_trains_accepts_consistent_trials():
    trials = [np.zeros((4, 100), dtype=int) for _ in range(3)]
    # Should not raise.
    validate_spike_trains(trials)


def test_validate_spike_trains_rejects_empty():
    with pytest.raises(ValueError, match="At least one"):
        validate_spike_trains([])


def test_validate_spike_trains_rejects_unit_mismatch():
    trials = [np.zeros((4, 100), dtype=int), np.zeros((5, 100), dtype=int)]
    with pytest.raises(ValueError, match="same number of units"):
        validate_spike_trains(trials)


def test_validate_spike_trains_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="2D"):
        validate_spike_trains([np.zeros(100, dtype=int)])


def test_validate_lfp_signal_returns_array_for_3d():
    lfp = np.ones((2, 50, 3), dtype=complex)
    out = validate_lfp_signal(lfp)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 50, 3)


def test_validate_lfp_signal_rejects_non_3d():
    with pytest.raises(ValueError, match="3D"):
        validate_lfp_signal(np.ones((2, 50)))
