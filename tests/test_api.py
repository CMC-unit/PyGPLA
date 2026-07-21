"""Tests for the public GPLA API."""

import warnings

import numpy as np
import pytest

from pygpla.api import gpla


def _minimal_inputs():
    spikes = [np.array([[0, 1, 0, 0]], dtype=int)]
    lfp_analytic = np.ones((1, 4, 1), dtype=complex)
    return spikes, lfp_analytic


def test_gpla_warns_that_real_lfp_is_interpreted_as_phase():
    spikes, lfp_analytic = _minimal_inputs()
    lfp_phase = np.angle(lfp_analytic)

    with pytest.warns(UserWarning, match="phase angles in radians, not as raw LFP voltage"):
        result = gpla(spikes, lfp_phase, plvNrmlzMethed="nSpk")

    assert np.isfinite(result.gplv)


def test_gpla_does_not_warn_for_complex_analytic_lfp():
    spikes, lfp_analytic = _minimal_inputs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gpla(spikes, lfp_analytic, plvNrmlzMethed="nSpk")

    assert not caught
    assert np.isfinite(result.gplv)
