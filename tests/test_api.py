"""Tests for the public GPLA API."""

import warnings

import numpy as np
import pytest

from pygpla.api import _stats_config_to_dict, gpla
from pygpla.config import StatTestConfig


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


def test_gpla_accepts_stat_test_config_for_rmt():
    spikes, lfp_analytic = _minimal_inputs()

    result = gpla(
        spikes,
        lfp_analytic,
        stats_config=StatTestConfig(test_type="RMT-based"),
        plvNrmlzMethed="var1_theoretical",
    )

    assert np.isnan(result.p_value)
    assert isinstance(result.stats["gPLV_stats"]["nullHypoReject"], bool)


def test_gpla_accepts_stat_test_config_for_spike_jittering():
    spikes, lfp_analytic = _minimal_inputs()
    config = StatTestConfig(
        test_type="spike-jittering",
        n_surrogates=2,
        jitter_window_width=0.01,
        sampling_frequency=1000.0,
        jitter_type="fake-jittering",
        alpha=0.01,
        additional_params={"rngSeed": 123, "SVspectrumStatsType": "default"},
    )

    result = gpla(
        spikes,
        lfp_analytic,
        stats_config=config,
        plvNrmlzMethed="nSpk",
    )

    assert result.p_value == pytest.approx(0.0)
    assert result.stats["gPLV_stats"]["nullDistribution"].shape == (2,)


def test_stat_test_config_maps_all_fields_to_legacy_keys():
    config = StatTestConfig(
        test_type="spike-jittering",
        n_surrogates=25,
        jitter_window_width=0.02,
        sampling_frequency=500.0,
        jitter_type="group-preserved-interval-jittering",
        alpha=0.01,
        additional_params={
            "rngSeed": 42,
            "SVspectrumStatsType": "RMT-heuristic",
            "testType": "ignored-in-favor-of-named-field",
        },
    )

    assert _stats_config_to_dict(config) == {
        "testType": "spike-jittering",
        "nJtr": 25,
        "jitterWinWidth": 0.02,
        "spkSF": 500.0,
        "jitterType": "group-preserved-interval-jittering",
        "alphaValue": 0.01,
        "rngSeed": 42,
        "SVspectrumStatsType": "RMT-heuristic",
    }
