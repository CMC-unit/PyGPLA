"""Unit tests for the coupling-matrix construction (pygpla.core.coupling)."""

import numpy as np
import pytest

from pygpla.core.coupling import compute_coupling_matrix


def test_coupling_matrix_shape_and_spike_counts():
    n_units, n_channels, n_samples = 4, 3, 200
    rng = np.random.default_rng(0)
    spikes = (rng.random((n_units, n_samples)) < 0.1).astype(int)
    lfp = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(n_channels, n_samples)))

    coupling, spike_counts = compute_coupling_matrix(spikes, lfp)

    assert coupling.shape == (n_channels, n_units)
    assert np.iscomplexobj(coupling)
    np.testing.assert_array_equal(spike_counts, spikes.sum(axis=1))


def test_perfectly_phase_locked_magnitude_matches_sqrt_nspk():
    # Analytic LFP with constant zero phase: coupling magnitude should equal
    # sqrt(n_spikes) under the square-root normalization, with phase ~ 0.
    n_samples = 500
    lfp = np.ones((1, n_samples), dtype=complex)  # phase 0 everywhere
    spikes = np.zeros((1, n_samples), dtype=int)
    spikes[0, ::5] = 1  # 100 spikes
    n_spk = spikes.sum()

    coupling, _ = compute_coupling_matrix(
        spikes, lfp, normalization_method="nSpk-square-root"
    )

    assert np.abs(coupling[0, 0]) == pytest.approx(np.sqrt(n_spk), rel=1e-12)
    assert np.angle(coupling[0, 0]) == pytest.approx(0.0, abs=1e-9)


def test_zero_spike_unit_yields_nan_column():
    n_samples = 100
    lfp = np.exp(1j * np.linspace(0, np.pi, n_samples))[None, :]
    spikes = np.zeros((2, n_samples), dtype=int)
    spikes[0, 10] = 1  # unit 0 fires, unit 1 is silent

    coupling, spike_counts = compute_coupling_matrix(spikes, lfp)

    assert spike_counts[1] == 0
    assert np.isnan(coupling[0, 1])
    assert not np.isnan(coupling[0, 0])


def test_unsupported_normalization_raises():
    lfp = np.ones((1, 10), dtype=complex)
    spikes = np.ones((1, 10), dtype=int)
    with pytest.raises(ValueError, match="normalization"):
        compute_coupling_matrix(spikes, lfp, normalization_method="bogus")


@pytest.mark.parametrize(
    "spikes, lfp",
    [
        (np.ones(10, dtype=int), np.ones((1, 10), dtype=complex)),  # 1D spikes
        (np.ones((1, 10), dtype=int), np.ones(10, dtype=complex)),  # 1D lfp
        (np.ones((1, 9), dtype=int), np.ones((1, 10), dtype=complex)),  # mismatch
    ],
)
def test_dimension_errors(spikes, lfp):
    with pytest.raises(ValueError):
        compute_coupling_matrix(spikes, lfp)
