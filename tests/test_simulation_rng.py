"""Reproducibility tests for the simulation generators (pygpla.simulations).

The generators must draw exclusively from the ``rng`` they are handed, so that a
seeded call is reproducible and a library call never disturbs the caller's own
global NumPy random state.
"""

import numpy as np
import pytest

from pygpla.simulations import (
    generate_homogeneous_poisson,
    generate_inhomogeneous_poisson,
    generate_phase_locked_spikes,
    simulate_transient_locked,
)

SIGNAL_PARAMS = dict(nCh=3, nUnit=6, SF=400.0, nTr=2, signalLength=2.0)
GLOBAL_PARAMS = dict(
    oscFreq=20.0,
    nCycl=10,
    syncSigProportion=0.7,
    lfpPhaseNoise_kappa=8.0,
    whiteNoise_sigma=0.05,
)


def _transient(seed):
    return simulate_transient_locked(
        GLOBAL_PARAMS,
        dict(avefiringRate=18.0),
        dict(lockingStrength_kappa=10.0, lockingPhase=0.0),
        SIGNAL_PARAMS,
        return_analytic=True,
        rng=np.random.default_rng(seed),
    )


def _phase_locked(seed):
    spikes, _, _ = generate_phase_locked_spikes(
        dict(lockingPhase=0.0, kappa=5.0, avefiringRate=18.0, lockingFreq=20.0),
        SIGNAL_PARAMS,
        rng=np.random.default_rng(seed),
    )
    return spikes


def _homogeneous(seed):
    return generate_homogeneous_poisson(
        20.0, 1.0, 200.0, n_tr=2, n_unit=4, rng=np.random.default_rng(seed)
    )


def _inhomogeneous(seed):
    fr = np.full((4, 200, 2), 20.0)
    return generate_inhomogeneous_poisson(
        fr, 1.0, 200.0, n_tr=2, n_unit=4, rng=np.random.default_rng(seed)
    )


SPIKE_GENERATORS = [_phase_locked, _homogeneous, _inhomogeneous]


@pytest.mark.parametrize("generator", SPIKE_GENERATORS)
def test_same_seed_reproduces_spikes(generator):
    np.testing.assert_array_equal(generator(0), generator(0))


@pytest.mark.parametrize("generator", SPIKE_GENERATORS)
def test_different_seeds_give_different_spikes(generator):
    assert not np.array_equal(generator(0), generator(1))


def test_transient_simulation_is_reproducible_including_spikes():
    _, lfp_a, spikes_a, _ = _transient(42)
    _, lfp_b, spikes_b, _ = _transient(42)

    np.testing.assert_array_equal(lfp_a, lfp_b)
    assert len(spikes_a) == len(spikes_b)
    for tr_a, tr_b in zip(spikes_a, spikes_b, strict=True):
        np.testing.assert_array_equal(tr_a, tr_b)


def test_transient_simulation_varies_with_seed():
    *_, spikes_a, _ = _transient(42)
    *_, spikes_b, _ = _transient(43)

    assert not all(
        np.array_equal(tr_a, tr_b) for tr_a, tr_b in zip(spikes_a, spikes_b, strict=True)
    )


@pytest.mark.parametrize("generator", SPIKE_GENERATORS)
def test_generators_do_not_disturb_global_random_state(generator):
    """A seeded caller must get the same draw whether or not pygpla ran."""

    np.random.seed(0)
    expected = np.random.rand()

    np.random.seed(0)
    generator(0)
    actual = np.random.rand()

    assert actual == expected
