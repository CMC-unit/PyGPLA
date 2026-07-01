"""Unit tests for spike-jitter surrogate generators (pygpla.stats.jitter)."""

import numpy as np
import pytest

from pygpla.stats.jitter import (
    group_preserved_jitter,
    interval_jitter,
    isi_preserved_jitter,
    population_jitter,
)

# Generators that take a scalar window_width.
SCALAR_JITTERS = [interval_jitter, isi_preserved_jitter, group_preserved_jitter]


def _example_spikes():
    rng = np.random.default_rng(7)
    spikes = (rng.random((5, 400)) < 0.05).astype(int)
    return spikes


@pytest.mark.parametrize("jitter_fn", SCALAR_JITTERS)
def test_scalar_jitter_preserves_shape_and_binary(jitter_fn):
    np.random.seed(0)
    spikes = _example_spikes()
    jittered = jitter_fn(spikes, 0.02, 1000.0)

    assert jittered.shape == spikes.shape
    assert set(np.unique(jittered)).issubset({0, 1})
    # Jittering moves spikes (possible bin collisions) but never creates them.
    assert jittered.sum() <= spikes.sum()


@pytest.mark.parametrize("jitter_fn", SCALAR_JITTERS)
def test_scalar_jitter_is_deterministic_under_seed(jitter_fn):
    spikes = _example_spikes()

    np.random.seed(123)
    first = jitter_fn(spikes, 0.02, 1000.0)
    np.random.seed(123)
    second = jitter_fn(spikes, 0.02, 1000.0)

    np.testing.assert_array_equal(first, second)


def test_population_jitter_shape_and_binary():
    np.random.seed(0)
    spikes = _example_spikes()
    jittered = population_jitter(spikes, (0.01, 0.03), 1000.0)

    assert jittered.shape == spikes.shape
    assert set(np.unique(jittered)).issubset({0, 1})
    assert jittered.sum() <= spikes.sum()


def test_population_jitter_is_deterministic_under_seed():
    spikes = _example_spikes()

    np.random.seed(123)
    first = population_jitter(spikes, (0.01, 0.03), 1000.0)
    np.random.seed(123)
    second = population_jitter(spikes, (0.01, 0.03), 1000.0)

    np.testing.assert_array_equal(first, second)


def test_empty_unit_stays_empty():
    spikes = np.zeros((3, 100), dtype=int)
    spikes[0, 10] = 1  # only unit 0 fires
    np.random.seed(1)
    jittered = interval_jitter(spikes, 0.02, 1000.0)
    assert jittered[1:].sum() == 0
    assert jittered.sum() == 1
