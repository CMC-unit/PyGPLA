"""Unit tests for whitening utilities (pygpla.core.whitening)."""

import numpy as np
import pytest

from pygpla.core.whitening import apply_whitening, whitenRed2


def test_whitenRed2_decorrelates_full_rank():
    rng = np.random.default_rng(3)
    n_ch, n_samples = 3, 2000
    # Correlated channels via a random mixing matrix.
    mixing = rng.standard_normal((n_ch, n_ch))
    data = mixing @ rng.standard_normal((n_ch, n_samples))

    whitened, W, Winv, mean_vec = whitenRed2(data, proportion=np.nan)

    # Whitened covariance should be approximately the identity.
    cov = (whitened @ whitened.conj().T) / n_samples
    np.testing.assert_allclose(cov, np.eye(W.shape[0]), atol=1e-6)
    assert mean_vec.shape == (n_ch,)


def test_whitenRed2_variance_proportion_reduces_rank():
    rng = np.random.default_rng(4)
    # Third channel is a near-duplicate => one direction carries little variance.
    base = rng.standard_normal((2, 1000))
    data = np.vstack([base, base[0:1] + 1e-6 * rng.standard_normal((1, 1000))])

    whitened, W, _, _ = whitenRed2(data, proportion=0.95)

    assert W.shape[0] < data.shape[0]
    assert whitened.shape[0] == W.shape[0]


def test_whitenRed2_requires_2d():
    with pytest.raises(ValueError):
        whitenRed2(np.ones((2, 2, 2)))


def test_apply_whitening_method_zero_is_passthrough():
    lfp = np.ones((2, 10, 3), dtype=complex)
    out, winv = apply_whitening(lfp, method=0)
    assert out is lfp
    assert winv is None


def test_apply_whitening_rejects_real_signal_and_unknown_method():
    real_lfp = np.ones((2, 10, 3))
    with pytest.raises(ValueError, match="analytic"):
        apply_whitening(real_lfp, method=1)
    with pytest.raises(ValueError, match="[Uu]nknown"):
        apply_whitening(np.ones((2, 10, 3), dtype=complex), method=99)
