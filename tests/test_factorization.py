"""Unit tests for the SVD factorization (pygpla.core.factorization)."""

import numpy as np
import pytest

from pygpla.core.factorization import factorize_coupling_matrix


def test_svd_reconstructs_matrix():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((5, 4)) + 1j * rng.standard_normal((5, 4))

    U, V, s = factorize_coupling_matrix(M)

    assert U.shape == (5, 4)
    assert V.shape == (4, 4)
    assert s.shape == (4,)
    # Singular values are non-negative and sorted descending.
    assert np.all(s >= 0)
    assert np.all(np.diff(s) <= 1e-9)
    # U @ diag(s) @ V^H reconstructs M.
    reconstructed = U @ np.diag(s) @ V.conj().T
    np.testing.assert_allclose(reconstructed, M, atol=1e-10)


def test_rank_one_leading_singular_value():
    u = np.array([1.0, 0.0, 0.0], dtype=complex)
    v = np.array([0.0, 1.0], dtype=complex)
    M = 3.0 * np.outer(u, v)  # rank-1 matrix with singular value 3

    _, _, s = factorize_coupling_matrix(M)

    assert s[0] == pytest.approx(3.0)
    assert s[1] == pytest.approx(0.0, abs=1e-10)


def test_nan_column_is_excluded_and_padded():
    rng = np.random.default_rng(2)
    M = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))
    M[:, 1] = np.nan  # e.g. a zero-spike unit

    with pytest.warns(RuntimeWarning, match="NaN"):
        U, V, s = factorize_coupling_matrix(M)

    # The NaN column maps to a NaN row in the spike singular vectors,
    # while the valid columns remain finite.
    assert np.isnan(V[1, :]).all()
    assert np.isfinite(V[0, :]).all()
    assert np.isfinite(U).all()
    assert np.isfinite(s).all()
