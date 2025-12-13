"""Whitening utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["apply_whitening", "whitenRed2", "whitenRed4"]


def whitenRed2(dataMat_raw: np.ndarray, proportion=0.99):
    """
    Select PCA components accounting for a proportion of variance and whiten.

    Parameters
    ----------
    dataMat_raw : np.ndarray
        Input matrix shaped (channels, samples).
    proportion : float | int
        Float in (0, 1] for variance proportion, int >= 1 for component count,
        or np.nan to keep full rank (MATLAB parity).

    Returns
    -------
    whitenDataMat : np.ndarray
        Whitened data of shape (nred, samples).
    whitenOpr : np.ndarray
        Whitening operator (nred, channels).
    whitenInvOpr : np.ndarray
        Unwhitening operator (channels, nred).
    meanVec : np.ndarray
        Channel-wise mean of input (channels,).
    """
    if dataMat_raw.ndim != 2:
        raise ValueError("dataMat_raw must be 2D (channels x samples)")

    N = dataMat_raw.shape[1]
    meanVec = dataMat_raw.mean(axis=1, keepdims=True)
    dataMat = dataMat_raw - meanVec

    covMat = (dataMat @ dataMat.conj().T) / float(N)

    d, U = np.linalg.eigh(covMat)
    idx = np.argsort(d)[::-1]
    d = d[idx]
    U = U[:, idx]

    if np.isscalar(proportion):
        if isinstance(proportion, (int, np.integer)) and proportion >= 1:
            n = int(proportion)
        elif np.isnan(proportion):
            n = d.size
        else:
            cum = np.cumsum(d)
            thr = float(proportion) * cum[-1]
            n = int(np.searchsorted(cum, thr) + 1)
    else:
        raise ValueError("Invalid proportion type")

    n = min(max(n, 1), d.size)
    Us = U[:, :n]
    ds = d[:n]

    eps = 1e-12
    W = (Us.T / np.sqrt(ds + eps)[:, None])
    Winv = Us * np.sqrt(ds + eps)

    whitenDataMat = W @ dataMat
    return whitenDataMat, W, Winv, meanVec.squeeze()


def whitenRed4(dataMat_raw: np.ndarray, proportion=0.99):
    """
    Trial-wise whitening with consistent reduced rank.
    """
    if dataMat_raw.ndim != 3:
        raise ValueError("dataMat_raw must be 3D (channels x samples x trials)")

    ch, T, nTr = dataMat_raw.shape

    _, W_global, _, _ = whitenRed2(dataMat_raw.reshape(ch, -1), proportion)
    nred = W_global.shape[0]

    whitenDataMat = np.zeros((nred, T, nTr), dtype=dataMat_raw.dtype)
    for k in range(nTr):
        wdata, _, _, _ = whitenRed2(dataMat_raw[:, :, k], nred)
        whitenDataMat[:, :, k] = wdata

    X = dataMat_raw.reshape(ch, -1).T
    Y = whitenDataMat.reshape(nred, -1).T
    W_T, *_ = np.linalg.lstsq(X, Y, rcond=None)
    W = W_T.T

    Winv_T, *_ = np.linalg.lstsq(Y, X, rcond=None)
    Winv = Winv_T.T

    return whitenDataMat, W, Winv


def apply_whitening(
    lfp_signal: np.ndarray,
    *,
    method: int = 0,
    variance_proportion: float | int | None = None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Apply PCA whitening to LFP data, returning the whitened signal and unwhitening operator.
    """

    if method == 0:
        return lfp_signal, None

    if method == 1:
        if np.isrealobj(lfp_signal):
            raise ValueError("Whitening requires analytic (complex) LFP signal")
        prop = np.nan if variance_proportion is None else variance_proportion
        ch, samples, trials = lfp_signal.shape
        flatten = lfp_signal.reshape(ch, -1)
        whitened_flat, _, Winv, _ = whitenRed2(flatten, prop)
        whitened = whitened_flat.reshape(whitened_flat.shape[0], samples, trials)
        return whitened, Winv

    if method == 2:
        if np.isrealobj(lfp_signal):
            raise ValueError("Whitening requires analytic (complex) LFP signal")
        prop = 0.99 if variance_proportion is None else variance_proportion
        whitened, _, Winv = whitenRed4(lfp_signal, prop)
        return whitened, Winv

    raise ValueError(f"Unknown whitening method: {method}")
