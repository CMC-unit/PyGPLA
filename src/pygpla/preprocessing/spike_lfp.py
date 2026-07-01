"""Spike/LFP preprocessing pipeline."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..config import PreprocessingConfig
from ..core.whitening import apply_whitening

__all__ = ["prepare_spike_lfp_data"]


def _apply_config_overrides(
    *,
    config: PreprocessingConfig | None,
    flag_whitening: int,
    nSpikeThreshold: Optional[int],
    unitSubset: Optional[np.ndarray],
    temporalWindow: Optional[Union[Tuple[int, int], Sequence[np.ndarray]]],
    flag_lfpNrmlz: int,
    plvNrmlzMethed: str,
) -> Tuple[
    int,
    Optional[int],
    Optional[np.ndarray],
    Optional[Union[Tuple[int, int], Sequence[np.ndarray]]],
    int,
    str,
    Optional[float | int],
]:
    """Map config dataclass fields onto legacy flags without changing behaviour."""

    variance_prop = None
    if config is None:
        return (
            flag_whitening,
            nSpikeThreshold,
            None if unitSubset is None else np.asarray(unitSubset),
            temporalWindow,
            flag_lfpNrmlz,
            plvNrmlzMethed,
            variance_prop,
        )

    if config.spike_threshold is not None and nSpikeThreshold is None:
        nSpikeThreshold = config.spike_threshold
    if config.unit_subset is not None and unitSubset is None:
        unitSubset = np.asarray(config.unit_subset)
    if config.temporal_window is not None and temporalWindow is None:
        temporalWindow = config.temporal_window
    if config.lfp_normalization:
        flag_lfpNrmlz = 1
    if config.plv_normalization_method:
        plvNrmlzMethed = config.plv_normalization_method

    if config.whitening.enabled:
        flag_whitening = config.whitening.method
        variance_prop = config.whitening.variance_proportion

    return (
        flag_whitening,
        nSpikeThreshold,
        None if unitSubset is None else np.asarray(unitSubset),
        temporalWindow,
        flag_lfpNrmlz,
        plvNrmlzMethed,
        variance_prop,
    )


def prepare_spike_lfp_data(
    spikeTrains_raw: Sequence[np.ndarray],
    lfpPhases_input: np.ndarray,
    *,
    flag_gPLVnrmlz: int = 1,
    nSpikeThreshold: Optional[int] = None,
    unitSubset: Optional[np.ndarray] = None,
    temporalWindow: Optional[Union[Tuple[int, int], Sequence[np.ndarray]]] = None,
    flag_origDimEigVec: int = 0,
    statTestInfo: Optional[dict] = None,
    iSV: int = 1,
    checkSameElecStuff_flag: int = 0,
    plvNrmlzMethed: str = "nSpk-square-root",
    flag_whitening: int = 0,
    flag_lfpNrmlz: int = 0,
    config: PreprocessingConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray, np.ndarray | None]:
    """
    Prepare spike trains and LFP phases by concatenating across trials, with optional whitening.

    Parameters
    ----------
    spikeTrains_raw :
        List of spike arrays, each shaped (units, samples) per trial.
    lfpPhases_input :
        Complex LFP array shaped (channels, samples, trials).
    flag_gPLVnrmlz :
        Legacy normalization flag (passed through for compatibility).
    nSpikeThreshold :
        Minimum total spikes per unit to retain; None disables thresholding.
    unitSubset :
        Optional iterable of unit indices to include (0-based).
    temporalWindow :
        Either (start, stop) indices or a list of per-trial index arrays.
    flag_origDimEigVec :
        If set, spike vectors are expanded back to original unit dimension (with NaN fill).
    statTestInfo :
        Optional stats config dict; passed through unmodified.
    iSV :
        Singular value index (1-based) used downstream.
    checkSameElecStuff_flag :
        Placeholder for MATLAB parity; unused here.
    plvNrmlzMethed :
        Normalization method for coupling ("nSpk", "nSpk-square-root", "var1_theoretical").
    flag_whitening :
        Whitening method flag (0 off, 1/2 PCA variants).
    flag_lfpNrmlz :
        If nonzero, normalize analytic LFP by channel-wise std after concatenation.
    config :
        Optional `PreprocessingConfig` to override legacy flags.

    Returns
    -------
    spikeTrains_allTrLong : np.ndarray
        Concatenated spike matrix (selected_units, samples * trials).
    lfpPhases_allTrLong : np.ndarray
        Concatenated LFP matrix (channels, samples * trials), optionally whitened/normalized.
    n : dict
        Dimensions dict: keys "LfpCh", "Sample", "SpkUnit", "Tr".
    selectedUnits : np.ndarray
        Indices of retained units relative to original input.
    unwhitenOpr : np.ndarray | None
        Unwhitening operator if whitening applied; else None.
    """

    (
        flag_whitening,
        nSpikeThreshold,
        unitSubset,
        temporalWindow,
        flag_lfpNrmlz,
        plvNrmlzMethed,
        variance_prop,
    ) = _apply_config_overrides(
        config=config,
        flag_whitening=flag_whitening,
        nSpikeThreshold=nSpikeThreshold,
        unitSubset=unitSubset,
        temporalWindow=temporalWindow,
        flag_lfpNrmlz=flag_lfpNrmlz,
        plvNrmlzMethed=plvNrmlzMethed,
    )

    nTr = lfpPhases_input.shape[2]

    if flag_whitening:
        lfpPhases_raw, unwhitenOpr = apply_whitening(
            lfpPhases_input,
            method=flag_whitening,
            variance_proportion=variance_prop,
        )
    else:
        lfpPhases_raw = lfpPhases_input
        unwhitenOpr = None

    n = {
        "LfpCh": lfpPhases_raw.shape[0],
        "Sample": lfpPhases_raw.shape[1],
        "SpkUnit": spikeTrains_raw[0].shape[0],
        "Tr": nTr,
    }

    if temporalWindow is None:
        selectedSamples = np.arange(n["Sample"])
    else:
        if (
            isinstance(temporalWindow, (tuple, list))
            and len(temporalWindow) == 2
            and not isinstance(temporalWindow[0], (np.ndarray, list))
        ):
            startInd, stopInd = temporalWindow
            selectedSamples = np.arange(int(startInd), int(stopInd) + 1)
        else:
            selectedSamples = temporalWindow

    if nSpikeThreshold is not None:
        spikeCount = np.vstack([st.sum(axis=1) for st in spikeTrains_raw]).T
        totSpikeCount = spikeCount.sum(axis=1)
        selectedUnits_thr = np.where(totSpikeCount > nSpikeThreshold)[0]
    else:
        selectedUnits_thr = np.arange(n["SpkUnit"])

    if unitSubset is None:
        userSelectedUnits = np.arange(n["SpkUnit"])
    else:
        userSelectedUnits = np.asarray(unitSubset)

    selectedUnits = np.intersect1d(selectedUnits_thr, userSelectedUnits)

    if isinstance(selectedSamples, list):
        lfpPerTr = [lfpPhases_raw[:, selectedSamples[i], i] for i in range(nTr)]
    else:
        lfpPerTr = [lfpPhases_raw[:, selectedSamples, i] for i in range(nTr)]

    spkPerTr = []
    if isinstance(selectedSamples, list):
        for i in range(nTr):
            spkPerTr.append(spikeTrains_raw[i][np.ix_(selectedUnits, selectedSamples[i])])
    else:
        for i in range(nTr):
            spkPerTr.append(spikeTrains_raw[i][np.ix_(selectedUnits, selectedSamples)])

    spikeTrains_allTrLong = np.concatenate(spkPerTr, axis=1)
    if isinstance(selectedSamples, list):
        lfpPhases_allTrLong = np.concatenate(lfpPerTr, axis=1)
    else:
        updated_nSample = len(selectedSamples)
        lfpPhases_allTrLong = np.empty(
            (lfpPerTr[0].shape[0], nTr * updated_nSample), dtype=lfpPerTr[0].dtype
        )
        for i in range(nTr):
            rng = slice(i * updated_nSample, (i + 1) * updated_nSample)
            lfpPhases_allTrLong[:, rng] = lfpPerTr[i]

    if flag_lfpNrmlz:
        if np.isrealobj(lfpPhases_allTrLong):
            raise ValueError(
                "Phase should not be normalized; provide analytic signal "
                "if normalization is desired"
            )
        stds = lfpPhases_allTrLong.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        lfpPhases_allTrLong = lfpPhases_allTrLong / stds

    return spikeTrains_allTrLong, lfpPhases_allTrLong, n, selectedUnits, unwhitenOpr
