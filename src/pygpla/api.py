"""High-level convenience interface for GPLA analyses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from .config import PreprocessingConfig, StatTestConfig
from .preprocessing import prepare_spike_lfp_data
from .stats.tests import run_statistical_test

__all__ = ["GPLAResult", "gpla"]


@dataclass(slots=True)
class GPLAResult:
    """Container for GPLA outputs."""

    lfp_vector: np.ndarray
    spike_vector: np.ndarray
    gplv: float
    p_value: float
    stats: Any
    metadata: Dict[str, Any]


def _stats_config_to_dict(stats_config: Union[Dict[str, Any], StatTestConfig, None]) -> Dict[str, Any] | None:
    if stats_config is None:
        return None
    if isinstance(stats_config, dict):
        return stats_config
    if is_dataclass(stats_config):
        return asdict(stats_config)
    raise TypeError("stats_config must be a dict, StatTestConfig, or None.")


def gpla(
    spike_trains,
    lfp_signal,
    *,
    flag_gPLVnrmlz: int = 0,
    nSpikeThreshold: Optional[int] = None,
    unitSubset=None,
    temporalWindow=None,
    flag_origDimEigVec: int = 0,
    stats_config: Union[Dict[str, Any], StatTestConfig, None] = None,
    iSV: int = 1,
    sameElecCheckInfo_r: Optional[Dict[str, Any]] = None,
    plvNrmlzMethed: str = "nSpk-square-root",
    plvNrmlzMethod: str | None = None,
    flag_whitening: int = 0,
    flag_lfpNrmlz: int = 0,
    preprocessing_config: PreprocessingConfig | None = None,
) -> GPLAResult:
    """
    High-level GPLA wrapper: preprocessing → GPLA core → statistical testing.

    Parameters
    ----------
    spike_trains :
        List of spike arrays shaped (units, samples) per trial.
    lfp_signal :
        Complex analytic LFP array shaped (channels, samples, trials).
    flag_gPLVnrmlz :
        Legacy gPLV normalization flag (0 keep raw, nonzero scales by matrix size).
    nSpikeThreshold :
        Minimum total spikes per unit to retain; None disables thresholding.
    unitSubset :
        Optional iterable of unit indices (0-based) to include.
    temporalWindow :
        Either (start, stop) indices or list of per-trial index arrays.
    flag_origDimEigVec :
        If set, spike vectors are expanded back to original unit dimension (NaN fill).
    stats_config :
        Dict or `StatTestConfig` specifying "RMT-based" or "spike-jittering" options.
    iSV :
        Singular value index (1-based) to analyze.
    sameElecCheckInfo_r :
        Optional same-electrode mapping for coupling correction.
    plvNrmlzMethed :
        Coupling normalization method ("nSpk", "nSpk-square-root", "var1_theoretical").
    plvNrmlzMethod :
        Alias for `plvNrmlzMethed` with corrected spelling. If both are provided with
        different values, a `ValueError` is raised.
    flag_whitening :
        Whitening method flag (0 off, 1/2 PCA variants).
    flag_lfpNrmlz :
        Normalize analytic LFP amplitude if set.
    preprocessing_config :
        Optional `PreprocessingConfig` to override legacy flags in a structured way.

    Returns
    -------
    GPLAResult
        Dataclass with LFP/spike vectors, gPLV, p-value, stats dict/NaN, and metadata.
    """

    if plvNrmlzMethod is not None:
        if plvNrmlzMethed != "nSpk-square-root" and plvNrmlzMethed != plvNrmlzMethod:
            raise ValueError(
                "Received conflicting normalization parameters: "
                f"plvNrmlzMethed={plvNrmlzMethed!r} and plvNrmlzMethod={plvNrmlzMethod!r}."
            )
        plvNrmlzMethed = plvNrmlzMethod

    spike_list = [np.asarray(st) for st in spike_trains]
    lfp_array = np.asarray(lfp_signal)

    (
        spikeTrains_allTrLong,
        lfpPhases_allTrLong,
        n,
        selectedUnits,
        unwhitenOpr,
    ) = prepare_spike_lfp_data(
        spike_list,
        lfp_array,
        flag_gPLVnrmlz=flag_gPLVnrmlz,
        nSpikeThreshold=nSpikeThreshold,
        unitSubset=unitSubset,
        temporalWindow=temporalWindow,
        flag_origDimEigVec=flag_origDimEigVec,
        statTestInfo=stats_config,
        iSV=iSV,
        checkSameElecStuff_flag=0,
        plvNrmlzMethed=plvNrmlzMethed,
        flag_whitening=flag_whitening,
        flag_lfpNrmlz=flag_lfpNrmlz,
        config=preprocessing_config,
    )

    if sameElecCheckInfo_r is not None:
        sameElecCheckInfo = dict(sameElecCheckInfo_r)
        if "spkU_lfpCh_cnvrtTabel" in sameElecCheckInfo:
            tbl = np.asarray(sameElecCheckInfo["spkU_lfpCh_cnvrtTabel"])
            if tbl.ndim == 1:
                sameElecCheckInfo["spkU_lfpCh_cnvrtTabel"] = tbl[selectedUnits]
            else:
                sameElecCheckInfo["spkU_lfpCh_cnvrtTabel"] = tbl[selectedUnits, :]
    else:
        sameElecCheckInfo = None

    stat_dict = _stats_config_to_dict(stats_config)

    (
        gPLV,
        pValue,
        lfpVec,
        spkVec_raw,
        couplingMatrix,
        singularValues,
        gPLV_nullHypoReject,
        gPLV_stats,
        PLV_stats,
        SV_stats,
    ) = run_statistical_test(
        spikeTrains_allTrLong,
        lfpPhases_allTrLong,
        stat_dict,
        iSV,
        sameElecCheckInfo,
        plvNrmlzMethed,
        unwhitenOpr,
        flag_gPLVnrmlz,
    )

    if isinstance(gPLV_stats, dict):
        gPLV_stats = dict(gPLV_stats)
        gPLV_stats["pValue"] = pValue
        gPLV_stats["nullHypoReject"] = gPLV_nullHypoReject

    if stat_dict is not None:
        stats = {
            "gPLV_stats": gPLV_stats,
            "PLV_stats": PLV_stats,
            "SV_stats": SV_stats,
        }
    else:
        stats = np.nan

    rawSvdStuff = {
        "singularValues": singularValues,
        "couplingMatrix": couplingMatrix,
    }

    if flag_origDimEigVec:
        spkVec = np.full((n["SpkUnit"], spkVec_raw.shape[1]), np.nan, dtype=complex)
        spkVec[selectedUnits, :] = spkVec_raw
    else:
        spkVec = spkVec_raw

    metadata = {
        "raw_svd": rawSvdStuff,
        "selected_units": selectedUnits,
        "dimensions": n,
        "unwhiten_operator": unwhitenOpr,
    }

    return GPLAResult(
        lfp_vector=lfpVec,
        spike_vector=spkVec,
        gplv=float(gPLV),
        p_value=float(pValue) if np.isscalar(pValue) else pValue,
        stats=stats,
        metadata=metadata,
    )
