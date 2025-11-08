"""Statistical tests for GPLA outputs."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..core.gpla import run_gpla_core
from .jitter import (
    group_preserved_jitter,
    interval_jitter,
    isi_preserved_jitter,
    population_jitter,
)

__all__ = ["run_statistical_test"]


def run_statistical_test(
    spike_trains: np.ndarray,
    lfp_signal: np.ndarray,
    stat_test_info: Dict[str, Any] | None,
    sv_index: int,
    same_electrode_info,
    normalization_method: str,
    unwhiten_operator: np.ndarray | None,
    flag_gplv_normalization: int,
) -> Tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    Dict[str, Any],
    Any,
    Any,
]:
    """
    Port of `tnstataliz_gPLV` from PyGPLA_dev providing statistical testing around GPLA.
    """

    (
        lfp_vec,
        spk_vec,
        gplv,
        complex_gplv,
        coupling_matrix,
        singular_values,
    ) = run_gpla_core(
        spike_trains,
        lfp_signal,
        normalize_gplv=0,
        sv_index=sv_index,
        same_electrode_info=same_electrode_info,
        normalization_method=normalization_method,
        unwhiten_operator=unwhiten_operator,
    )

    singular_values = np.atleast_1d(singular_values)

    if stat_test_info is None:
        return (
            gplv,
            np.nan,
            lfp_vec,
            spk_vec,
            coupling_matrix,
            singular_values,
            False,
            np.nan,
            np.nan,
            np.nan,
        )

    test_type = stat_test_info.get("testType", None)

    if test_type == "RMT-based":
        dim_ratio = coupling_matrix.shape[0] / coupling_matrix.shape[1]
        lambda_edge = (1.0 + np.sqrt(dim_ratio)) ** 2
        scaled_sv = (singular_values ** 2) / coupling_matrix.shape[1]

        gplv_stats = {
            "gPLV_testStat": float(scaled_sv[0]),
            "nullHypoReject": bool(scaled_sv[0] > lambda_edge),
        }
        sv_stats = {
            "nullHypoReject": (scaled_sv > lambda_edge),
            "SV_testStat": scaled_sv,
        }

        return (
            gplv,
            np.nan,
            lfp_vec,
            spk_vec,
            coupling_matrix,
            singular_values,
            gplv_stats["nullHypoReject"],
            gplv_stats,
            np.nan,
            sv_stats,
        )

    if test_type == "spike-jittering":
        n_jtr = (
            int(stat_test_info["nJtr"])
            if "nJtr" in stat_test_info
            else int(stat_test_info.get("nPrm", 0))
        )
        alpha = float(stat_test_info.get("alphaValue", 0.05))
        jitter_type = stat_test_info.get("jitterType", "interval-jittering")

        if "rngSeed" in stat_test_info:
            try:
                np.random.seed(int(stat_test_info["rngSeed"]))
            except Exception:
                pass

        n_ch, n_unit = coupling_matrix.shape[0], coupling_matrix.shape[1]
        jtr_gplvs = np.full((n_jtr,), np.nan)
        jtr_coupling = np.full((n_ch, n_unit, n_jtr), np.nan, dtype=complex)
        max_sv = min(n_ch, n_unit)
        jtr_singular = np.full((max_sv, n_jtr), np.nan)

        sf = float(stat_test_info.get("spkSF", 1.0))
        jitter_ww = float(stat_test_info.get("jitterWinWidth", 0.05))
        jitter_range = stat_test_info.get("jitterWinWidthRange", (jitter_ww, jitter_ww))

        def run_one(j_spk):
            (
                _lfp,
                _spk,
                g_val,
                _cg,
                Mj,
                svals,
            ) = run_gpla_core(
                j_spk,
                lfp_signal,
                normalize_gplv=0,
                sv_index=sv_index,
                same_electrode_info=None,
                normalization_method=normalization_method,
                unwhiten_operator=unwhiten_operator,
            )
            return g_val, Mj, svals

        for idx in range(n_jtr):
            if jitter_type == "interval-jittering":
                j_spk = interval_jitter(spike_trains, jitter_ww, sf)
            elif jitter_type == "ISI-preserved-interval-jittering":
                j_spk = isi_preserved_jitter(spike_trains, jitter_ww, sf)
            elif jitter_type == "group-preserved-interval-jittering":
                j_spk = group_preserved_jitter(spike_trains, jitter_ww, sf)
            elif jitter_type == "sync-jittering":
                j_spk = population_jitter(spike_trains, jitter_range, sf)
            elif jitter_type == "fake-jittering":
                j_spk = spike_trains.copy()
            else:
                raise ValueError(f"Unsupported jitterType: {jitter_type}")

            g_val, Mj, svals = run_one(j_spk)
            jtr_gplvs[idx] = g_val
            jtr_coupling[:, :, idx] = Mj
            m = min(max_sv, svals.shape[0])
            jtr_singular[:m, idx] = svals[:m]

        p_value = np.nansum(jtr_gplvs > gplv) / float(n_jtr)
        gplv_stats = {
            "nullDistribution": jtr_gplvs,
            "gPLV_testStat": gplv,
        }
        null_reject = p_value < alpha

        abs_Mj = np.abs(jtr_coupling)
        obs = np.abs(coupling_matrix)[..., None]
        plv_pvals = np.nansum(abs_Mj > obs, axis=2) / float(n_jtr)
        plv_stats = {
            "pValue": plv_pvals,
            "nullDistribution": jtr_coupling,
            "nullHypoReject": (plv_pvals < alpha),
            "nullDistribution_cPLV": jtr_coupling,
        }

        sv_type = stat_test_info.get("SVspectrumStatsType", "default")
        if sv_type == "default":
            sv_pvals = (
                np.nansum(
                    jtr_singular > singular_values[:max_sv, None],
                    axis=1,
                )
                / float(n_jtr)
            )
            sv_stats = {
                "pValue": sv_pvals,
                "nullDistribution": jtr_singular,
            }
        elif sv_type == "cummulative":
            raise NotImplementedError("SVspectrumStatsType 'cummulative' not supported.")
        elif sv_type == "RMT-heuristic":
            largest = np.tile(jtr_singular[0:1, :], (max_sv, 1))
            sv_pvals = np.nansum(
                largest > singular_values[:max_sv, None],
                axis=1,
            ) / float(n_jtr)
            sv_stats = {
                "pValue": sv_pvals,
                "nullDistribution": jtr_singular,
            }
        else:
            raise ValueError("Unsupported SVspectrumStatsType")

        return (
            gplv,
            p_value,
            lfp_vec,
            spk_vec,
            coupling_matrix,
            singular_values,
            null_reject,
            gplv_stats,
            plv_stats,
            sv_stats,
        )

    raise ValueError(f"Unsupported testType: {test_type}")
