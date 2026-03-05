import numpy as np

from pygpla.api import gpla
from pygpla.simulations import simulate_transient_locked


def _build_cases(n_unit: int):
    phase_linear = np.linspace(0, np.pi, n_unit)
    n_chunk = 3
    chunk_sz = n_unit // n_chunk
    cluster = np.concatenate(
        [
            np.full(chunk_sz, 0.0),
            np.full(chunk_sz, 2 * np.pi / 3),
            np.full(n_unit - 2 * chunk_sz, 4 * np.pi / 3),
        ]
    )
    rng = np.random.default_rng(123)
    rng.shuffle(cluster)

    return [
        dict(kappa=10.0, phase=np.pi * np.ones(n_unit)),
        dict(kappa=10.0, phase=phase_linear),
        dict(kappa=10.0, phase=cluster),
        dict(kappa=0.0, phase=np.pi / 2 * np.ones(n_unit)),
    ]


def _run_transient_case(case, signal_params, global_params, seed):
    coupling_params = dict(
        lockingStrength_kappa=case["kappa"],
        lockingPhase=case["phase"],
    )
    spike_params = dict(avefiringRate=18.0)

    _, lfp_analytic, spikes, _ = simulate_transient_locked(
        global_params,
        spike_params,
        coupling_params,
        signal_params,
        return_analytic=True,
        rng=np.random.default_rng(seed),
    )

    result = gpla(
        spikes,
        lfp_analytic,
        stats_config=None,
        plvNrmlzMethed="var1_theoretical",
        flag_whitening=0,
        flag_lfpNrmlz=0,
    )
    return result


def test_transient_models_separate_locked_and_null_conditions():
    signal_params = dict(
        nCh=1,
        nUnit=12,
        SF=600.0,
        nTr=6,
        signalLength=6.0,
    )
    global_params = dict(
        oscFreq=20.0,
        nCycl=15,
        syncSigProportion=0.7,
        lfpPhaseNoise_kappa=8.0,
        whiteNoise_sigma=0.05,
    )

    cases = _build_cases(signal_params["nUnit"])
    seeds = [101, 202, 303, 404]

    results = [
        _run_transient_case(case, signal_params, global_params, seed)
        for case, seed in zip(cases, seeds, strict=True)
    ]
    gplv_values = [res.gplv for res in results]

    assert all(np.isfinite(val) for val in gplv_values)

    # Coupled scenarios should yield stronger gPLV than the uncoupled control
    baseline = gplv_values[-1]
    assert all(val > baseline for val in gplv_values[:3])

    # Spike vector dimensionality should match selected units (all units kept)
    for res in results:
        assert res.spike_vector.shape[0] == signal_params["nUnit"]


def test_gpla_accepts_corrected_plv_normalization_keyword_alias():
    signal_params = dict(
        nCh=1,
        nUnit=8,
        SF=400.0,
        nTr=3,
        signalLength=3.0,
    )
    global_params = dict(
        oscFreq=20.0,
        nCycl=10,
        syncSigProportion=0.7,
        lfpPhaseNoise_kappa=8.0,
        whiteNoise_sigma=0.05,
    )
    spike_params = dict(avefiringRate=12.0)
    coupling_params = dict(lockingStrength_kappa=8.0, lockingPhase=0.0)

    _, lfp_analytic, spikes, _ = simulate_transient_locked(
        global_params,
        spike_params,
        coupling_params,
        signal_params,
        return_analytic=True,
        rng=np.random.default_rng(7),
    )

    result = gpla(
        spikes,
        lfp_analytic,
        stats_config=None,
        plvNrmlzMethod="var1_theoretical",
        flag_whitening=0,
        flag_lfpNrmlz=0,
    )

    assert np.isfinite(result.gplv)
