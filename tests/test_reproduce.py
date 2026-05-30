from clap_family.experiments.reproduce import horizon_sweep


def test_horizon_sweep_returns_results_per_horizon():
    res = horizon_sweep(horizons=(8, 16), samples=120, elites=15, iters=3, rollout=30, seed=0)
    assert set(res.keys()) == {8, 16}
    for H, row in res.items():
        assert {"target_dwell", "trap_dwell", "unsafe_time", "jerk"} <= set(row)
        assert 0.0 <= row["target_dwell"] <= 1.0
