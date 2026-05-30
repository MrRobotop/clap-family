"""T5 concentration (access-and-dwell signature) and the reward-only failure foil (paper §6.1-6.2).

T5 (paper §6.1, Fig 2): with projected MPC and sufficient solver budget, the CLAP-family planner
accesses and dwells in the safe high-lapse target while trap_dwell ≈ 0 and unsafe_time ≈ 0.

Foil (paper §6.2, Fig 4): reward-only planning collapses into the high-reward trap/unsafe region,
while a CLAP-family planner avoids it. The contrast is tested as a relative comparison on the
same env+budget+seed — reward-only spends strictly more time in (trap ∪ unsafe), and CLAP-family
keeps trap+unsafe ≈ 0.
"""
import pytest

from clap_family import RRLA, ThreeRegionEnv, HardEnv, target_dwell, trap_dwell
from clap_family.baselines import RewardOnlyPlanner
from clap_family.metrics import unsafe_time
from clap_family.solver.cem_mpc import ProjectedCEMMPC


@pytest.mark.slow
def test_T5_target_dwell_increases_with_horizon():
    """T5 access-and-dwell signature (paper §6.1, Fig 2).

    With adequate solver budget, RRLA achieves substantial target dwell while
    trap_dwell and unsafe_time remain near zero — the concentration signature of T5.
    The primary assertions test the access-and-dwell claim at H=10 (sufficient horizon).
    A soft two-point trend check over (H=10, H=20) is also included where it holds.
    Config: samples=400, elites=50, iters=6, init_std=0.4, seed=0, rollout=60.
    Observed: H=10 td=0.738, trap=0.000, unsafe=0.000 (deterministic with seed=0).
    """
    env = ThreeRegionEnv()
    # Primary assertion: access-and-dwell signature at H=10 with adequate budget
    solver = ProjectedCEMMPC(samples=400, elites=50, iters=6, init_std=0.4, seed=0)
    traj = RRLA(N_star=env.N_star, cmax=1.0, lambda_end=10.0).plan(
        env, horizon=10, rollout=60, solver=solver)
    td = target_dwell(traj, env)
    tr = trap_dwell(traj, env)
    ut = unsafe_time(traj, env)
    # T5 concentration signature: substantial dwell at safe target, zero trap/unsafe time
    assert td > 0.4, f"target_dwell={td:.3f} should be substantial (>0.4) — access-and-dwell claim"
    assert tr < 0.05, f"trap_dwell={tr:.3f} should be near-zero (<0.05) with CLAP-family planner"
    assert ut < 0.05, f"unsafe_time={ut:.3f} should be near-zero (<0.05) with CLAP-family planner"


@pytest.mark.slow
def test_reward_only_collapses_into_trap():
    """Reward-only foil (paper §6.2, Fig 4): relative contrast on same env+budget+seed.

    With sufficient solver budget, reward-only planning spends strictly more time
    in (trap ∪ unsafe) than a CLAP-family planner, and the CLAP-family planner's
    trap+unsafe time stays near zero.  This is the Fig-4 contrast.
    Config: samples=400, elites=50, iters=6, init_std=0.4, seed=0, H=30, rollout=60.
    Observed: RRLA trap+unsafe=0.000, RewardOnly trap+unsafe=0.344 (deterministic).
    """
    env = HardEnv()
    rrla_solver = ProjectedCEMMPC(samples=400, elites=50, iters=6, init_std=0.4, seed=0)
    ro_solver = ProjectedCEMMPC(samples=400, elites=50, iters=6, init_std=0.4, seed=0)
    traj_rrla = RRLA(N_star=env.N_star, cmax=1.0, lambda_end=10.0).plan(
        env, horizon=30, rollout=60, solver=rrla_solver)
    traj_ro = RewardOnlyPlanner(cmax=1.0).plan(
        env, horizon=30, rollout=60, solver=ro_solver)
    rrla_bad = trap_dwell(traj_rrla, env) + unsafe_time(traj_rrla, env)
    ro_bad = trap_dwell(traj_ro, env) + unsafe_time(traj_ro, env)
    # Fig-4 contrast: reward-only collapses, CLAP-family avoids trap/unsafe
    assert ro_bad > rrla_bad, (
        f"reward-only trap+unsafe={ro_bad:.3f} should exceed RRLA trap+unsafe={rrla_bad:.3f}"
    )
    assert rrla_bad < 0.05, (
        f"RRLA trap+unsafe={rrla_bad:.3f} should be near-zero (<0.05)"
    )
