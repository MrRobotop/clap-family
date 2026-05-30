"""Reproduce the paper's main signatures (Fig 2/3 dwell-vs-horizon; learned-gate tradeoff).

CLI:  python -m clap_family.experiments.reproduce [--plots]
Outputs results.json and (with [plots] extra) PNG figures.
"""
import json

from ..variants.rrla import RRLA
from ..variants.learned_gate import LearnedGateACLAP
from ..baselines import RewardOnlyPlanner
from ..solver.cem_mpc import ProjectedCEMMPC
from ..envs.three_region import ThreeRegionEnv
from ..envs.transition_error_env import TransitionErrorEnv
from ..metrics import target_dwell, trap_dwell, unsafe_time, transition_exposure, jerk_proxy


def horizon_sweep(horizons=(20, 40, 80, 120), samples=256, elites=32, iters=5,
                  rollout=120, seed=0):
    """RRLA target dwell vs planning horizon in ThreeRegionEnv (reproduces Fig 2)."""
    env = ThreeRegionEnv()
    out = {}
    for H in horizons:
        solver = ProjectedCEMMPC(samples=samples, elites=elites, iters=iters,
                                 init_std=0.4, seed=seed)
        traj = RRLA(N_star=env.N_star, cmax=1.0, lambda_end=10.0).plan(
            env, horizon=H, rollout=rollout, solver=solver)
        out[H] = {
            "target_dwell": target_dwell(traj, env),
            "trap_dwell": trap_dwell(traj, env),
            "unsafe_time": unsafe_time(traj, env),
            "jerk": jerk_proxy(traj, dt=1.0),
        }
    return out


def learned_gate_tradeoff(samples=256, elites=32, iters=5, rollout=120, seed=0):
    """Learned-Gate A-CLAP vs reward-only on the hidden-transition-error env (reproduces Table 3)."""
    env = TransitionErrorEnv()
    results = {}
    for name, planner in (("reward_only", RewardOnlyPlanner(cmax=1.0)),
                          ("learned_gate", LearnedGateACLAP(N_star=env.N_star, cmax=1.0,
                                                            chi=2.0, omega=1.0, lambda_end=10.0))):
        solver = ProjectedCEMMPC(samples=samples, elites=elites, iters=iters,
                                 init_std=0.4, seed=seed)
        traj = planner.plan(env, horizon=40, rollout=rollout, solver=solver)
        results[name] = {
            "target_dwell": target_dwell(traj, env),
            "transition_exposure": transition_exposure(traj, env),
            "jerk": jerk_proxy(traj, dt=1.0),
        }
    return results


def main(make_plots=False):
    sweep = horizon_sweep()
    gate = learned_gate_tradeoff()
    results = {"horizon_sweep": sweep, "learned_gate_tradeoff": gate}
    with open("results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(json.dumps({str(k): v for k, v in results.items()}, indent=2))
    if make_plots:
        _plots(sweep, gate)
    return results


def _plots(sweep, gate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Hs = sorted(sweep)
    plt.figure()
    plt.plot(Hs, [sweep[H]["target_dwell"] for H in Hs], "o-")
    plt.xlabel("Planning horizon H")
    plt.ylabel("Target dwell fraction")
    plt.title("Projected MPC long-horizon dwell")
    plt.ylim(0, 1)
    plt.savefig("fig_horizon_dwell.png", dpi=120, bbox_inches="tight")

    plt.figure()
    names = list(gate)
    plt.bar(names, [gate[n]["transition_exposure"] for n in names])
    plt.ylabel("Transition-error exposure")
    plt.title("Learned-gate transition-error reduction")
    plt.savefig("fig_learned_gate.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    main(make_plots=ap.parse_args().plots)
