import numpy as np
import pytest

from clap_family import CLAP
from clap_family.solver.cem_mpc import ProjectedCEMMPC
from clap_family.envs.base_env import SimpleEnv
from clap_family.fields import FieldSet, ConstantField, GaussianBumpField


def _env():
    # value bump at origin; start at (3,0); everything safe/certain
    fields = FieldSet(V=GaussianBumpField([0, 0], width=1.0, amplitude=6.0),
                      U=ConstantField(0.0), C=ConstantField(0.0),
                      O=ConstantField(0.0), c=ConstantField(1.0))
    return SimpleEnv(fields=fields, start=np.array([3.0, 0.0]),
                     target=np.array([[0.0, 0.0]]), N_star=0.99)


def test_solver_returns_trajectory_of_expected_length():
    env = _env()
    solver = ProjectedCEMMPC(samples=128, elites=16, iters=4, init_std=0.3, seed=0)
    traj = CLAP(N_star=env.N_star, dt=1.0, cmax=1.0).plan(env, horizon=10, rollout=20, solver=solver)
    assert traj.shape == (21, 2)


def test_solver_moves_toward_target():
    env = _env()
    solver = ProjectedCEMMPC(samples=256, elites=32, iters=5, init_std=0.4, seed=0)
    traj = CLAP(N_star=env.N_star, dt=1.0, cmax=1.0, lambda_end=10.0).plan(
        env, horizon=15, rollout=30, solver=solver)
    start_dist = np.linalg.norm(traj[0])
    end_dist = np.linalg.norm(traj[-1])
    assert end_dist < start_dist   # planner reduces distance to the high-value origin


def test_solver_warns_when_no_target_dwell_forms():
    env = _env()
    solver = ProjectedCEMMPC(samples=32, elites=4, iters=2, init_std=0.05, seed=0)
    # horizon=1, rollout=2, tiny std, far start -> cannot reach the target basin
    with pytest.warns(RuntimeWarning):
        CLAP(N_star=env.N_star, dt=1.0, cmax=1.0).plan(env, horizon=1, rollout=2, solver=solver)
