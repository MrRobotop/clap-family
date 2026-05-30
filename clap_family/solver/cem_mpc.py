"""Projected Cross-Entropy-Method MPC (paper §5, Listing 1 `cem_mpc_step`).

Receding-horizon loop: sample velocity sequences, project under the speed limit, roll out
with the single-integrator dynamics, score with the chosen variant's action, refit to elites,
execute the first velocity, and replan.
"""
import warnings

import numpy as np

from .project import project_velocity


class ProjectedCEMMPC:
    def __init__(self, samples=256, elites=32, iters=5, init_std=0.3, seed=None):
        self.samples = samples
        self.elites = elites
        self.iters = iters
        self.init_std = init_std
        self.rng = np.random.default_rng(seed)
        if self.iters < 1:
            raise ValueError("iters must be >= 1")

    def _rollout(self, z0, seq, dt, cmax):
        z = [np.asarray(z0, dtype=float)]
        for v in seq:
            z.append(z[-1] + dt * project_velocity(v, cmax))
        return np.asarray(z)

    def _solve_window(self, z0, loss, env, horizon, dt, cmax):
        d = len(z0)
        mean = np.zeros((horizon, d))
        std = self.init_std * np.ones((horizon, d))
        best_traj, best_loss = None, np.inf
        for _ in range(self.iters):
            cand = self.rng.normal(mean, std, size=(self.samples, horizon, d))
            losses = np.empty(self.samples)
            for i, seq in enumerate(cand):
                tr = self._rollout(z0, seq, dt, cmax)
                losses[i] = loss.action(tr, env.fields, target_states=env.target)
                if losses[i] < best_loss:
                    best_loss = losses[i]
                    best_traj = tr
            idx = np.argsort(losses)[: self.elites]
            mean = cand[idx].mean(axis=0)
            std = cand[idx].std(axis=0) + 1e-8
        return best_traj

    def plan(self, env, loss, horizon, rollout=None):
        """Execute receding-horizon control; return the executed trajectory (rollout+1, d)."""
        dt = loss.params.dt
        cmax = loss.params.cmax
        rollout = horizon if rollout is None else rollout
        z = np.asarray(env.start, dtype=float)
        executed = [z]
        for _ in range(rollout):
            window = self._solve_window(z, loss, env, horizon, dt, cmax)
            first_v = (window[1] - window[0]) / dt
            z = z + dt * project_velocity(first_v, cmax)
            executed.append(z)
        traj = np.asarray(executed)
        if not np.any(env.in_target(traj)):
            warnings.warn("planner formed no target dwell — horizon may be < K_access (paper §5)",
                          RuntimeWarning, stacklevel=2)
        return traj
