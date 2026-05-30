"""BaseCLAP: the discrete CLAP action (paper Eq. 23) assembled from override hooks.

Most variants override exactly one hook; ACLAP and its subclasses override the full gate-localized set (_gate, _effective_uncertainty, _speed_weight, _accel_weight).
  _lapse_deficit         (RRLA)
  _effective_uncertainty (DU-CLAP, AdaptiveDU, A-CLAP)
  _gate                  (A-CLAP, LearnedGate, PhaseAdaptive)
  _speed_weight/_accel_weight (A-CLAP risk-localized gamma(r), kappa(r))
"""
import numpy as np

from dataclasses import replace
from ..params import CLAPParams
from ..geometry import velocity, acceleration, norm_sq, speed_ratio_q
from ..lapse import conservative_lapse
from ..admissible import dist_to_target


class BaseCLAP:
    """Base CLAP action object (Eq. 27). Pass a CLAPParams or hyperparameter kwargs."""

    def __init__(self, params: CLAPParams = None, **kwargs):
        if params is None:
            params = CLAPParams(**kwargs)
        elif kwargs:
            params = replace(params, **kwargs)
        self.params = params

    # ---- override hooks (defaults reproduce base CLAP) ----
    def _gate(self, z_states, dist, fields, frac):
        """Risk gate r in [0,1] per state. Base CLAP: no gate (zeros)."""
        return np.zeros(z_states.shape[0])

    def _effective_uncertainty(self, U0, q_state, a_sq_state, gate):
        """Effective uncertainty entering the lapse. Base CLAP: U0 unchanged."""
        return U0

    def _lapse_deficit(self, N_bar, q):
        """Per-state lapse deficit. Base CLAP: N* - N̄ (Eq. 3)."""
        return self.params.N_star - N_bar

    def _speed_weight(self, gate):
        """Per-state gamma. Base CLAP: constant gamma."""
        return np.full_like(gate, self.params.gamma)

    def _accel_weight(self, gate):
        """Per-state kappa. Base CLAP: constant kappa."""
        return np.full_like(gate, self.params.kappa)

    # ---- shared machinery ----
    def _prepare(self, z, fields):
        """Compute the per-state quantities shared by running_cost and action."""
        p = self.params
        z = np.asarray(z, dtype=float)
        H = z.shape[0] - 1
        states = z[:-1]
        v = velocity(z, p.dt)
        a = acceleration(z, p.dt)
        c = fields.c(states)
        q = np.clip(speed_ratio_q(v, c, eps=p.eps), 0.0, 1.0 - p.rho)
        a_sq = np.zeros(H)
        if H >= 2:
            a_sq[1:] = norm_sq(a)
        return states, v, a, q, a_sq, H

    def running_cost(self, z, fields, target_states=None):
        """Per-state integrand (lapse deficit + speed barrier + accel), length H (Eq. 23)."""
        p = self.params
        states, v, a, q, a_sq, H = self._prepare(z, fields)
        dist = (dist_to_target(states, target_states)
                if target_states is not None else np.zeros(H))
        frac = np.arange(H) / max(H - 1, 1)
        gate = self._gate(states, dist, fields, frac)               # (H,)
        U0 = fields.U(states)
        a_sq_over = a_sq / (p.a0 ** 2)
        U_eff = self._effective_uncertainty(U0, q, a_sq_over, gate)  # (H,)
        N_bar = conservative_lapse(fields.V(states), U_eff, fields.C(states),
                                   fields.O(states), p.beta, p.eta, p.zeta, p.tau)
        deficit = self._lapse_deficit(N_bar, q)                      # (H,)
        gamma_t = self._speed_weight(gate)                           # (H,)
        kappa_t = self._accel_weight(gate)                           # (H,)
        barrier = -gamma_t * np.log(1.0 - q)                         # (H,)
        accel = 0.5 * kappa_t * a_sq                                 # (H,), zero at t=0
        return deficit + barrier + accel

    def action(self, z, fields, target_states=None):
        """Total discrete action A_{Δ,H} = Σ integrand·dt + λ_end·d_G(z_H, M*)^2 (Eq. 23)."""
        p = self.params
        rc = self.running_cost(z, fields, target_states)
        total = float(np.sum(rc) * p.dt)
        if target_states is not None and p.lambda_end > 0:
            dH = dist_to_target(np.asarray(z, dtype=float)[-1:], target_states)[0]
            total += p.lambda_end * float(dH) ** 2
        return total

    def plan(self, env, horizon, solver=None, rollout=None, **solver_kwargs):
        """Run projected CEM-MPC on env. Imported lazily to avoid an import cycle."""
        from ..solver.cem_mpc import ProjectedCEMMPC
        if solver is None:
            solver = ProjectedCEMMPC(**solver_kwargs)
        return solver.plan(env, self, horizon=horizon, rollout=rollout)
