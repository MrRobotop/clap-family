"""A-CLAP — Access-Aware CLAP (paper Eq. 13-14, 31).

A risk gate r(z) in [0,1] localizes dynamic conservatism:
  U_r = U0 + r(z) * (chi*q + omega*||a||^2/a0^2)
  gamma(r) = gamma0 + gamma1*r,  kappa(r) = kappa0 + kappa1*r.
Reliable access corridors (low r) are not penalized as strongly as risky regions.
"""
import numpy as np
from .base import BaseCLAP


class ACLAP(BaseCLAP):
    def __init__(self, params=None, risk_field=None, **kwargs):
        super().__init__(params, **kwargs)
        # default risk = OOD score clipped to [0,1]
        self.risk_field = risk_field

    def _resolve_risk(self, z_states, fields):
        if self.risk_field is None:
            return np.clip(fields.O(z_states), 0.0, 1.0)
        return np.clip(self.risk_field(z_states), 0.0, 1.0)

    def _gate(self, z_states, dist, fields, frac):
        return self._resolve_risk(z_states, fields)

    def _effective_uncertainty(self, U0, q_state, a_sq_state, gate):
        p = self.params
        return U0 + gate * (p.chi * q_state + p.omega * a_sq_state)

    def _speed_weight(self, gate):
        p = self.params
        return p.gamma + p.gamma1 * gate

    def _accel_weight(self, gate):
        p = self.params
        return p.kappa + p.kappa1 * gate
