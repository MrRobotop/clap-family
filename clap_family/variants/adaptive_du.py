"""AdaptiveDU-CLAP — interpolates base CLAP and DU-CLAP via a fixed mixing gate (paper §1, Table 1).

U_eff = U0 + adaptive_gate * (chi*q + omega*||a||^2/a0^2).
"""
from .base import BaseCLAP


class AdaptiveDUCLAP(BaseCLAP):
    def _effective_uncertainty(self, U0, q_state, a_sq_state, gate):
        # gate is unused here; adaptive mixing is controlled by the scalar p.adaptive_gate
        p = self.params
        return U0 + p.adaptive_gate * (p.chi * q_state + p.omega * a_sq_state)
