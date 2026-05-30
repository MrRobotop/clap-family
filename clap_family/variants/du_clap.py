"""DU-CLAP — Dynamic-Uncertainty CLAP (paper Eq. 11, 30).

U_dyn = U0 + chi*q + omega*||a||^2 / a0^2 : fast or abrupt motion lowers reliability.
"""
from .base import BaseCLAP


class DUCLAP(BaseCLAP):
    def _effective_uncertainty(self, U0, q_state, a_sq_state, gate):
        p = self.params
        return U0 + p.chi * q_state + p.omega * a_sq_state
