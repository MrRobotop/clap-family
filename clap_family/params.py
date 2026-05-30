"""CLAPParams: all CLAP-family hyperparameters in one validated container."""
from dataclasses import dataclass

from .constants import EPS


@dataclass
class CLAPParams:
    """Hyperparameters for the CLAP action (paper Eq. 1, 3, 11-16, 23).

    Lapse field (Eq. 1):     beta, eta, zeta, tau
    Action weights (Eq. 3):  gamma (speed barrier), kappa (accel), lambda_end (terminal)
    Dynamic uncertainty:     chi (speed), omega (accel), a0 (accel scale)   [Eq. 11]
    Risk gate (A-CLAP):      gamma1, kappa1                                  [Eq. 14]
    Learned gate:            alpha, eps_model                               [Eq. 15]
    Phase gate:              alpha_access, alpha_dwell, dwell_radius        [Eq. 16]
    AdaptiveDU mixing:       adaptive_gate in [0, 1]
    Speed margin:            rho (keeps log barrier off its singularity), cmax fallback
    """
    dt: float = 1.0
    beta: float = 1.0
    eta: float = 1.0
    zeta: float = 1.0
    tau: float = 1.0
    gamma: float = 0.5
    kappa: float = 0.1
    lambda_end: float = 10.0
    N_star: float = 1.0
    chi: float = 0.0
    omega: float = 0.0
    a0: float = 1.0
    gamma1: float = 0.0
    kappa1: float = 0.0
    alpha: float = 20.0
    eps_model: float = 0.1
    alpha_access: float = 0.4
    alpha_dwell: float = 1.0
    dwell_radius: float = 0.1
    adaptive_gate: float = 0.5
    rho: float = 1e-3
    cmax: float = 1.0
    eps: float = EPS

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.tau <= 0:
            raise ValueError("tau must be > 0")
        if not (0.0 < self.rho < 1.0):
            raise ValueError("rho must be in (0, 1)")
        if self.cmax <= 0:
            raise ValueError("cmax must be > 0")
        if not (0.0 <= self.adaptive_gate <= 1.0):
            raise ValueError("adaptive_gate must be in [0, 1]")
