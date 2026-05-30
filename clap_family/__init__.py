"""clap_family — Conservative Lapse-Action Planning.

A variational access-and-dwell framework for safe latent trajectory optimization.
See: R. Patil, "Conservative Lapse-Action Planning", 2026.
"""

__version__ = "0.1.0"

from .params import CLAPParams
from .fields import (
    Field, CallableField, ConstantField, GaussianBumpField, EnsembleField, FieldSet,
)
from .admissible import AdmissibleSet, dist_to_target
from .lapse import sigmoid, conservative_lapse, estimate_N_star
from .variants.clap import CLAP
from .variants.rrla import RRLA
from .variants.du_clap import DUCLAP
from .variants.adaptive_du import AdaptiveDUCLAP
from .variants.a_clap import ACLAP
from .variants.learned_gate import LearnedGateACLAP
from .variants.phase_adaptive import PhaseAdaptiveLGACLAP
from .variants.aliases import (
    clap, rrla, du_clap, adaptive_du_clap, a_clap, learned_gate_a_clap, phase_adaptive_lg_a_clap,
)
from .solver.cem_mpc import ProjectedCEMMPC
from .envs.base_env import SimpleEnv
from .metrics import target_dwell, trap_dwell, unsafe_time, transition_exposure, jerk_proxy
from .envs.three_region import ThreeRegionEnv
from .envs.hard_env import HardEnv
from .envs.transition_error_env import TransitionErrorEnv

__all__ = [
    "CLAPParams", "Field", "CallableField", "ConstantField", "GaussianBumpField",
    "EnsembleField", "FieldSet", "AdmissibleSet", "dist_to_target",
    "sigmoid", "conservative_lapse", "estimate_N_star",
    "CLAP", "RRLA", "DUCLAP", "AdaptiveDUCLAP", "ACLAP", "LearnedGateACLAP",
    "PhaseAdaptiveLGACLAP",
    "clap", "rrla", "du_clap", "adaptive_du_clap", "a_clap", "learned_gate_a_clap",
    "phase_adaptive_lg_a_clap",
    "ProjectedCEMMPC", "SimpleEnv",
    "target_dwell", "trap_dwell", "unsafe_time", "transition_exposure", "jerk_proxy",
    "ThreeRegionEnv", "HardEnv", "TransitionErrorEnv",
]
