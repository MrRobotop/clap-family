import numpy as np
from clap_family import ThreeRegionEnv, HardEnv, TransitionErrorEnv


def test_three_region_has_separated_regions():
    env = ThreeRegionEnv()
    # target has high value + low cost; trap has high value + high cost/uncertainty
    t = env.target[0]
    assert env.fields.V(np.atleast_2d(t))[0] > 0
    assert env.in_target(np.atleast_2d(t))[0]
    assert not env.in_trap(np.atleast_2d(t))[0]


def test_hard_env_trap_is_attractive_but_unsafe():
    env = HardEnv()
    trap = np.atleast_2d(env.trap_center)
    assert env.fields.V(trap)[0] > 0           # attractive raw value
    assert env.fields.C(trap)[0] > env.fields.C(np.atleast_2d(env.target[0]))[0]


def test_transition_error_env_has_hidden_error_shortcut():
    env = TransitionErrorEnv()
    assert env.fields.E_theta is not None
    # shortcut region has elevated transition error
    short = np.atleast_2d(env.shortcut_center)
    far = np.atleast_2d(env.start)
    assert env.fields.E_theta(short)[0] > env.fields.E_theta(far)[0]
