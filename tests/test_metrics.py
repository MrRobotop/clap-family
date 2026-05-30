import numpy as np
from clap_family.metrics import (
    target_dwell, trap_dwell, unsafe_time, transition_exposure, jerk_proxy,  # noqa: F401
)
from clap_family.envs.base_env import SimpleEnv
from clap_family.fields import FieldSet, ConstantField, CallableField


def _env():
    fields = FieldSet(V=ConstantField(1.0), U=ConstantField(0.0), C=ConstantField(0.0),
                      O=ConstantField(0.0), c=ConstantField(1.0),
                      E_theta=CallableField(lambda z: np.full(z.shape[0], 0.2)))
    return SimpleEnv(fields=fields, start=np.zeros(2), target=np.array([[0.0, 0.0]]),
                     target_radius=0.5, trap_center=[5.0, 0.0], trap_radius=0.5)


def test_target_dwell_fraction():
    env = _env()
    traj = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]])   # 2 of 3 near target
    assert np.isclose(target_dwell(traj, env), 2 / 3)


def test_trap_dwell_fraction():
    env = _env()
    traj = np.array([[0.0, 0.0], [5.0, 0.0]])   # 1 of 2 in trap
    assert np.isclose(trap_dwell(traj, env), 0.5)


def test_transition_exposure_is_mean_E_theta():
    env = _env()
    traj = np.zeros((4, 2))
    assert np.isclose(transition_exposure(traj, env), 0.2)


def test_jerk_proxy_zero_for_constant_velocity():
    traj = np.cumsum(np.ones((5, 2)), axis=0)   # straight line
    assert np.isclose(jerk_proxy(traj, dt=1.0), 0.0)
