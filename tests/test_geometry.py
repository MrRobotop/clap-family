import numpy as np
from clap_family.geometry import velocity, acceleration, norm_sq, speed_ratio_q


def test_velocity_and_acceleration_shapes():
    z = np.zeros((5, 2))
    assert velocity(z, 1.0).shape == (4, 2)
    assert acceleration(z, 1.0).shape == (3, 2)


def test_stationary_trajectory_has_zero_motion():
    z = np.ones((5, 2))
    assert np.allclose(velocity(z, 1.0), 0.0)
    assert np.allclose(acceleration(z, 1.0), 0.0)


def test_constant_velocity_has_zero_acceleration():
    z = np.cumsum(np.ones((5, 2)), axis=0)   # straight line, step 1
    v = velocity(z, 1.0)
    assert np.allclose(v, 1.0)
    assert np.allclose(acceleration(z, 1.0), 0.0)


def test_norm_sq_identity_metric():
    v = np.array([[3.0, 4.0]])
    assert np.allclose(norm_sq(v), 25.0)


def test_speed_ratio_q_matches_definition():
    v = np.array([[1.0, 0.0]])
    c = np.array([2.0])
    # q = ||v||^2 / c^2 = 1/4
    assert np.allclose(speed_ratio_q(v, c), 0.25)
