import numpy as np
from clap_family.solver.project import project_velocity


def test_velocity_under_limit_unchanged():
    v = np.array([0.3, 0.4])    # norm 0.5
    assert np.allclose(project_velocity(v, cmax=1.0), v)


def test_velocity_over_limit_scaled_to_cmax():
    v = np.array([3.0, 4.0])    # norm 5
    out = project_velocity(v, cmax=1.0)
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert np.allclose(out, v / 5.0)
