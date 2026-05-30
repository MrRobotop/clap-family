import numpy as np
from clap_family.variants.base import BaseCLAP
from clap_family.fields import FieldSet, ConstantField, GaussianBumpField


def _target_fieldset():
    # high value at origin, everything safe/certain/in-distribution, unit speed limit
    return FieldSet(V=GaussianBumpField([0, 0], width=0.5, amplitude=6.0),
                    U=ConstantField(0.0), C=ConstantField(0.0),
                    O=ConstantField(0.0), c=ConstantField(1.0))


def test_running_cost_shape_matches_states():
    m = BaseCLAP(N_star=1.0)
    z = np.zeros((6, 2))            # H+1=6 -> H=5 states scored
    rc = m.running_cost(z, _target_fieldset())
    assert rc.shape == (5,)


def test_action_is_finite_scalar():
    m = BaseCLAP(N_star=1.0, lambda_end=0.0)
    z = np.zeros((6, 2))
    J = m.action(z, _target_fieldset(), target_states=np.array([[0.0, 0.0]]))
    assert np.isscalar(J) and np.isfinite(J)


def test_stationary_at_target_gives_near_zero_action():
    fs = _target_fieldset()
    # N_star equals the lapse value at the origin so deficit ~ 0 there
    nb0 = np.asarray(__import__("clap_family.lapse", fromlist=["conservative_lapse"])
                     .conservative_lapse(6.0, 0, 0, 0, 1, 1, 1, 1.0))
    m = BaseCLAP(N_star=float(nb0))
    z = np.zeros((6, 2))           # stationary at origin: v=0, a=0
    J = m.action(z, fs, target_states=np.array([[0.0, 0.0]]))
    assert J < 1e-6
