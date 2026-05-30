import numpy as np
from clap_family.fields import (
    CallableField, ConstantField, GaussianBumpField, EnsembleField, FieldSet,
)


def test_constant_field_broadcasts_over_states():
    f = ConstantField(0.5)
    z = np.zeros((4, 2))
    out = f(z)
    assert out.shape == (4,) and np.allclose(out, 0.5)


def test_callable_field_wraps_function():
    f = CallableField(lambda z: z.sum(axis=-1))
    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(f(z), [3.0, 7.0])


def test_gaussian_bump_peaks_at_center():
    f = GaussianBumpField(center=[0.0, 0.0], width=1.0, amplitude=2.0)
    at_center = f(np.array([[0.0, 0.0]]))
    far = f(np.array([[5.0, 5.0]]))
    assert at_center[0] > far[0]
    assert np.isclose(at_center[0], 2.0)


def test_ensemble_field_is_member_std():
    members = [CallableField(lambda z: z[:, 0]), CallableField(lambda z: z[:, 0] + 2.0)]
    f = EnsembleField(members)
    out = f(np.array([[0.0, 0.0]]))
    assert np.isclose(out[0], 1.0)   # std of {0, 2} = 1


def test_fieldset_holds_named_fields():
    fs = FieldSet(V=ConstantField(1.0), U=ConstantField(0.0), C=ConstantField(0.0),
                  O=ConstantField(0.0), c=ConstantField(1.0))
    z = np.zeros((3, 2))
    assert np.allclose(fs.V(z), 1.0) and fs.E_theta is None
