import numpy as np
import pytest
from clap_family.admissible import AdmissibleSet, dist_to_target
from clap_family.fields import FieldSet, ConstantField


def _fs(C=0.0, U=0.0, O=0.0):  # noqa: E741
    return FieldSet(V=ConstantField(1.0), U=ConstantField(U), C=ConstantField(C),
                    O=ConstantField(O), c=ConstantField(1.0))


def test_membership_respects_all_thresholds():
    adm = AdmissibleSet(Cmax=1.0, Umax=1.0, Omax=1.0)
    z = np.zeros((2, 2))
    assert np.all(adm.contains(z, _fs(C=0.5, U=0.5, O=0.5)))
    assert not np.any(adm.contains(z, _fs(C=2.0)))


def test_dist_to_target_is_min_over_set():
    targets = np.array([[0.0, 0.0], [10.0, 0.0]])
    z = np.array([[1.0, 0.0], [9.0, 0.0]])
    d = dist_to_target(z, targets)
    assert np.allclose(d, [1.0, 1.0])


def test_default_admissible_is_everything():
    adm = AdmissibleSet()
    assert np.all(adm.contains(np.random.randn(5, 3), _fs(C=1e6, U=1e6, O=1e6)))


def test_dist_to_target_empty_raises():
    with pytest.raises(ValueError):
        dist_to_target(np.zeros((2, 2)), np.zeros((0, 2)))
