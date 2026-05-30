import numpy as np
from clap_family import du_clap, DUCLAP, CLAPParams
from clap_family.fields import FieldSet, ConstantField


def _fs():
    return FieldSet(V=ConstantField(1.0), U=ConstantField(0.0), C=ConstantField(0.0),
                    O=ConstantField(0.0), c=ConstantField(1.0))


def test_alias_matches_class_action():
    fs = _fs()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    p = CLAPParams(N_star=0.7, chi=2.0, omega=1.0, lambda_end=0.0)
    via_fn = du_clap(z, fs, p)
    via_cls = DUCLAP(params=p).action(z, fs)
    assert np.isclose(via_fn, via_cls)


def test_all_seven_aliases_exist():
    from clap_family import (clap, rrla, du_clap, adaptive_du_clap, a_clap,
                             learned_gate_a_clap, phase_adaptive_lg_a_clap)
    for fn in (clap, rrla, du_clap, adaptive_du_clap, a_clap,
               learned_gate_a_clap, phase_adaptive_lg_a_clap):
        assert callable(fn)
