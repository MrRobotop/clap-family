import numpy as np
from clap_family import CLAP, CLAPParams
from clap_family.fields import FieldSet, ConstantField


def test_clap_importable_from_top_level():
    m = CLAP(beta=1.0)
    assert isinstance(m.params, CLAPParams)


def test_clap_equals_base_behaviour():
    fs = FieldSet(V=ConstantField(0.0), U=ConstantField(0.0), C=ConstantField(0.0),
                  O=ConstantField(0.0), c=ConstantField(1.0))
    m = CLAP(N_star=0.6, lambda_end=0.0)
    z = np.zeros((5, 2))
    # all-zero fields -> N̄=sigmoid(0)=0.5 -> deficit 0.1 per state, 4 states, dt=1
    assert np.isclose(m.action(z, fs), 0.1 * 4, atol=1e-9)
