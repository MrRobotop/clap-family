import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic tests."""
    np.random.seed(0)
