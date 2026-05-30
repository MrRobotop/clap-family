import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_adapter_reexports_from_torch_package():
    """CLAPRegularizationCallback/CLAPTrainerCallback/CLAPRewardWrapper are re-exported
    from clap_family.torch when their optional deps are available."""
    lightning = pytest.importorskip("lightning")  # noqa: F841
    transformers = pytest.importorskip("transformers")  # noqa: F841
    gymnasium = pytest.importorskip("gymnasium")  # noqa: F841

    from clap_family.torch import (  # noqa: F401
        CLAPRegularizationCallback,
        CLAPTrainerCallback,
        CLAPRewardWrapper,
    )


def test_lightning_callback_importable_and_adds_term():
    pytest.importorskip("lightning")
    from clap_family.torch.lightning import CLAPRegularizationCallback
    cb = CLAPRegularizationCallback(lam=0.1, variant="clap", N_star=0.9)
    rollouts = torch.zeros(2, 6, 2, requires_grad=True)
    term = cb.regularization_term(rollouts)     # pure helper, no Trainer needed
    assert term.ndim == 0 and torch.isfinite(term)


def test_hf_callback_importable():
    pytest.importorskip("transformers")
    from clap_family.torch.hf import CLAPTrainerCallback
    cb = CLAPTrainerCallback(lam=0.1, variant="du_clap", N_star=0.9, chi=1.0)
    assert hasattr(cb, "on_log")


def test_rl_reward_wrapper_shapes_reward():
    gym = pytest.importorskip("gymnasium")
    from clap_family.torch.rl import CLAPRewardWrapper

    class _Toy(gym.Env):
        observation_space = gym.spaces.Box(-10, 10, (2,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            self._z = np.zeros(2, np.float32)
            return self._z, {}

        def step(self, a):
            self._z = self._z + np.asarray(a, np.float32)
            return self._z, 0.0, False, False, {}

    from clap_family.fields import FieldSet, ConstantField, GaussianBumpField
    fields = FieldSet(V=GaussianBumpField([0, 0], 1.0, 5.0), U=ConstantField(0.0),
                      C=ConstantField(0.0), O=ConstantField(0.0), c=ConstantField(1.0))
    env = CLAPRewardWrapper(_Toy(), fields=fields, beta=1.0)
    obs, _ = env.reset()
    obs, reward, *_ = env.step(np.array([0.1, 0.0], np.float32))
    assert np.isfinite(reward)
