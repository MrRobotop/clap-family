"""Optional PyTorch training-integration adapters. Importing this requires `pip install clap_family[torch]`.

CLAPRegularizer is always available (requires torch).
CLAPRegularizationCallback, CLAPTrainerCallback, CLAPRewardWrapper are exported only when
their respective optional dependencies (lightning, transformers, gymnasium) are installed;
importing clap_family.torch will not fail if they are absent.
"""
from .regularizer import CLAPRegularizer

__all__ = ["CLAPRegularizer"]

try:
    from .lightning import CLAPRegularizationCallback
    __all__ = [*__all__, "CLAPRegularizationCallback"]
except ImportError:
    pass

try:
    from .hf import CLAPTrainerCallback
    __all__ = [*__all__, "CLAPTrainerCallback"]
except ImportError:
    pass

try:
    from .rl import CLAPRewardWrapper
    __all__ = [*__all__, "CLAPRewardWrapper"]
except ImportError:
    pass
