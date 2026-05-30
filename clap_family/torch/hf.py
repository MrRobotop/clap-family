"""HuggingFace transformers.Trainer adapter. Requires `pip install clap_family[hf]`.

Usage recipe (docs): subclass Trainer and add `self.clap.regularization_term(rollouts)` inside
`compute_loss`. This callback handles metric logging; the helper computes the term.
"""
from transformers import TrainerCallback

from .regularizer import CLAPRegularizer


class CLAPTrainerCallback(TrainerCallback):
    """HuggingFace TrainerCallback that provides a CLAP regularization helper.

    Subclass HF Trainer, pass this callback, and call `self.clap.regularization_term(rollouts)`
    inside your custom `compute_loss` to add the CLAP term to the training objective.
    """

    def __init__(self, lam=0.1, variant="clap", **kwargs):
        self.lam = lam
        self.reg = CLAPRegularizer(variant=variant, **kwargs)

    def regularization_term(self, rollouts):
        """Compute lam * CLAP(rollouts). (B,H+1,d) -> scalar."""
        return self.lam * self.reg(rollouts)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # logging hook; the term itself is added inside a custom compute_loss (see docs)
        return control
