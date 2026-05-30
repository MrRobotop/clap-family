"""PyTorch Lightning adapter. Requires `pip install clap_family[lightning]`."""
import lightning.pytorch as pl

from .regularizer import CLAPRegularizer


class CLAPRegularizationCallback(pl.Callback):
    """Adds a CLAP regularization term computed from latent rollouts exposed by the LightningModule.

    Convention: your LightningModule may implement `clap_rollouts(batch) -> (B,H+1,d)`. If present,
    this callback logs the CLAP term each training batch. The pure helper `regularization_term`
    is also usable directly inside your `training_step`.
    """

    def __init__(self, lam=0.1, variant="clap", **kwargs):
        super().__init__()
        self.lam = lam
        self.reg = CLAPRegularizer(variant=variant, **kwargs)

    def regularization_term(self, rollouts):
        """Compute lam * CLAP(rollouts). (B,H+1,d) -> scalar."""
        return self.lam * self.reg(rollouts)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        fn = getattr(pl_module, "clap_rollouts", None)
        if fn is None:
            return
        term = self.regularization_term(fn(batch))
        pl_module.log("clap_reg", term, prog_bar=True)
