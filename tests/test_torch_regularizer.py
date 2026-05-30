import pytest

torch = pytest.importorskip("torch")
from clap_family.torch import CLAPRegularizer  # noqa: E402


def test_regularizer_forward_is_scalar_and_differentiable():
    reg = CLAPRegularizer(variant="clap", beta=1.0, gamma=0.5, kappa=0.1, N_star=0.9)
    B, H, d = 4, 12, 2
    rollouts = torch.zeros(B, H + 1, d, requires_grad=True)
    rollouts.data += 0.1 * torch.randn(B, H + 1, d)
    loss = reg(rollouts)                      # (B,H+1,d) -> scalar
    assert loss.ndim == 0
    loss.backward()
    assert rollouts.grad is not None and torch.isfinite(rollouts.grad).all()


def test_du_variant_penalizes_fast_motion_more():
    reg = CLAPRegularizer(variant="du_clap", chi=3.0, omega=1.0, N_star=0.9, gamma=0.0, kappa=0.0)
    slow = torch.zeros(2, 10, 2)
    fast = torch.cumsum(torch.full((2, 10, 2), 0.3), dim=1)
    assert reg(fast).item() > reg(slow).item()


def test_aclap_risk_gated_weights_applied():
    """Eq. 14: per-state γ(r)/κ(r) weights change the loss when gamma1/kappa1 > 0."""
    import torch.nn as nn

    # OOD head returns constant 0.5 -> r = 0.5 everywhere.
    class ConstOOD(nn.Module):
        def forward(self, z):
            return torch.full(z.shape[:-1], 0.5, dtype=z.dtype, device=z.device)

    # Moving rollout so barrier/accel terms are nonzero.
    B, H, d = 4, 12, 2
    rollouts = torch.cumsum(torch.full((B, H + 1, d), 0.1), dim=1)

    # Baseline: gamma1=kappa1=0 (risk weights do nothing).
    reg_base = CLAPRegularizer(
        variant="a_clap", ood_head=ConstOOD(),
        beta=1.0, gamma=0.5, kappa=0.1, N_star=0.9,
        gamma1=0.0, kappa1=0.0,
    )
    # With risk weights: gamma1=kappa1=2.0, r=0.5 -> effective gamma=1.5, kappa=1.1.
    reg_gated = CLAPRegularizer(
        variant="a_clap", ood_head=ConstOOD(),
        beta=1.0, gamma=0.5, kappa=0.1, N_star=0.9,
        gamma1=2.0, kappa1=2.0,
    )

    loss_base = reg_base(rollouts).item()
    loss_gated = reg_gated(rollouts).item()

    # Larger effective weights must produce a strictly larger running cost.
    assert loss_gated > loss_base, (
        f"Expected Eq. 14 risk-gated weights to raise cost: {loss_gated} <= {loss_base}"
    )
