"""Differentiable CLAP running-cost action as a training regularizer (paper §8, Eq. 23).

Recomputes the **running-integral portion** of the discrete CLAP action (Eq. 23) — lapse
deficit + log speed barrier + acceleration penalty — over a batch of latent trajectories
(B, H+1, d) in pure torch so it can be added to any training loss and backpropagated.

Note: the terminal target-distance term λ_end·d(z_H, M*)² (Eq. 23, term 4) is intentionally
omitted: a training regularizer operates over free-form rollouts with no target manifold.

Fields may be torch nn.Modules (value/uncertainty/OOD/transition-error heads) or plain
callables returning tensors.
"""
import torch
import torch.nn as nn

from ..params import CLAPParams


def _sigmoid(x):
    return torch.sigmoid(x)


class CLAPRegularizer(nn.Module):
    """Add `lam * CLAPRegularizer(...)(latent_rollouts)` to your task loss.

    Computes the CLAP running-cost action (lapse deficit + speed barrier + acceleration
    penalty) over latent rollouts. The terminal target-distance term is not included.

    variant: one of 'clap','rrla','du_clap','adaptive_du_clap','a_clap'.
    Heads (optional callables z:(...,d)->(...,)): value_head, uncertainty_head, cost_head,
    ood_head, speed_head (defaults: value 0, others 0, speed limit 1).

    For the 'a_clap' variant the per-state risk r=clamp(O,0,1) risk-localizes both the
    dynamic-uncertainty gate and the barrier/accel weights:
        γ(r) = gamma + gamma1*r,  κ(r) = kappa + kappa1*r  (Eq. 14)
    """

    def __init__(self, variant="clap", params=None,
                 value_head=None, uncertainty_head=None, cost_head=None,
                 ood_head=None, speed_head=None, transition_head=None, **kwargs):
        super().__init__()
        self.variant = variant
        self.params = params if params is not None else CLAPParams(**kwargs)
        self.value_head = value_head
        self.uncertainty_head = uncertainty_head
        self.cost_head = cost_head
        self.ood_head = ood_head
        self.speed_head = speed_head
        self.transition_head = transition_head

    def _field(self, head, z, default):
        if head is None:
            return torch.full(z.shape[:-1], float(default), dtype=z.dtype, device=z.device)
        return head(z)

    def forward(self, rollouts):
        """rollouts: (B, H+1, d) -> scalar mean action over the batch."""
        p = self.params
        z = rollouts
        states = z[:, :-1, :]                              # (B,H,d)
        v = (z[:, 1:, :] - z[:, :-1, :]) / p.dt            # (B,H,d)
        a = (z[:, 2:, :] - 2 * z[:, 1:-1, :] + z[:, :-2, :]) / (p.dt ** 2)  # (B,H-1,d)
        c = self._field(self.speed_head, states, 1.0)      # (B,H)
        q = (v * v).sum(-1) / (c * c + p.eps)
        q = torch.clamp(q, 0.0, 1.0 - p.rho)
        a_sq = torch.zeros_like(q)
        a_sq[:, 1:] = (a * a).sum(-1)
        V = self._field(self.value_head, states, 0.0)
        U0 = self._field(self.uncertainty_head, states, 0.0)
        C = self._field(self.cost_head, states, 0.0)
        O = self._field(self.ood_head, states, 0.0)  # noqa: E741

        # Per-state risk for a_clap (reused for U gate and Eq. 14 weights).
        r = torch.clamp(O, 0.0, 1.0)

        if self.variant in ("du_clap",):
            U = U0 + p.chi * q + p.omega * a_sq / (p.a0 ** 2)
        elif self.variant in ("adaptive_du_clap",):
            U = U0 + p.adaptive_gate * (p.chi * q + p.omega * a_sq / (p.a0 ** 2))
        elif self.variant in ("a_clap",):
            U = U0 + r * (p.chi * q + p.omega * a_sq / (p.a0 ** 2))
        else:
            U = U0

        N_bar = _sigmoid((V - p.beta * U - p.eta * C - p.zeta * O) / p.tau)
        deficit = p.N_star - N_bar
        if self.variant == "rrla":
            deficit = deficit / torch.sqrt(1.0 - q)

        # Eq. 14: risk-localized barrier/accel weights for a_clap; scalar otherwise.
        if self.variant == "a_clap":
            gamma_t = p.gamma + p.gamma1 * r   # (B,H)
            kappa_t = p.kappa + p.kappa1 * r   # (B,H)
        else:
            gamma_t = p.gamma
            kappa_t = p.kappa

        barrier = -gamma_t * torch.log(1.0 - q)
        accel = 0.5 * kappa_t * a_sq
        integrand = (deficit + barrier + accel) * p.dt        # (B,H)
        return integrand.sum(dim=1).mean()
