"""Use CLAPRegularizer as a safety/smoothness regularizer on latent world-model rollouts.

Run: pip install clap_family[torch] && python examples/torch_world_model_reg.py
"""
import torch
from clap_family.torch import CLAPRegularizer

# Pretend a world model produced these latent rollouts: (batch, horizon+1, latent_dim)
rollouts = torch.randn(8, 21, 4, requires_grad=True)

clap_reg = CLAPRegularizer(variant="du_clap", beta=1.0, gamma=0.5, kappa=0.1,
                           chi=2.0, omega=1.0, N_star=0.9)
task_loss = (rollouts ** 2).mean()           # stand-in for your real training loss
total = task_loss + 0.1 * clap_reg(rollouts)
total.backward()
print(f"task={task_loss.item():.4f}  clap_reg={clap_reg(rollouts).item():.4f}  total={total.item():.4f}")
