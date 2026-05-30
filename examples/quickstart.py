"""Plan in the 3-region env with DU-CLAP and print theorem-aligned metrics."""
from clap_family import DUCLAP, ThreeRegionEnv, ProjectedCEMMPC, target_dwell, trap_dwell, unsafe_time

env = ThreeRegionEnv()
solver = ProjectedCEMMPC(samples=256, elites=32, iters=5, init_std=0.4, seed=0)
traj = DUCLAP(N_star=env.N_star, cmax=1.0, chi=2.0, omega=1.0, lambda_end=10.0).plan(
    env, horizon=40, rollout=80, solver=solver)

print(f"target dwell: {target_dwell(traj, env):.3f}")
print(f"trap dwell:   {trap_dwell(traj, env):.3f}")
print(f"unsafe time:  {unsafe_time(traj, env):.3f}")
