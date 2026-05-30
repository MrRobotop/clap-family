# Reproducibility Checklist

This document lists every quantity that must be reported in a paper or experiment using `clap_family`, following the paper's Appendix D checklist. Any submission using this codebase should fill in all items below.

Run the reference experiment with:

```bash
python -m clap_family.experiments.reproduce
```

or with figures:

```bash
python -m clap_family.experiments.reproduce --plots
```

Results are written to `results.json`. Figures (requires `[plots]`) are written to `fig_horizon_dwell.png` and `fig_learned_gate.png`.

---

## Required reporting items

### Latent space

- [ ] Dimensionality `d` of the latent state `z`
- [ ] How the latent space is defined (encoder architecture, training objective, or analytic construction)
- [ ] Whether the latent space is bounded or unbounded, and the approximate range of states visited during planning

### Metric G

- [ ] Whether G is identity (v1 default) or a learned/analytic Riemannian metric
- [ ] If learned: architecture, training data, and how G is made positive-definite
- [ ] Effect of G choice on speed ratio `q = ‖v‖²_G / c(z)²` and the log barrier

### Field definitions (V, U, C, O, E_θ)

- [ ] `V(z)`: value/reward head — architecture and training signal
- [ ] `U(z)`: epistemic uncertainty estimate — method (ensemble std, MC dropout, evidential, etc.)
- [ ] `C(z)`: safety cost — definition and normalization
- [ ] `O(z)`: out-of-distribution score — method (density model, reconstruction error, etc.)
- [ ] `c(z)`: latent speed limit — constant or state-dependent; if state-dependent, how it is computed
- [ ] `E_θ(z)`: transition-model error estimate — only required for `LearnedGateACLAP` / `PhaseAdaptiveLGACLAP`

### Conservative lapse hyperparameters (CLAPParams)

- [ ] `beta` (uncertainty weight)
- [ ] `eta` (safety cost weight)
- [ ] `zeta` (OOD score weight)
- [ ] `tau` (temperature)
- [ ] `gamma` (log-barrier weight)
- [ ] `kappa` (acceleration weight)
- [ ] `lambda_end` (terminal distance weight)
- [ ] `rho` (speed-ratio margin, keeps barrier off singularity)
- [ ] `cmax` (speed limit fallback)
- [ ] `a0` (acceleration normalization scale, for DU-CLAP / A-CLAP variants)

### Admissible set thresholds (AdmissibleSet)

- [ ] `Cmax` (maximum allowed safety cost)
- [ ] `Umax` (maximum allowed uncertainty)
- [ ] `Omax` (maximum allowed OOD score)
- [ ] Confirmation that the admissible set is non-empty over the grid used to estimate `N̄*`

### Variant-specific parameters

For `DUCLAP` / `AdaptiveDUCLAP`:
- [ ] `chi` (speed-dependent uncertainty coefficient)
- [ ] `omega` (acceleration-dependent uncertainty coefficient)
- [ ] `adaptive_gate` (mixing coefficient, AdaptiveDU only)

For `ACLAP`:
- [ ] `gamma1`, `kappa1` (risk-localized weight increments)
- [ ] Definition of `risk_field r(z)` (default: OOD score clipped to [0,1])

For `LearnedGateACLAP`:
- [ ] `alpha` (gate sharpness)
- [ ] `eps_model` (transition-error threshold)
- [ ] `E_theta` architecture and training signal

For `PhaseAdaptiveLGACLAP`:
- [ ] `alpha_access` (gate multiplier during access phase)
- [ ] `alpha_dwell` (gate multiplier during dwell phase)
- [ ] `dwell_radius` (distance threshold for dwell phase)
- [ ] How target states are provided at runtime

### MPC / solver configuration (ProjectedCEMMPC)

- [ ] `horizon` H (planning horizon)
- [ ] `rollout` (number of executed steps)
- [ ] `samples` (CEM population size)
- [ ] `elites` (number of elite samples)
- [ ] `iters` (CEM iterations per step)
- [ ] `init_std` (initial sampling standard deviation)
- [ ] `seed` (random seed for reproducibility)
- [ ] `dt` (time step)

### Experimental setup

- [ ] Environment: which of `ThreeRegionEnv`, `HardEnv`, `TransitionErrorEnv` or custom
- [ ] Number of independent seeds / runs
- [ ] Hardware and Python/NumPy version used

### Reported metrics

All experiments should report all applicable metrics from `clap_family.metrics`:

- [ ] `target_dwell` — fraction of trajectory time in the target region (paper T5)
- [ ] `trap_dwell` — fraction of trajectory time in the reward-trap region
- [ ] `unsafe_time` — fraction of trajectory time in the unsafe region (paper T2/T3)
- [ ] `transition_exposure` — mean transition-model-error along the trajectory (learned-gate variants)
- [ ] `jerk_proxy` — mean acceleration magnitude as a smoothness proxy (paper T6)

---

## Reference experiment parameters

The following are the default values used in `clap_family.experiments.reproduce`:

```
horizons:     (20, 40, 80, 120)
samples:      256
elites:       32
iters:        5
rollout:      120
seed:         0
variant:      RRLA (horizon sweep), LearnedGateACLAP (tradeoff)
env:          ThreeRegionEnv (horizon sweep), TransitionErrorEnv (tradeoff)
```
