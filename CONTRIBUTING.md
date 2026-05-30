# Contributing to clap_family

Thank you for your interest in contributing. This document encodes the collaborator contract for extending the package: variants, environments, backends, and training adapters each have a small prescribed pattern to keep the math DRY and the theorem contract intact.

---

## Quick setup

```bash
git clone https://github.com/rishabhpatil/clap-family
cd clap-family
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the suite:

```bash
pytest -q -W ignore::RuntimeWarning   # 72 passed + 2 skipped (torch tests skip without torch)
ruff check clap_family tests
```

---

## How to add a variant

Each new variant must:

1. **Subclass `BaseCLAP`** from `clap_family/variants/base.py` and **override exactly one hook**:

   | Hook | Signature | Override when you want to change |
   |---|---|---|
   | `_lapse_deficit(self, N_bar, q)` | `→ (H,) array` | How the deficit scales with speed |
   | `_effective_uncertainty(self, U0, q_state, a_sq_state, gate)` | `→ (H,) array` | How motion inflates uncertainty |
   | `_gate(self, z_states, dist, fields, frac)` | `→ (H,) array in [0,1]` | Risk gate `r(z)` |
   | `_speed_weight(self, gate)` | `→ (H,) array` | Per-state `γ` |
   | `_accel_weight(self, gate)` | `→ (H,) array` | Per-state `κ` |

   Example — a new variant that modulates the deficit by a custom factor:

   ```python
   # clap_family/variants/my_variant.py
   import numpy as np
   from .base import BaseCLAP

   class MyVariant(BaseCLAP):
       """MyVariant — describe the distinguishing math and cite the equation."""
       def _lapse_deficit(self, N_bar, q):
           return (self.params.N_star - N_bar) * (1.0 + q)  # example
   ```

2. **Add a reduction test** — every new variant must have at least one test showing it collapses to base `CLAP` at the appropriate zero-limit. See `tests/test_duclap.py` for the pattern:

   ```python
   def test_my_variant_reduces_to_clap_when_param_zero():
       ...
       assert np.isclose(MyVariant(param=0.0).action(z, fs), CLAP().action(z, fs))
   ```

3. **Export** — add `from .variants.my_variant import MyVariant` to `clap_family/__init__.py` and append to `__all__`.

4. **Docstring** must cite the paper equation it implements (e.g., `"Distinguishing math: ... (paper Eq. N)"`).

---

## How to add an environment

An environment is a `SimpleEnv` (from `clap_family/envs/base_env.py`) or a dataclass that provides:

- `fields: FieldSet` — the V/U/C/O/c (and optionally E_theta) field functions
- `start: np.ndarray` — starting state
- `target: np.ndarray` — target manifold `M*` as `(k, d)` array
- `N_star: float` — the maximum lapse value over the admissible region
- `in_target(z)`, `in_trap(z)`, `in_unsafe(z)` — boolean masks used by the metrics

See `clap_family/envs/three_region.py` for a complete example. Place new envs in `clap_family/envs/` and export from `clap_family/__init__.py`.

Each new environment needs at least one test checking region separation (target vs trap vs unsafe are disjoint, the target has positive value, etc.). See `tests/test_envs.py`.

---

## How to add a backend

v1 is NumPy-only. The `xp` seam (array backend) is a documented extension point for a future JAX or CuPy backend. To slot in a new backend:

1. The seam lives in `clap_family/geometry.py`, `clap_family/lapse.py`, and `clap_family/admissible.py` — all array operations go through these modules.
2. A backend adapter would replace `np` with `jnp` (or similar) at the module level; none of the variant hooks or the solver change.
3. The CI matrix currently tests Python 3.10–3.12 with NumPy. New backends should add a separate CI job and document their extra requirement in `pyproject.toml`.

No JAX or CuPy backend ships in v1; this is an open extension point.

---

## How to add a training adapter

Each training-framework adapter lives in **one lazily-imported file** under `clap_family/torch/` and must not be imported at package load time. The pattern:

1. Create `clap_family/torch/my_framework.py` with a single class that wraps `CLAPRegularizer`.
2. Guard framework imports at module level (not at import of `clap_family`):
   ```python
   import my_framework          # top of the file — will raise ImportError if absent
   from .regularizer import CLAPRegularizer
   ```
3. Add `my_framework` to `pyproject.toml` optional-dependencies as its own extra:
   ```toml
   [project.optional-dependencies]
   myframework = ["torch>=2.0", "my_framework>=X.Y"]
   ```
4. Write a test that auto-skips without the framework: `pytest.importorskip("my_framework")`.

See `clap_family/torch/lightning.py` and `tests/test_torch_adapters.py` for the reference pattern.

---

## PR quality bars

A PR will be accepted when all of the following hold:

- [ ] `pytest -q -W ignore::RuntimeWarning` passes (72+ passed, torch tests skip gracefully)
- [ ] `ruff check clap_family tests` is clean (no warnings)
- [ ] Every new public function or class has a one-line docstring citing the paper equation it implements
- [ ] New variants include a **reduction test** (collapses to base CLAP at zero-limit)
- [ ] New envs include a **region-separation test**
- [ ] New training adapters are **lazily imported** and have a `pytest.importorskip` guard
- [ ] No `dist/`, `*.egg-info`, `results.json`, or `*.png` artifacts committed (they are gitignored)

---

## Open research directions (paper §9)

The following are good-first-research-direction problems identified in the paper. If you want to take one on, open an issue first to discuss scope.

1. **Full discrete-to-continuous convergence** — the current discrete action (Eq. 22–23) converges to the continuous form (Eq. 3) as `Δt → 0`, but a tight convergence rate bound under finite `H` and bounded curvature is open.

2. **Moving-lapse tracking** — `N̄*` is estimated once on a fixed grid. An online update rule for `N̄*` as the planner moves through `Z` (non-stationary `V`/`U`/`C`/`O`) is needed for deployment.

3. **Stochastic CLAP** — extend the deterministic action to stochastic dynamics `z_{t+1} = f(z_t, u_t) + ε_t`; the log speed barrier and lapse deficit both need risk-adjusted versions.

4. **Robust CLAP under model ambiguity** — replace the point-estimate fields `V, U, C, O` with distributional fields and derive a worst-case lapse that is robust to field estimation error (connects to distributionally-robust optimization).

5. **Multi-agent CLAP** — extend the single-trajectory action to a joint action over `N` agents; the speed barrier becomes a coupled constraint and the target manifold is a product space.

6. **Learned latent metric G** — currently `G = I` (identity). Learning a task-aware Riemannian metric `G(z)` from data (e.g., via a neural metric network) and propagating it through the speed-ratio `q = ‖v‖²_G / c(z)²` is the most impactful single extension.
