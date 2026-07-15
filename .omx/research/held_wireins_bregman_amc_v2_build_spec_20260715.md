# HELD wire-ins drain v2 — implementation specification

## Finding 1: Bregman dual-metric correction

Objective: land the squared-Hessian dual-coordinate law and a fail-closed DSL
adoption guard for the single canonical metric `argmax_native_vjp_fidelity_v1`.

Authority:

- `.omx/research/codex_premise_falsification_bregman_dual_euclidean_20260714_codex.md`
- `.omx/research/bregman_v9_all_surfaces_measurement_20260714.json`
- broadcast correction `2026-07-14T14:16:00Z`

Required implementation:

1. Rebuild `tac.information_geometry.bregman_v9_surfaces` with deterministic
   NumPy helpers that distinguish all four quantities:
   `dtheta.T@H@dtheta`, `deta.T@solve(H,deta)`, `deta.T@deta`, and
   `dtheta.T@H@H@dtheta`. Validate shapes, finiteness, symmetry, and positive
   definiteness; the Fisher-natural cotangent path must call a typed linear
   solve, never an explicit inverse or a no-solve alias.
2. Add canonical equation id `bregman_dual_metric_squared_hessian_v1` with an
   `EmpiricalAnchor` for the retained deterministic 600-state measurement:
   600/600 raw-dual-vs-ordinary-Hessian mismatches, max primal/exact-dual error
   `5.684341886080802e-14`, and max raw-dual/squared-Hessian error
   `9.094947017729282e-13`. Axis must remain
   `MEASURED_LOCAL_CPU_SYNTHETIC_MATH_FIXTURE_NOT_SCORE`; real-n600 selection is
   `NO_VERDICT_DATA_CUSTODY`; `score_claim=false` and promotion is forbidden.
   Provide explicit build and populate functions with no import-time state
   mutation.
3. Add a guard under `tac.witness_dsl` and resolve it through the existing
   `lever_registry` (not a parallel registry, not a trainer Lever). The sole
   registered metric id is `argmax_native_vjp_fidelity_v1`. A canonical metric
   adoption must explicitly declare Fisher-natural cotangent geometry, a typed
   `H^-1`/linear solve, `solve_elided=false`, and raw dual Euclidean scoped only
   to `squared_hessian_H_squared_only`. Missing/duplicate/unknown/shortcut
   bindings fail closed. No new trainer flags.
4. Add behavior tests for numerical identities, invalid/non-SPD inputs, guard
   acceptance and shortcut rejection, exact registry resolution, canonical
   equation anchor custody, temporary-registry population, and no duplicate
   equation id in the live registry.
5. Review correction: the identity-bearing aggregate must fail closed unless
   `delta_eta = H @ delta_theta` within a documented fp64 tolerance. The
   canonical producer list and DSL metric descriptor must contain no nonexistent
   module: use only the landed information-geometry helper and the sealed
   `.omx/research/bregman_v9_all_surfaces_binding_20260714.json` artifact, and
   make registry resolution fail if that artifact is missing.

Allowed files:

- `src/tac/information_geometry/**`
- `src/tac/canonical_equations/bregman_v9_surfaces_20260714.py`
- `src/tac/canonical_equations/tests/test_bregman_v9_surfaces_20260714.py`
- `src/tac/canonical_equations/__init__.py` only if needed for package query API
- `src/tac/witness_dsl/bregman_dual_metric_guard.py`
- `src/tac/witness_dsl/lever_registry.py`
- `src/tac/tests/test_bregman_dual_metric_guard.py`

Do not touch the trainer, `witness_autoconfig`, `spec_v9_cgauge`, scorer
surrogate, hot run files, frontier pointer, or any flags.

Acceptance:

```bash
.venv/bin/python -m pytest -q \
  src/tac/canonical_equations/tests/test_bregman_v9_surfaces_20260714.py \
  src/tac/tests/test_bregman_dual_metric_guard.py \
  src/tac/tests/test_lever_registry.py
.venv/bin/python -m ruff check \
  src/tac/information_geometry \
  src/tac/canonical_equations/bregman_v9_surfaces_20260714.py \
  src/tac/canonical_equations/tests/test_bregman_v9_surfaces_20260714.py \
  src/tac/witness_dsl/bregman_dual_metric_guard.py \
  src/tac/witness_dsl/lever_registry.py \
  src/tac/tests/test_bregman_dual_metric_guard.py
```

## Finding 2: AMC per-row tiered code allocation

This is intentionally deferred until Finding 1 is reviewed and committed. It
will be a separate coherent commit. The canonical equation id is
`amc_perrow_tiered_code_bitalloc_v1`; the tool is an existing measurement
producer, so no DSL Lever is required. `TieredCodeQATLever` remains OWED because
its two specified trainer flags do not exist. Reactivation requires both a
competitive witness checkpoint and operator GO to add the train-time flags
through the typed DSL.
