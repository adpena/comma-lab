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

Objective: register `amc_perrow_tiered_code_bitalloc_v1` as an advisory,
checkpoint-scoped equation without inventing a train-time lever.

Authority:

- `.omx/research/fable_amc_saliency_codex.md` sections 1, 4-6, and 8
- `.omx/research/witness_sensitivity_bitalloc_336_20260713.md`
- `.omx/research/sub015_DAG_cheapen_real95_tilehalo_fp16_20260713.md` lines
  103-120 (the durable 6/6 byte-identity custody summary)

Required implementation:

1. Add only
   `src/tac/canonical_equations/amc_perrow_tiered_code_bitalloc_20260714.py`
   and its test. Do not edit or rebuild
   `tools/apply_amc_saliency_tiered_bitalloc_witness.py`.
2. The callable law must encode the exact pair-local SegNet composition:
   `d_seg(q) = sum_i mismatch_i(q_i) / sum_i pixel_count_i`. Validate finite,
   integral, non-negative mismatch/pixel counts; refuse empty inputs,
   mismatched lengths, zero pixels, or mismatches greater than pixels. This law
   is exact only because each pair's frame-1 code row affects that pair's
   SegNet row; it does not claim PoseNet or Brotli-rate additivity.
3. Register two explicit anchors:
   - 07-13 measured n600 baseline: checkpoint SHA
     `2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c`,
     GT SHA `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`,
     63,664 measured archive bytes, measured d_seg
     `0.03365824381510417`, and measured d_pose `151.79642088984443`.
   - 07-14 tiered response anchor: six tiered archive byte counts are MEASURED;
     baseline/uniform custody is byte-identical `6/6`; d_seg rows are
     `DERIVED_EXACT_FROM_MEASURED_PER_PAIR_ROWS`; pairkkt is 52,981 B and
     0.03152334, AMC saliency is 52,762 B and 0.03401882, random is 52,992 B
     and 0.03392613, and uniform int4 is 0.03380370. Record the scoped empirical
     conclusion: response allocation dominates proxy saliency on this
     `INSTANCE x FORMULATION`; it is not a family theorem.
4. Axis is exactly
   `[macOS-CPU advisory; NumPy-fp32 receiver; CPU frozen scorers]`;
   `score_claim=false`, `promotion_eligible=false`, pointer unchanged, fresh
   joint n600 d_pose/d_seg and exact contest-CPU transfer OWED. The missing raw
   result directories in this worktree must not be disguised: provenance and
   anchor source artifacts are the committed research memos above.
5. Canonical producer is the existing measurement tool. Canonical consumers
   are the existing measured reverse-waterfill equation/#157 path and existing
   byte-allocation surfaces only; every dotted code path must exist.
6. `TieredCodeQATLever` is explicitly `OWED_NOT_BUILT`; its nonexistent
   `--code-row-bits-map` and `--code-qat-tiered` flags must appear only as
   owed-design metadata, never in DSL emission/reference/factory surfaces.
   Reactivation requires a competitive witness checkpoint plus operator GO to
   add both train-time flags through the typed DSL.
7. Tests must cover the exact additive law and all validation failures,
   anchor labels/numbers/custody, advisory/no-score boundary, explicit
   population into a temporary registry, live-registry duplicate refusal, tool
   existence, consumer existence, and absence of the two owed flags from
   `lever_registry.dsl_referenced_flags()`, `dsl_emitted_flags()`, and
   `lever_factories()`.

Acceptance:

```bash
.venv/bin/python -m pytest -q \
  src/tac/canonical_equations/tests/test_amc_perrow_tiered_code_bitalloc_20260714.py \
  src/tac/tests/test_lever_registry.py
.venv/bin/python -m ruff check \
  src/tac/canonical_equations/amc_perrow_tiered_code_bitalloc_20260714.py \
  src/tac/canonical_equations/tests/test_amc_perrow_tiered_code_bitalloc_20260714.py
```
