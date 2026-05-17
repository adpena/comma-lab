# L5 v2 TT5L Materialized Work-Unit Builder - 2026-05-17

## Purpose

Replace hand-authored TT5L paired-dispatch JSON with a normal operator builder
that materializes the plan from real archive/runtime custody.

This is a no-score-claim hardening and frontier-actuation step. It does not
dispatch Modal jobs, does not claim improvement, and does not promote a lane.

## Change

Added:

- `src/tac/deploy/modal/paired_dispatch.py`
  - shared `paired_auth_eval_axis_command(...)`;
  - shared Modal CPU/CUDA wrapper and remote submission-dir constants.
- `src/tac/optimization/l5_v2_tt5l_materialized_work_unit.py`
  - builds the L5 v2 TT5L materialized paired CPU/CUDA work-unit plan;
  - computes archive bytes/SHA-256;
  - computes per-axis Modal-uploaded runtime tree and content-tree SHA-256s;
  - emits no-authority fields: `score_claim=false`,
    `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`,
    `ready_for_provider_dispatch=false`, `dispatch_attempted=false`.
- `tools/build_l5_v2_tt5l_materialized_paired_work_unit.py`
  - operator CLI that selects a TT5L side-info variant from the variant
    manifest and writes the canonical JSON/Markdown plan artifacts.
- `src/tac/tests/test_l5_v2_tt5l_materialized_work_unit_builder.py`
  - regression coverage for full-contest nonzero TT5L materialization and
    variant-manifest SHA verification.

The existing paired dispatcher now uses the shared per-axis command builder so
the materializer and dispatcher cannot drift in command shape.

## Live Artifact Refresh

Command:

```bash
.venv/bin/python tools/build_l5_v2_tt5l_materialized_paired_work_unit.py
```

Result:

- JSON: `.omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.json`
- Markdown: `.omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.md`
- archive:
  `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_20260517_codex/random_lsb/archive.zip`
- archive bytes: `38911`
- archive sha256:
  `b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7`
- materialized variant: `random_lsb`
- TT5L pair count: `600`
- TT5L side-info nonzero values: `27000`

The previous materialized packet pointed at the trained/source TT5L archive,
whose side-info was all-zero. That history remains preserved in git and in the
earlier materialized-plan memo. The refreshed packet intentionally points at the
full-contest `random_lsb` side-info control because it is the first existing
full-contest archive that proves nonzero side-info bytes can enter a paired
CPU/CUDA effect-curve measurement without becoming a score claim.

## Readiness Check

Current validator receipt:

```text
artifact_valid=True
blockers=[]
archive_path=experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_20260517_codex/random_lsb/archive.zip
archive_sha256=b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7
num_pairs=600
nonzero_values=27000
ready_for_exact_eval_dispatch=False
next_action=review_and_execute_l5_v2_tt5l_materialized_paired_measurement
ready_for_operator_dispatch=True
score_claim=False
provider_dispatch=False
```

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_l5_v2_tt5l_materialized_work_unit_builder.py \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_probe_action_advances_after_work_unit_materialized \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_rejects_all_zero_sideinfo \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_rejects_noncontest_pair_count \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_rejects_weak_axis_commands \
  -p no:cacheprovider
```

Result: `6 passed`.

Static checks:

```bash
.venv/bin/ruff check --select F,E9 \
  src/tac/deploy/modal/paired_dispatch.py \
  tools/dispatch_modal_paired_auth_eval.py \
  src/tac/optimization/l5_v2_tt5l_materialized_work_unit.py \
  tools/build_l5_v2_tt5l_materialized_paired_work_unit.py \
  src/tac/tests/test_l5_v2_tt5l_materialized_work_unit_builder.py
```

Result: `All checks passed!`.

```bash
.venv/bin/python -m py_compile \
  src/tac/deploy/modal/paired_dispatch.py \
  tools/dispatch_modal_paired_auth_eval.py \
  src/tac/optimization/l5_v2_tt5l_materialized_work_unit.py \
  tools/build_l5_v2_tt5l_materialized_paired_work_unit.py
```

Result: pass.

```bash
git diff --check
```

Result: pass.

## 6-Hook Wire-In

1. Sensitivity-map contribution: non-binding. This is dispatch-custody plumbing,
   not a scorer sensitivity primitive.
2. Pareto constraint: active. The plan is refused unless both CPU and CUDA axes
   share one archive SHA and review-only no-authority flags.
3. Bit-allocator hook: non-binding. No byte allocation policy changed.
4. Cathedral autopilot dispatch hook: active via
   `review_and_execute_l5_v2_tt5l_materialized_paired_measurement`.
5. Continual-learning posterior update: deferred until the paired exact cells
   are executed and harvested.
6. Probe-disambiguator: active indirectly. This materializes the TT5L side-info
   effect-curve control needed before the L5 v2 probe can treat side-info
   consumption as evidence.

## Remaining Work

- Operator review and explicit `--execute` are still required before Modal spend.
- The random-LSB packet is a side-info consumption/control measurement, not a
  trained usefulness proof.
- Paired CPU/CUDA result artifacts must be harvested and routed through the
  side-info effect curve before architecture-lock, rank, kill, promotion, or
  stack authority.
