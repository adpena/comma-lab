# Codex Findings - Byte-Shaving Scorer-Response Ref Planning Guard

timestamp_utc: 2026-05-23T08:26:51Z
agent: codex
lane_id: lane_codex_byte_shaving_scorer_response_ref_planning_guard_20260523
score_claim: false
score_claim_valid: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Scope

Follow-up to the normalized-objective scorer-response guard series. The
byte-shaving signal-surface builder accepted scorer-response JSON files as
planning refs, but only recorded metadata. That preserved custody, but did not
prove the byte-shaving planner had interpreted scorer-response rows through the
canonical normalized full-video objective.

This was a signal-loss risk for MLX/window rows: a misleading raw
`delta_vs_baseline_score` could be carried into byte-shaving planning refs
without a reusable summary that showed the normalized target actually used by
downstream planning.

## Landed Fix

- `byte_shaving_signal_surface_builder` now normalizes scorer-response refs via
  `normalize_legacy_response_dataset_authority(...)`.
- Scorer-response refs now validate canonical planning targets through
  `scorer_response_planning_value_for_target(...)` before the byte-shaving
  signal surface is returned.
- Scorer-response refs carry a `planning_summary` with normalized best/worst
  deltas and improvement counts, plus `planning_target_accessor`.
- MLX scorer-response refs missing normalized full-video objective fields now
  fail closed instead of becoming byte-shaving signal-surface refs.
- Planning-only source refs, MLX calibration refs, atom refs, and the
  byte-shaving CLI markdown now echo `score_claim_valid=false`.

## Regression Guards

- A synthetic MLX scorer-response ref with raw `delta_vs_baseline_score=-10`
  but normalized projected full-video delta `+0.001` produces zero improvement
  counts and a positive best planning delta in the byte-shaving ref summary.
- The same MLX row without normalized objective fields raises a
  `ByteShavingCampaignError` containing `missing normalized full-video
  objective`.
- Existing candidate-queue, campaign-plan, optimizer-queue, and scorer-response
  dataset tests continue to pass together.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_byte_shaving_signal_surface_builder.py \
  src/tac/tests/test_byte_shaving_campaign.py \
  src/tac/tests/test_optimizer_candidate_queue.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_build_scorer_response_dataset_cli.py
```

Result: `142 passed in 1.64s`.

```bash
.venv/bin/ruff check \
  src/tac/optimization/byte_shaving_signal_surface_builder.py \
  src/tac/tests/test_byte_shaving_signal_surface_builder.py \
  tools/build_byte_shaving_signal_surface.py
git diff --check
.venv/bin/python tools/lane_maturity.py validate
```

Result: `ruff` passed, `git diff --check` was clean, and lane validation was
clean.

Review tracker:

- `policy-check` clean for
  `src/tac/optimization/byte_shaving_signal_surface_builder.py`,
  `src/tac/tests/test_byte_shaving_signal_surface_builder.py`, and
  `tools/build_byte_shaving_signal_surface.py`.

## Remaining Work

1. Apply the same scorer-response planning-target accessor audit to
   cross-family portfolio and any remaining exact-eval acquisition planners
   that ingest raw scorer-response rows directly.
2. Extend the queue worker so byte-shaving surfaces can feed autonomous
   local-first materialization/control/advisory sweeps without manual copying
   of JSON refs.
3. Preserve exact axis separation: byte-shaving signal surfaces remain
   planning-only until materialized packets, locality controls, and contest
   CPU/CUDA auth evals exist.
