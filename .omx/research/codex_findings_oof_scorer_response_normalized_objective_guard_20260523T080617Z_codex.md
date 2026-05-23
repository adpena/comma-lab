# Codex Findings - OOF Scorer Response Normalized Objective Guard

timestamp_utc: 2026-05-23T08:06:17Z
agent: codex
lane_id: lane_codex_oof_scorer_response_normalized_objective_guard_20260523
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false

## Scope

Adversarial follow-up to the MLX scorer-response spend-gate and artifact-safety
landings. Kuhn's local-signal audit identified that OOF scorer-response
training and validation still read `row[target]` directly, so MLX window rows
could train or validate against raw window-local `delta_vs_baseline_score`
instead of the normalized full-video planning objective.

## Landed Fix

- Added `scorer_response_planning_value_for_target(...)` as the canonical target
  accessor for scorer-response planner, OOF training, validation, candidate
  family metrics, top-k metrics, and feature correlations.
- MLX scorer-response rows now fail closed for planner/training targets unless
  their normalized full-video objective fields are present and internally
  consistent.
- For MLX rows, the default `delta_vs_baseline_score` target maps to
  `projected_full_video_delta_vs_baseline_score`, not the raw window-local
  delta. Scorer-gain, scorer-delta, break-even-byte, and byte-margin targets
  map to their normalized full-video equivalents.
- `build_scorer_response_validation_gate(...)` now records the planning target
  accessor in its thresholds and blocks datasets with MLX rows missing a valid
  normalized planning target instead of silently marking predictions usable.
- `attach_out_of_fold_linear_predictions(...)` now trains on the same canonical
  planning target accessor and records it in `prediction_fit`.
- Nested prediction-fit, family-gate, and candidate-family metric payloads now
  carry explicit `score_claim_valid=false` false-authority markers.

## Regression Guards

- OOF training rejects MLX scorer-response rows missing normalized full-video
  objective fields.
- OOF family metrics ignore raw MLX window deltas and use projected full-video
  deltas for observed improvement and top-k utility.
- Validation gates block MLX scorer-response rows missing normalized objective
  fields.
- Validation gates can pass correlation while still refusing spend triage when
  projected full-video MLX deltas are non-improving, even if raw window deltas
  look strongly improving.
- Nested OOF/family metrics preserve `score_claim_valid=false`.

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py
```

Result: `101 passed in 0.92s`.

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_mlx_effective_spend_triage_selection.py \
  src/tac/tests/test_cross_family_candidate_portfolio.py
```

Result: `128 passed in 1.35s`.

```bash
.venv/bin/ruff check \
  src/tac/optimization/scorer_response_dataset.py \
  src/tac/optimization/scorer_response_prediction.py \
  src/tac/tests/test_scorer_response_dataset.py
git diff --check
```

Result: `ruff` passed and `git diff --check` was clean.

## Remaining Work

1. Route cathedral distilled-scorer and PACT-NeRV duplicate-validator consumers
   through this same target accessor or an equivalent normalized-objective
   contract.
2. Extend byte-shaving, PacketIR, and macOS advisory loader tests to reject
   truthy `score_claim_valid` and raw MLX target leakage.
3. Add an autopilot/reporting surface that counts how many candidate rows are
   excluded by normalized-objective target blockers before queue admission.
