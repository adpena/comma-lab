# Codex Findings: Decoder-Q Surface Advisory Gate

Generated: 2026-05-22T04:23:30Z

## Verdict

LANDED_FAIL_CLOSED_GATE.

The MLX decoder-q response surface remains useful as a planning prior, but it can no longer drive exact-eval spend unless a `tools/run_decoder_q_candidate_advisory_batch.py` artifact proves that at least one `response_surface_guided` fixed-length candidate improves local advisory score.

## Trigger

The first surface-guided waterbucket candidates were byte-valid and official-inflate-visible, but all three regressed on local macOS CPU advisory:

- `a2f90a216aac4184`: `+0.0004300000000000137`
- `a9b04920db67ec71`: `+0.0004830000000000112`
- `8f3a33e49b9b7906`: `+0.0005339143250268352`

That falsifies the prior standalone suppress/invert selector for exact-eval selection.

## Implementation

- Added `build_decoder_q_surface_advisory_gate(...)` in `tac.optimization.scorer_response_dataset`.
- Added `--decoder-q-surface-advisory-batch` to `tools/plan_ll_scorer_response_next.py`.
- The gate requires:
  - schema `fec6_decoder_q_candidate_advisory_batch_v1`
  - producer `tools/run_decoder_q_candidate_advisory_batch.py`
  - explicit false authority on batch/candidate/nested advisory surfaces
  - `bucket == response_surface_guided`
  - `fixed_length_runtime_compatible=true`
  - `length_delta == 0`
  - advisory success
  - reported delta consistent with `advisory_eval.canonical_score - inputs.baseline_score`
  - at least one improving surface-guided candidate

If these conditions fail, the planner emits:

- `do_not_dispatch_decoder_q_response_surface_after_advisory_regression`
- priority-1 probe `ll_decoder_q_surface_sign_calibration_repair`

## Live Artifact

Regenerated plan:

`experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/ll_next_probe_plan_same_axis_600rows_oof_validated_decoderq_surface_advisory_gated.json`

Observed gate:

- Status: `blocked`
- Exact-eval allowed: `false`
- Surface-guided candidates: 3
- Fixed-length surface-guided candidates: 3
- Improving surface-guided candidates: 0
- Best delta: `+0.0004300000000000137`

## Review Closure

The xhigh adversarial review flagged four issues before commit:

- missing fixed-length verification
- stale/sign-flipped delta trust
- shallow nested false-authority validation
- missing producer pinning

All four are now covered by regression tests.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_scorer_response_dataset.py -k 'decoder_q_surface_advisory or decoder_q_response_surface' -q` -> 6 passed
- `.venv/bin/python -m pytest src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_decoder_q_surface_objective.py src/tac/tests/test_mlx_score_calibration.py -q` -> 65 passed
- `git diff --check` on edited files -> clean

## Authority

This is a planner/spend gate only. It creates no score claim, promotion authority, rank/kill authority, or exact-eval readiness.
