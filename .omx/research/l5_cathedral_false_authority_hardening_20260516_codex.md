# L5/Cathedral false-authority hardening (2026-05-16)

## Problem

Read-only adversarial review found three remaining authority leaks after the
L5-v2 custody fixes:

1. Composition-ranking JSON could carry
   `prediction_band_verdict.valid_for_rank_reward=false` while preserving a
   positive `expected_information_gain`, letting stale prediction-band priors
   influence Cathedral rank order.
2. `rank_axis="predicted_score_delta"` sorted on raw negative deltas even when
   prediction-band rank reward had already been suppressed.
3. CUDA-axis custody used substring matching, so negated strings such as
   `cpu-no-cuda` or `cuda-disabled` could satisfy CUDA token checks.

## Fix

- Cathedral composition-ranking ingestion now re-checks prediction-band verdicts,
  zeroes EIG when rank reward is blocked, and appends
  `prediction_band_rank_reward_suppressed`.
- `rank_candidates(..., rank_axis="predicted_score_delta")` now neutralizes rows
  already marked with `prediction_band_rank_reward_suppressed`.
- Composition-ranking ingestion now uses strict JSON boolean parsing for envelope
  and authority-adjacent flags instead of Python truthiness.
- Shared exact-eval custody and L5-v2 paired-axis gate checks now use
  non-negated CUDA capability token matching.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_exact_eval_custody.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py::test_rank_by_predicted_score_delta_neutralizes_suppressed_prediction_band \
  src/tac/tests/test_build_composition_ranking_json.py::test_autopilot_consumer_suppresses_blocked_prediction_band_eig \
  src/tac/tests/test_build_composition_ranking_json.py::test_autopilot_consumer_rejects_string_envelope_bools -q
```

Result: `40 passed`.

## Authority

This is rank/dispatch hardening only. It does not create a score claim,
promotion claim, or dispatch claim. Suppressed prediction bands remain visible
as planning annotations, but they cannot drive active EIG or score-delta rank
authority.
