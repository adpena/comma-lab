# L5 v2 first-anchor timing-smoke artifact contract

Date: 2026-05-16

Scope: TT5L / L5 v2 staircase readiness

## Summary

The TT5L campaign readiness surface no longer advances from paired CPU/CUDA
axis planning to `materialize_tt5l_exact_or_diagnostic_anchor_pair` solely from
side-info, probe, and paired-axis-plan artifacts. It now requires a separate
first-anchor timing-smoke custody artifact at:

` .omx/state/tt5l_first_anchor_timing_smoke.json`

The artifact is non-promotional. It only proves that the first-anchor timing
smoke was actually measured with paired axis intent, concrete provider/runtime
custody, an exact command argv, positive elapsed time, a rate metric, and a
hash-checked result artifact.

## Contract

Required schema fields:

- `schema = tt5l_first_anchor_timing_smoke_v1`
- `lane_id = lane_time_traveler_l5_autonomy_substrate_20260513`
- `predicate_id = tt5l_first_anchor_timing_smoke_rate_v1`
- `predicate_passed = true`
- `required_axes = ["contest_cpu", "contest_cuda"]`
- `provider`
- `hardware` or `gpu`
- `provider_call_id` or `call_id`
- `command_argv`
- `elapsed_seconds > 0`
- `seconds_per_epoch > 0` or `seconds_per_candidate > 0`
- `result_artifact_path` inside the repo and not a transient temp path
- `result_artifact_sha256` matching the result artifact bytes
- `score_claim = false`
- `promotion_eligible = false`
- `ready_for_exact_eval_dispatch = false`

## Why

This closes a false-authority gap in the L5 v2 staircase. A paired-axis plan is
not the same thing as a measured timing smoke. Without the separate artifact,
the operator surface could imply the next TT5L action was anchor materialization
even though no measured cost/runtime custody existed for the first-anchor smoke.

This is not a score claim, not promotion evidence, and not exact-eval dispatch
authority. It is a dispatch-readiness custody gate for the next measurement.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_all_lanes_operator_briefing_gate.py src/tac/tests/test_cathedral_autopilot.py -q`

Results: 91 L5 v2 tests passed; 158 focused L5/Cathedral/operator-briefing
tests passed.
