# L5 V2 Probe Intake To Paired Measurement Action Hardening

score_claim=false
promotion_eligible=false
rank_or_kill_eligible=false
ready_for_exact_eval_dispatch=false
dispatch_attempted=false

## Purpose

The L5-v2 TT5L campaign surface previously risked retreading the probe-intake
step even when a valid fail-closed observation intake artifact already existed.
That is a local-minimum behavior: it spends operator attention regenerating a
known diagnostic artifact instead of advancing to the next material evidence
gate.

This hardening adds an explicit probe-observation-intake status reader and
changes the next action when intake is already valid but the probe gate remains
unmet:

`materialize_l5_v2_paired_probe_measurements`

That action points to the existing lattice schedule builder, paired dispatch
plan builder, and the remaining byte-closed archive/runtime fill-ins required
before any paired CPU/CUDA work can run.

## Evidence

- Code:
  `src/tac/optimization/l5_staircase_v2.py`
- Tests:
  `src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_probe_action_uses_existing_fail_closed_intake`
- Existing intake artifact:
  `.omx/research/l5_v2_probe_observation_intake_20260516_codex.json`
- Existing paired dispatch plan artifact:
  `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.json`

## Dispatch Meaning

No dispatch authority is added. The paired work units remain blocked until each
candidate supplies:

- byte-closed archive path;
- archive SHA-256;
- submission runtime or inflate runtime path;
- operator execution intent;
- paired CPU/CUDA evidence after dispatch and harvest.

The point is to make the next L5-v2 action frontier-moving: materialize
byte-closed C1/Z5/TT5L probe packets, not regenerate the fail-closed intake.

## Verification

Focused verification run before commit:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_probe_action_uses_existing_fail_closed_intake \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_probe_action_advances_after_template_exists \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_readiness_surfaces_measurement_schedule_without_authority \
  src/tac/tests/test_operator_briefing.py::test_briefing_json_composite_has_all_three_keys -q
```

Result: `4 passed`.
