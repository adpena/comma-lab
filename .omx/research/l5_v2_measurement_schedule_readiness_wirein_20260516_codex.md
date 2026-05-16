# L5 v2 measurement schedule readiness wire-in

- date: 2026-05-16
- scope: TT5L-first L5 v2 staircase readiness surface
- code: `src/tac/optimization/l5_staircase_v2.py`
- tests: `src/tac/tests/test_l5_staircase_v2.py`
- schedule_tool: `tools/build_l5_v2_lattice_measurement_schedule.py`
- schedule_artifact: `.omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.json`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

The L5 v2 measurement schedule existed as a durable artifact, but the TT5L
campaign readiness payload did not surface it. That made the first-match
C1/Z5/TT5L measurement lattice too easy to bypass or rediscover by hand, which
is exactly the repeated "tool exists but operator flow cannot see it" failure
class called out in `AGENTS.md`.

## Change

`l5_v2_tt5l_campaign_readiness` now exposes the measurement schedule tool,
JSON artifact, Markdown report, and explicit non-authority flags. When the next
non-PR106 action is `populate_and_evaluate_c1_z5_tt5l_probe_observations`, the
action payload also includes a concrete measurement-schedule command template
that consumes the probe-observation intake artifact and emits the canonical
schedule artifacts.

## Authority

This is a planning and operator-discoverability change only. The measurement
schedule routes the next paired C1/Z5/TT5L probes; it does not create a score,
promotion, rank, dispatch, or architecture-lock claim.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_l5_v2_measurement_schedule.py -q`
- `.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py src/tac/optimization/l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_measurement_schedule.py tools/build_l5_v2_lattice_measurement_schedule.py`
