# L5 v2 Dykstra Artifact Provenance Hardening

Date: 2026-05-16
Author: Codex
Scope: TT5L Dykstra score-axis sanity artifact

## Verdict

`score_claim=false`; `promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`.

The TT5L Dykstra score-axis artifact now requires tool provenance before it can
count as valid score-axis sanity.

## Failure Class

`dykstra_score_axis_artifact_handwritten_json_bypass`

The readiness checker validated the scalar Dykstra fields, but it did not bind
the JSON to the canonical tool, predicate identity, command arguments, UTC
generation time, or tool content hash. A hand-written field-complete JSON could
therefore look equivalent to a real Dykstra run.

## Landed Fix

`tools/check_substrate_dykstra_feasibility.py` now emits:

- `schema=dykstra_feasibility_verdict_v1`
- `predicate_id=dykstra_score_axis_feasibility_v1`
- `generated_by_tool=tools/check_substrate_dykstra_feasibility.py`
- `generated_at_utc`
- `command_argv`
- `tool_sha256`

`src/tac/optimization/l5_staircase_v2.py` requires those fields and verifies
that `tool_sha256` matches the canonical tool content under the repo root.

## Verification

```bash
.venv/bin/python -m ruff check \
  tools/check_substrate_dykstra_feasibility.py \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_check_substrate_dykstra_feasibility.py \
  src/tac/tests/test_l5_staircase_v2.py

.venv/bin/python -m pytest \
  src/tac/tests/test_check_substrate_dykstra_feasibility.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_l5_v2_measurement_schedule.py \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py \
  src/tac/tests/test_cathedral_autopilot.py -q
```

Observed:

- `ruff`: all checks passed
- `pytest`: `166 passed in 1.16s`

## Residual Risk

This hardens score-axis sanity custody. It remains planning-only and still
requires the separate TT5L move-level feasibility artifact before side-info
curves or timing smoke can advance.
