# Operator Briefing Dispatch-Gate Hardening (2026-05-13)

## Summary

Hardened `tools/operator_briefing.py` so predicted score-target plausibility no
longer makes a gated or placeholder command active by itself.

The observed bug was `lane_pr106_stacked` appearing in
`active_composition_lanes` because its predicted low score was below `0.19`.
That row's own gate still requires latent + yshift + lrl1 sister lanes to land
empirically, and its one-liner still contains `<path/to/...>` placeholders.

## Fix

- Added `dispatch_routing` next to `score_target_routing`.
- `score_target_routing` remains a narrow prediction-band filter.
- `dispatch_routing.active` is false when:
  - a row has a `gate_condition` without `gate_ready=true`;
  - the operator one-liner contains unresolved `<...>` placeholders.
- Added top-level `ready_for_operator_dispatch` and
  `ready_for_exact_eval_dispatch` fail-closed fields for generic consumers.
- JSON active lists now filter on `dispatch_routing.active`, not only predicted
  score band.
- Wired all-lanes preflight Gate #28 to parse operator briefing JSON and fail
  if any active worklist row bypasses dispatch readiness, gate satisfaction, or
  placeholder checks.

## Verification

```bash
.venv/bin/python -m ruff check tools/operator_briefing.py src/tac/tests/test_operator_briefing.py
.venv/bin/python -m pytest -q src/tac/tests/test_operator_briefing.py src/tac/tests/test_score_target_filter.py src/tac/tests/test_all_lanes_operator_briefing_gate.py
.venv/bin/python tools/operator_briefing.py --json --top 3 --skip-provider-readiness
.venv/bin/python tools/all_lanes_preflight.py
```

Observed:

- `active_supplementary_lanes=[]`
- `active_gated_lanes=[]`
- `active_composition_lanes=[]`
- `lane_pr106_stacked.score_target_routing.active=true`
- `lane_pr106_stacked.dispatch_routing.active=false`
- `lane_pr106_stacked.ready_for_operator_dispatch=false`
- `lane_pr106_stacked.ready_for_exact_eval_dispatch=false`
- blockers:
  - `gate_condition_not_satisfied`
  - `operator_one_liner_has_unresolved_placeholders`

## Evidence Grade

Operator-routing hardening only. No score claim, no archive promotion, no GPU
dispatch.
