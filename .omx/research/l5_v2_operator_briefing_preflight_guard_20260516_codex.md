# L5-v2 operator briefing preflight guard - 2026-05-16

## Scope

This landing extends `tools/all_lanes_preflight.py` so the operator-briefing
dispatch gate validates the new `l5_v2_frontier_readiness` section. The guard
prevents the L5-v2 / PR106 PacketIR briefing surface from becoming implicit
score, promotion, rank/kill, or exact-dispatch authority.

## Guarded Invariants

`l5_v2_frontier_readiness` must:

- exist in `tools/operator_briefing.py --json`;
- use schema `pact.l5_v2_frontier_readiness.v1`;
- keep `score_claim=false`;
- keep `promotion_eligible=false`;
- keep `rank_or_kill_eligible=false`;
- keep `ready_for_exact_eval_dispatch=false`;
- mark target rows as fail-fast only when any target exists.

Every sampled PR106 target row must keep:

- `score_claim=false`;
- `promotion_eligible=false`;
- `ready_for_exact_eval_dispatch=false`;
- `dispatch_status=requires_claim_lane_dispatch_before_provider_launch`.

## Verification

```text
.venv/bin/ruff check tools/all_lanes_preflight.py \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py
All checks passed.

.venv/bin/python -m pytest src/tac/tests/test_all_lanes_operator_briefing_gate.py -q
16 passed in 0.18s

.venv/bin/python -m pytest \
  src/tac/tests/test_operator_briefing.py \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py -q
33 passed in 59.87s
```

## Note

This is a guardrail landing only. It does not claim a score and does not launch
Modal, Kaggle, or any other provider job.
