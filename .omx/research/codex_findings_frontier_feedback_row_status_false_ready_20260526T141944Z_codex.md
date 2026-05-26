# Codex Findings: Frontier Feedback Row Status False Ready

Date: 2026-05-26T14:19:44Z

## Summary

The aggregate frontier feedback briefing correctly reported
`AUTONOMOUS_CHAIN_QUEUE_BLOCKED` when repair-waterfill and autonomous-chain child
queues existed but were frozen. The per-refresh row still reported
`AUTONOMOUS_CHAIN_QUEUE_READY` because it checked only artifact path presence.

That was a false-readiness surface: a blocked or frozen child queue could look
runnable in row-level consumers even though queue authority remained false.

## Fix

`tools/operator_briefing.py` now derives the per-refresh autonomous-chain status
from child queue health:

- both child queues must be `READY_LOCAL_QUEUE` for
  `AUTONOMOUS_CHAIN_QUEUE_READY`;
- present but frozen/error/blocking child queues report
  `AUTONOMOUS_CHAIN_QUEUE_BLOCKED`;
- a single autonomous queue is only `AUTONOMOUS_CHAIN_QUEUE_PARTIAL` when that
  queue itself is runnable.

The regression in `test_operator_briefing.py` now asserts the latest refresh row
status is blocked when both child queues are frozen.

## Verification

- `ruff check tools/operator_briefing.py src/tac/tests/test_operator_briefing.py`
- `pytest src/tac/tests/test_operator_briefing.py::test_operator_briefing_surfaces_repair_waterfill_action_functional_queue -q`

No score, promotion, rank/kill, GPU launch, or exact-dispatch authority is
granted by this briefing path.
