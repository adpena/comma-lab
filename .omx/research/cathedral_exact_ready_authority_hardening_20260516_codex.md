# Cathedral Exact-Ready Authority Hardening - 2026-05-16

## Finding

`tac.optimizer.exact_readiness.promoted_row()` emitted byte-closed rows with
`ready_for_exact_eval_dispatch=true` but without `dispatch_packet_ready=true`.
`tools/cathedral_autopilot_autonomous_loop.py` treats those as distinct
authority gates, so rows produced by the canonical exact-readiness promoter
could later be refused by the autonomous dispatch loop with
`dispatch_packet_ready_false`.

Classification: integration/authority bug. No score claim. No contest result
status changes.

## Fix

- Exact-readiness promoted rows now set `dispatch_packet_ready=true` alongside
  `ready_for_exact_eval_dispatch=true`.
- The optimizer exact-readiness test asserts the emitted dispatch row carries
  both gates.
- The Cathedral exact-ready queue loader test now models the full promoted-row
  authority surface, including `lane_id`, and proves
  `OperatorAuthorizedModeConfig.can_authorize()` succeeds for the loaded row.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py::test_load_exact_ready_queue_preserves_authority_after_live_audit \
  src/tac/tests/test_optimizer_exact_readiness.py::test_promotes_byte_closed_candidate_without_score_claim
```

Result: 2 passed.

```bash
.venv/bin/python -m ruff check \
  src/tac/optimizer/exact_readiness.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_optimizer_exact_readiness.py
```

Result: all checks passed.

## Follow-Up

This does not authorize dispatch by itself. Lane claims, provider/runtime
custody, exact archive/runtime hashes, and operator spend gates remain required
before any GPU or remote run.
