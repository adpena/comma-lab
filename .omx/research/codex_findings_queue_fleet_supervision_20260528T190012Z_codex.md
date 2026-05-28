# Codex Findings: Queue Fleet Supervision

UTC: 2026-05-28T19:00:12Z

## Scope

Landed and exercised the queue-fleet control-plane surface. This belongs in
`comma_lab.scheduler`, not `tac`, because it discovers `.omx` /
`experiments/results` queue artifacts, classifies live queue health, and runs
bounded operator supervision. Reusable compression, scorer, archive, and
packet primitives remain in `tac`.

## Code Landed

- Commit: `cac9c8860` (`Add queue fleet supervisor surface`)
- Main surfaces:
  - `src/comma_lab/scheduler/queue_fleet.py`
  - `tools/queue_fleet.py`
  - operator briefing Phase 6i queue-fleet summary
- Boundary protection:
  - `src/comma_lab/README.md` documents `comma_lab.scheduler.queue_fleet`
  - `src/tac/tests/test_queue_fleet_tool.py` asserts the fleet surface lives in
    `comma_lab.scheduler.queue_fleet`

## Execution Anchor

- Command class: local-only bounded queue-fleet supervision
- Output root:
  `.omx/research/queue_fleet_supervisor_20260528T185903Z`
- Result artifact:
  `.omx/research/queue_fleet_supervisor_20260528T185903Z/fleet_supervisor_result.json`
- Result SHA-256:
  `22583682dfb33fd3e1ade9c7343c3bf34c10442498d7adb5542d4ec8bf706e5b`
- Result bytes: `209348`

## Outcome

- Selected queues: `4`
- Completed child supervisors: `4`
- Failed child supervisors: `0`
- Initial fleet status:
  `EMPTY_OR_IDLE=2, INVALID_QUEUE=19, NEEDS_INIT=2, NEEDS_RECOVERY=4, READY_TO_SUPERVISE=4, TERMINAL=9`
- Final fleet status:
  `EMPTY_OR_IDLE=2, INVALID_QUEUE=19, NEEDS_INIT=2, NEEDS_RECOVERY=4, READY_TO_SUPERVISE=2, TERMINAL=11`

Child outcomes:

- `frontier_final_rate_attack_fp11_canonical_chain_20260528T1800Z_post_execute_feedback_autonomous_chain_optimization`
  - Return code: `0`
  - Final reason: `max_ticks_reached`
  - Final counts: `succeeded=21`
- `frontier_final_rate_attack_fp11_canonical_chain_20260528T1800Z_post_execute_feedback_repair_campaign_score`
  - Return code: `0`
  - Final reason: `terminal_queue_state`
  - Final counts: `succeeded=48`
- `frontier_final_rate_attack_strict_exec_20260528T144357Z_post_execute_feedback_autonomous_chain_optimization`
  - Return code: `0`
  - Final reason: `max_ticks_reached`
  - Final counts: `queued=3, succeeded=18`
- `frontier_final_rate_attack_strict_exec_20260528T144357Z_post_execute_feedback_repair_campaign_score`
  - Return code: `0`
  - Final reason: `max_ticks_reached`
  - Final counts: `queued=6, succeeded=42`

## Authority

This is local queue telemetry and bounded local execution only. It is not score
authority, exact-eval authority, rank/kill authority, promotion authority, or
paid dispatch authority. All emitted payloads keep the false-authority fields
false.

## Next Action

Resume the remaining two `READY_TO_SUPERVISE` queues with the same fleet
surface, then harvest any new materializer or repair outputs into the existing
frontier final-rate feedback ledgers.
