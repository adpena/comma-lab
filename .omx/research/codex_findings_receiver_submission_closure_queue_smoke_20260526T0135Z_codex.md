# Receiver Submission Closure Queue Smoke - 2026-05-26T01:35Z

## Scope

Follow-up hardening for the materializer submission-runtime closure lane after
the receiver repair queue exposed an executable-vs-work-order gap.

## Changes

- Hardened `_submission_closure_candidate_ids_for_source_queue(...)` so queue
  compilation still emits an explicit closure attempt when a bridge report
  points at a source queue path that is not readable from the current repo root.
  When the source queue is readable, closure execution is filtered to rows whose
  receiver contract is already satisfied.
- Regenerated the live frontier operation portfolio at
  `.omx/research/frontier_operation_portfolio_20260526T0132Z_submission_closure_hardened/`.
  The receiver-repair queue now has 4 experiments and 10 executable local steps.

## Verification

- `ruff check` on the touched scheduler surface passed.
- Focused regression suite passed: 173 tests across frontier feedback,
  materializer chain harvest, submission closure, family-agnostic materializers,
  and exact readiness.
- Queue validation passed for
  `.omx/research/frontier_operation_portfolio_20260526T0132Z_submission_closure_hardened/receiver_repair_queue.json`.
- Queue execution passed with `failure_count=0`, `success_count=10`, and no
  orphaned steps in
  `.omx/state/experiment_queue_frontier_operation_portfolio_20260526T0132Z_receiver_repair_receiver_repair.sqlite`.

## Resulting Signal

Submission closure now clears the structural runtime-custody blockers for the
ZIP-header candidate. The remaining exact-readiness blocker for that candidate
is rate-floor policy only:

`above_active_floor_archive_bytes_without_operator_override:345646>185578`

Renderer DFL1 and packet-member-merge remain blocked by true receiver/runtime
proof gaps and should not be promoted through the closure path until their
full-frame parity/runtime-adapter proofs are repaired.

## Authority

All generated queue, closure, and bridge artifacts remain false-authority:
`score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
and `ready_for_exact_eval_dispatch=false`. No contest score changed in this
pass.
