# Operation Portfolio Exact-Readiness Bridge

Date: 2026-05-25T23:57:22Z
Lane: codex_frontier_operation_portfolio_acquisition_20260525
Status: landed-local-artifact-pending-commit

## What changed

The frontier operation portfolio now preserves existing materializer exact-readiness bridge reports instead of stopping at candidate-manifest evidence. Materializer rows summarize bridge report counts, ready/blocked candidate counts, and the top exact-readiness blockers. The combined DFL1 + merge + header-elide chain row aggregates those reports so receiver/runtime blockers are queue-visible.

This keeps the rate budget signal coupled to the receiver/exact-readiness boundary: bytes saved can fund targeted SegNet/PoseNet repairs only after runtime consumption, full-frame parity where required, archive/runtime manifests, and exact auth-axis gates are satisfied.

## Live artifacts

Portfolio refresh:
`.omx/research/frontier_operation_portfolio_20260525T235200Z/operation_portfolio.json`

Validated follow-up queue:
`.omx/research/frontier_operation_portfolio_20260525T235200Z/dqs1_followup_queue.json`

Current-frontier chain handoff:
`.omx/research/frontier_chain_receiver_handoff_20260525T235229Z/experiment_queue.json`

## Current evidence

The richer portfolio has 32 operations, 5 queue-executable operations, and 14 follow-up signals. Its top chain remains `chain_dfl1_merge_header_elide_minimal_envelope`, with 794 observed saved bytes from known parts.

The chain now records 3 exact-readiness bridge reports, 3 blocked candidates, and 0 ready candidates. Top blockers include missing archive/runtime manifests, missing `inflate.sh` / `report.txt`, unresolved receiver-contract blockers, missing runtime-tree hashes, and active-floor byte blockers. That means the chain is high-value acquisition signal, not dispatch authority.

The generated current-frontier handoff queue validated with 1 experiment and 3 steps. It showed only `packet_member_zip_header_elide_v1` is executable against the canonical current frontier archive; DFL1/merge do not currently derive executable contracts on that archive. The system now preserves that distinction instead of pretending the robust-current multi-op chain applies everywhere.

## Authority

All emitted artifacts remain false-authority:
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`

## Next concrete work

1. Build the missing archive/runtime manifest and source-runtime adapter closure for the DFL1 + merge chain.
2. Convert the exact-readiness blocker summary into a prioritized receiver repair queue, starting with runtime tree hashes and full-frame parity context.
3. Keep the geometry/drop-many queue executable path running locally as the component-aware repair-budget explorer.
