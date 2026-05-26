# Targeted Component Correction Budget Queue Finding

Generated at: 2026-05-26T02:33:04Z

## Summary

Receiver-closed rate wins now feed a queue-owned targeted correction acquisition
surface instead of remaining as advisory JSON. The queue converts byte savings
from receiver/runtime-proven materializers into local-only SegNet/PoseNet
component probes across pair, frame, region, boundary, batch, full-video, and
inverse-scorer operation families.

## Engineering Changes

- Added targeted component-correction acquisition schemas and queue metadata to
  `frontier_rate_attack_feedback.py`.
- Wired receiver-closed budget rows into two bounded queue paths:
  work-order emission and local CPU component advisory, with optional MLX
  response/cache steps for research signal only.
- Added `tools/build_frontier_targeted_component_correction_work_order.py` for
  deterministic queue handoffs.
- Extended feedback refresh output to write
  `targeted_component_correction_acquisition.json` and
  `targeted_component_correction_queue.json`.
- Fixed receiver-closed budget discovery so closures under
  `experiments/results/frontier_final_rate_attack` are harvested by default.

## Live Artifact

The refreshed frontier-results run wrote:

- `.omx/research/frontier_rate_attack_feedback_refresh_20260526T_targeted_component_correction_frontier_results_v2/receiver_closed_correction_budget.json`
- `.omx/research/frontier_rate_attack_feedback_refresh_20260526T_targeted_component_correction_frontier_results_v2/targeted_component_correction_acquisition.json`
- `.omx/research/frontier_rate_attack_feedback_refresh_20260526T_targeted_component_correction_frontier_results_v2/targeted_component_correction_queue.json`

That run harvested 414 receiver-closed bytes across packet-member merge and ZIP
header elision, generated 10 component-correction acquisition rows, and selected
2 local queue experiments. The generated merge work order carries 258 saved
bytes, five sibling correction families, and a false spend gate.

## Verification

- `ruff check` passed on the touched scheduler, CLI, work-order tool, and test file.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` passed: 17 tests.
- `tools/experiment_queue.py validate` passed for the generated DQS1 follow-up,
  receiver repair, and targeted component-correction queues.
- A live targeted component-correction work order was emitted for
  `packet_member_merge_v1` with `score_claim=false`,
  `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Authority Boundary

The receiver-closed rate budget is an acquisition signal only. It is not score,
promotion, rank/kill, budget-spend, or paid-dispatch authority. Local CPU and
MLX component responses may prioritize follow-up work, but exact CPU/CUDA auth
eval remains required before any score claim or submission action.
