# Packet Member Merge Receiver-Closed Budget Finding

Generated at: 2026-05-26T01:55:51Z

## Summary

`packet_member_merge_v1` is no longer trapped as a parser-only materializer signal.
The family materializer sweep can now compile the generated packet-member merge
receiver runtime from the source runtime, emit the runtime adapter proof, preserve
the runtime path/tree into the optimizer queue, and pass the standard static
submission-closure plus exact-readiness bridge path.

The resulting receiver-closed correction budget increased from 156 bytes to 414
bytes:

- `packet_member_merge_v1`: 258 receiver-closed bytes
- `packet_member_zip_header_elide_v1`: 156 receiver-closed bytes

This remains planning signal only. It does not claim score, promotion, rank/kill,
or exact-eval dispatch authority.

## Artifacts

- Receiver-backed merge sweep:
  `experiments/results/frontier_final_rate_attack/frontier_packet_merge_receiver_closed_20260526T015551Z/packet_member_merge_v1/sweep.json`
- Merge candidate manifest:
  `experiments/results/frontier_final_rate_attack/frontier_packet_merge_receiver_closed_20260526T015551Z/packet_member_merge_v1/rows/001_robust_current_correct_4dd46fed78ed/candidate.json`
- Generated receiver runtime:
  `experiments/results/frontier_final_rate_attack/frontier_packet_merge_receiver_closed_20260526T015551Z/packet_member_merge_v1/rows/001_robust_current_correct_4dd46fed78ed/candidate.runtime/`
- Runtime adapter proof:
  `experiments/results/frontier_final_rate_attack/frontier_packet_merge_receiver_closed_20260526T015551Z/packet_member_merge_v1/rows/001_robust_current_correct_4dd46fed78ed/candidate.runtime_consumption_proof.json`
- Optimizer source queue preserving runtime custody:
  `experiments/results/frontier_final_rate_attack/frontier_packet_merge_receiver_closed_20260526T015551Z/packet_member_merge_v1/exact_eval_handoff/source_queue.json`
- Static submission closure:
  `experiments/results/frontier_final_rate_attack/frontier_packet_merge_receiver_closed_20260526T015551Z/packet_member_merge_v1/submission_closure/submission_closure_report.json`
- Exact-readiness bridge:
  `experiments/results/frontier_final_rate_attack/frontier_packet_merge_receiver_closed_20260526T015551Z/packet_member_merge_v1/exact_readiness_bridge/exact_readiness_bridge_report.json`
- Refreshed portfolio:
  `.omx/research/frontier_operation_portfolio_20260526T015551Z_packet_merge_receiver_closed_budget/feedback_refresh_report.json`
- Receiver-closed budget:
  `.omx/research/frontier_operation_portfolio_20260526T015551Z_packet_merge_receiver_closed_budget/receiver_closed_correction_budget.json`

## Verification

- `tools/run_family_agnostic_materializer_sweep.py` produced a 258-byte
  `packet_member_merge_v1` candidate for
  `submissions/robust_current/archive_correct.zip` with
  `receiver_contract_satisfied=true`, `runtime_adapter_ready=true`, and no
  readiness blockers.
- `tools/build_optimizer_candidate_queue.py` preserved
  `candidate_runtime_dir`, `candidate_runtime_tree_sha256`, and
  `packet_member_merge_receiver_runtime_tree_sha256` in the queue row.
- `tools/build_materializer_submission_closure.py` built a contest-shaped
  submission using the generated receiver runtime, not the unmodified source
  runtime.
- `tools/run_materializer_exact_readiness_bridge.py` reduced the closed merge
  candidate to only safe planning blockers, with the active-rate floor still
  blocking exact dispatch.
- `tools/build_frontier_rate_attack_feedback_refresh.py` harvested both closed
  candidates into a 414-byte receiver-closed correction budget.
- `tools/experiment_queue.py validate` passed for both the generated DQS1 queue
  and receiver-repair queue.

## Authority Boundary

The new 414-byte budget is allowed only for targeted SegNet/PoseNet correction
acquisition planning. Component-measured correction rows and exact CPU/CUDA auth
eval remain required before spend, score, promotion, rank/kill, or dispatch
authority.

The next high-EV bridge is to turn this budget into a queue-owned component
repair acquisition policy: use paired SegNet/PoseNet behavior rows to spend
only the receiver-closed bytes whose predicted
`delta_segnet + delta_posenet + lambda * delta_bytes` improves.
