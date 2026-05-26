# Codex Findings: Materializer Auxiliary Queue Execution Bridge

generated_at_utc: 2026-05-26T21:09:41Z
agent: codex
repo: /Users/adpena/Projects/pact
authority: research_only_false_authority

## Scope

This pass continued the frontier final-rate-attack automation lane by running the
FEC8 targeted component correction chain materializer execution queue through
`experiment_queue.v1` instead of treating the materializer handoff as manual
leaf work.

## Findings

1. The feedback-cycle auxiliary execution set omitted materializer execution
   queues. `tools/run_frontier_rate_attack_feedback_cycle.py
   --execute-auxiliary-queues` could run receiver repair, chain compiler, and
   correction queues, but skipped both `operation_materializer_execution_queue`
   and `targeted_component_correction_chain_materializer_execution_queue`.
   This was a real automation gap: the cycle emitted materializer execution
   intent but did not consume it.

2. Long materializer experiment IDs exceeded macOS path component limits in
   worker log directories. The queue identity and experiment identity were
   valid canonical IDs, but using raw IDs as filesystem path components failed
   before execution with `OSError: [Errno 63] File name too long`. The fix keeps
   canonical IDs unchanged in SQLite/events and deterministically shortens only
   log-path components.

3. Submission/runtime closure inferred too little from packet-shaped source
   artifacts. The harvested FEC8 source row preserved the source archive path
   but did not carry `source_runtime_dir`. The source packet had a sibling
   `submission_dir/inflate.sh` and a `packet_manifest.json` runtime path, so the
   closure builder now discovers source runtimes from explicit row fields,
   inflate paths, sibling packet layout, and packet manifest runtime fields.

## Live Queue Evidence

Queue:
`.omx/research/frontier_rate_attack_feedback_refresh_fec8_rate_packet_bridge_20260526_codex_v2/targeted_component_correction_chain_materializer_execution_queue.json`

State:
`.omx/state/experiment_queue_frontier_rate_attack_fec8_rate_packet_bridge_20260526_codex_v2_targeted_chain_materializer_execution.sqlite`

Final status after the fixes:

- `materialize_local_proof_chain`: succeeded
- `harvest_materializer_chains`: succeeded
- `build_materializer_submission_closure`: succeeded after source-runtime
  discovery fix and rewind
- `run_materializer_exact_readiness_bridge`: succeeded
- `build_exact_eval_dispatch_plan`: succeeded

Dispatch verdict remains correctly fail-closed. The materializer candidate was
byte-closed and receiver-closed, but the archive delta was zero bytes and the
candidate retained `candidate_not_rate_positive`, so no exact-eval dispatch
queue was authorized.

Key generated artifacts:

- candidate archive: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/frontier_targeted_component_correction_chain_materializers/targeted_component_chain_materializers/targeted_component_chain_targeted_component_materialization_receiver_closed_rate_packet_lane_pr101_frame_exploit_selector_fec8_static_se_4ab4c1053240f68d_001/packet_member_zip_header_elide_v1/candidate.zip`
- closed source queue: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/frontier_targeted_component_correction_chain_materializers/targeted_component_chain_materializers/targeted_component_chain_targeted_component_materialization_receiver_closed_rate_packet_lane_pr101_frame_exploit_selector_fec8_static_se_4ab4c1053240f68d_001/packet_member_zip_header_elide_v1/exact_eval_handoff/submission_closure/closed_source_queue.json`
- exact-readiness report: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/frontier_targeted_component_correction_chain_materializers/targeted_component_chain_materializers/targeted_component_chain_targeted_component_materialization_receiver_closed_rate_packet_lane_pr101_frame_exploit_selector_fec8_static_se_4ab4c1053240f68d_001/packet_member_zip_header_elide_v1/exact_eval_handoff/exact_readiness_bridge_report.json`
- dispatch plan: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/frontier_targeted_component_correction_chain_materializers/targeted_component_chain_materializers/targeted_component_chain_targeted_component_materialization_receiver_closed_rate_packet_lane_pr101_frame_exploit_selector_fec8_static_se_4ab4c1053240f68d_001/packet_member_zip_header_elide_v1/exact_eval_handoff/dispatch_plan.json`

## Tests

- `.venv/bin/ruff check src/tac/optimizer/materializer_submission_closure.py src/tac/tests/test_materializer_submission_closure.py src/comma_lab/scheduler/experiment_queue.py src/tac/tests/test_experiment_queue.py tools/run_frontier_rate_attack_feedback_cycle.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/python -m pytest src/tac/tests/test_materializer_submission_closure.py -k 'discovers_source_packet_submission_dir or clears_static_readiness_blockers'`
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py -k 'worker_log_paths_shorten_long_queue_components or executes_local_steps'`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -k 'feedback_cycle_auxiliary_execution_includes_materializer_queues'`

## Next Work

The next automation target is to make the refreshed feedback cycle generate and
execute materializer queues with source runtime context already present, then
promote positive-rate candidates into grouped chain search rather than
single-materializer local optima. Zero-delta candidates should remain useful as
receiver and custody proofs, but they should not consume exact-eval dispatch
budget unless paired with downstream distortion-spend operations.
