# Byte-Range Runtime Tree Custody Queue Finding

UTC: 2026-05-26T06:44:26Z

## Finding

The operation-chain queue exposed a real receiver-custody bug in the byte-range entropy-recode path. `pr103_lc_ac_runtime_adapter_v1` was using `runtime_tree_sha256` for a digest over runtime file records, while submission closure resolves and verifies the actual runtime directory tree via `tac.repo_io.tree_sha256`. The mismatch blocked `build_byte_range_submission_closure` even though the generated adapter directory was explicit and receiver-consumable.

## Fix

- Normalized PR103 adapter semantics: `runtime_tree_sha256` now means the actual runtime directory tree digest.
- Preserved the legacy digest as `runtime_file_records_sha256`.
- Propagated both fields through byte-range receiver proof and chain manifest surfaces.
- Hardened materializer chain harvest to prefer the live `candidate_runtime_dir` tree hash when a receiver adapter directory exists.
- Added regressions for PR103 adapter custody, byte-range proof runtime binding, and chain harvest choosing live runtime tree custody over stale step metadata.

## Live Queue Evidence

Queue:
`.omx/research/frontier_rate_attack_feedback_refresh_20260526T062541Z_operation_chain_queue/operation_chain_compiler_queue.json`

State after repair: all 8 steps succeeded.

Durable queue-state mirror:
`.omx/research/frontier_rate_attack_feedback_refresh_20260526T062541Z_operation_chain_queue/operation_chain_compiler_queue_execution_status.json`

Candidate:
`byte_range_entropy_recode_8460014d7085`

Artifacts:

- Chain manifest: `experiments/results/frontier_operation_chain_compiler/frontier_rate_attack_operation_chain_queue_20260526t062541z_chain_compiler/chain_registered_multisurface_materializer_program/byte_range_entropy_recode_chain/byte_range_entropy_recode_chain_manifest.json`
- Harvested source queue: `experiments/results/frontier_operation_chain_compiler/frontier_rate_attack_operation_chain_queue_20260526t062541z_chain_compiler/chain_registered_multisurface_materializer_program/byte_range_entropy_recode_chain/exact_eval_handoff/source_queue.json`
- Submission closure report: `experiments/results/frontier_operation_chain_compiler/frontier_rate_attack_operation_chain_queue_20260526t062541z_chain_compiler/chain_registered_multisurface_materializer_program/byte_range_entropy_recode_chain/exact_eval_handoff/submission_closure/submission_closure_report.json`
- Exact-readiness bridge report: `experiments/results/frontier_operation_chain_compiler/frontier_rate_attack_operation_chain_queue_20260526t062541z_chain_compiler/chain_registered_multisurface_materializer_program/byte_range_entropy_recode_chain/exact_eval_handoff/exact_readiness_bridge_report.json`

Materializer result:

- Source archive bytes: `178223`
- Candidate archive bytes: `178207`
- Realized saved bytes: `16`
- Candidate archive SHA-256: `8460014d70855ce9226285f80513d6d743ed23723870a6a38b009cfca40f423e`
- Candidate runtime tree SHA-256: `71034b205fa9eb0c52e281bf66573e6d618a4ee51c22b547cd05e682a756f2b8`
- Runtime file-record digest: `93d0dbecdbff59daeb64b11760b9cd5454ccd8e22f9d9bec6c71217c6ec6f3a6`
- Submission runtime tree SHA-256: `3215cbed6685239a0fba2b12f7f858d12c12a8181f30763930d7e73c6ca767c8`
- Submission runtime content tree SHA-256: `5996c58ec9380eb15cb4a77b2347a2c8714969068881079784c21b772a316877`

Authority status:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact-readiness bridge wrote `ready_candidate_count=0`
- closed queue now carries separate adapter and submission runtime tree fields:
  `adapter_runtime_tree_sha256`, `candidate_runtime_tree_sha256`,
  `submission_runtime_tree_sha256`, and `submission_runtime_content_tree_sha256`
- exact-readiness report mirrors both sides:
  `runtime_consumption_proof_runtime_tree_sha256=71034b...`,
  `candidate_row_adapter_runtime_tree_sha256=71034b...`, and
  `submission_runtime_tree_sha256=3215...`

Remaining blockers are expected and fail-closed:

- `candidate_inflate_output_parity_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `full_frame_render_output_parity_missing`
- `shell_inflate_output_parity_missing`

## Verification

- `.venv/bin/python -m ruff check src/tac/pr103_lc_ac_runtime_adapter.py src/tac/optimization/byte_range_entropy_recode_materializer.py src/tac/optimization/byte_range_entropy_recode_chain.py src/tac/optimizer/materializer_chain_harvest.py src/tac/tests/test_pr103_lc_ac_runtime_adapter.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_range_entropy_recode_materializer.py src/tac/tests/test_pr103_lc_ac_runtime_adapter.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_materializer_submission_closure.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_frontier_rate_attack_feedback.py::test_operation_chain_queue_wires_byte_range_harvest_closure_and_readiness -q` -> 133 passed
- `.venv/bin/python tools/lane_maturity.py validate` -> 1379 lanes validated cleanly
- `tools/review_tracker.py policy-check` clean for touched Python files

## Next

The queue can now consume byte-range receiver artifacts end to end, but this is still a single materializer chain. The next high-EV move is to reuse the same custody bridge for multi-operation chains where rate savings are intentionally traded for targeted SegNet/PoseNet correction budget, then let the queue select composed operator programs rather than hand-running leaf materializers.
