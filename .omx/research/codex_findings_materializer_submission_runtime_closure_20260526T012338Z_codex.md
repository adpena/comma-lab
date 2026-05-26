# Codex Findings: Materializer Submission Runtime Closure

Generated: 2026-05-26T01:23:38Z

## Verdict

Receiver-positive materializer rows were still too manual at the exact-readiness boundary: a candidate could carry a valid receiver proof and real saved bytes, while the exact-readiness bridge still saw only a bare `candidate.zip` directory and failed on missing `inflate.sh`, `report.txt`, archive manifest, and runtime tree custody.

This tranche adds a queue-owned closure step. `tools/build_materializer_submission_closure.py` now turns a selected harvested materializer source row into a contest-shaped submission directory containing `archive.zip`, copied minimal inflate runtime, `report.txt`, `archive_manifest.json`, and `runtime_consumption_proof.json`, then emits a closed source queue for the existing exact-readiness bridge.

## What Landed

- Added reusable closure implementation in `src/tac/optimizer/materializer_submission_closure.py`.
- Added operator/queue CLI `tools/build_materializer_submission_closure.py`.
- Wired `submission_runtime_manifest_closure` receiver-repair rows to run `build_submission_runtime_closure -> run_exact_readiness_bridge`.
- Reserved one receiver-repair queue slot for actionable submission/runtime closure rows so immediately executable byte wins do not get crowded out by broader advisory receiver repairs.
- Hardened runtime copying to exclude stale eval/run directories and stale auth/result JSON while preserving the minimal inflate runtime.
- Tightened receiver-proof authority so `receiver_contract_satisfied=true` alone no longer proves runtime consumption; family proofs must carry `runtime_consumption_proof_passed`, `passed`, or a passing runtime probe.
- Added regression coverage for closure building and exact-readiness bridge handoff.

## Empirical Anchor

Executed the generated receiver-repair queue path for `packet_member_zip_header_elide_544c5f580ec2`:

- Closure packet: `experiments/results/frontier_operation_portfolio/frontier_receiver_repair/frontier_operation_portfolio_20260526t012046z_receiver_repair/receiver_repair_materializer_packet_member_zip_header_elide_v1_submission_runtime_manifest_closure_archive_manifest_missing/submission_closure/submission`
- Closed source queue: `experiments/results/frontier_operation_portfolio/frontier_receiver_repair/frontier_operation_portfolio_20260526t012046z_receiver_repair/receiver_repair_materializer_packet_member_zip_header_elide_v1_submission_runtime_manifest_closure_archive_manifest_missing/submission_closure/closed_source_queue.json`
- Exact-readiness bridge: `experiments/results/frontier_operation_portfolio/frontier_receiver_repair/frontier_operation_portfolio_20260526t012046z_receiver_repair/receiver_repair_materializer_packet_member_zip_header_elide_v1_submission_runtime_manifest_closure_archive_manifest_missing/exact_readiness_bridge/exact_readiness_bridge_report.json`

Result: static receiver/runtime custody blockers cleared. The remaining blocker is only the active rate floor override:

`above_active_floor_archive_bytes_without_operator_override:345646>185578, active_score_frontier=0.206316386616; above rate-only byte floor`

No score, promotion, rank/kill, or dispatch authority is claimed.

## Queue Artifacts

Refreshed portfolio artifacts at `.omx/research/frontier_operation_portfolio_20260526T012046Z/`.

The receiver-repair queue now includes closure steps for both chain-level and standalone materializer exact-readiness repair rows. The standalone header-elide row is the concrete executable path for the current 156-byte saved budget signal.

## Six-Hook Wire-In

1. Sensitivity-map contribution: N/A for static submission closure; it preserves existing candidate saved-byte and receiver proof fields for downstream scorer-sensitive repair selection.
2. Pareto constraint: active through the exact-readiness active-rate-floor blocker and receiver proof gates.
3. Bit-allocator hook: active through the targeted correction budget context; freed bytes remain gated until receiver/exact-readiness proof.
4. Cathedral/autopilot dispatch hook: active through `receiver_repair_queue.json` command steps.
5. Continual-learning posterior: memo plus refreshed queue artifacts preserve the result; no score posterior update because no exact auth score was produced.
6. Probe-disambiguator: active through the bridge distinction between closure blockers and residual active-floor dispatch blockers.

## Next

The next queue-owned move is to apply the same closure machinery to chain rows once their component source queues have one candidate each, then use the remaining full-frame parity/runtime-proof repair rows to decide which saved-byte budget can fund SegNet/PoseNet targeted corrections.
