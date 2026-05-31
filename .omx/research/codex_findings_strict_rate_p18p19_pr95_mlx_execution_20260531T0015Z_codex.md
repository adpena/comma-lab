# Codex Findings: Strict Final-Rate Attack, P18/P19 Distortion Budget, PR95 MLX Control Arm

UTC: 2026-05-31T00:15Z

## Scope

This pass executed the strict queue-owned final-rate attack against the current CPU frontier, then advanced the scorer-region distortion-budget bridge and PR95/HNeRV MLX control arm without promoting any proxy signal to score authority.

## Strict Final-Rate Attack

Queue artifact:
`.omx/research/frontier_final_rate_attack_strict_exec_20260530T2300Z/experiment_queue.json`

Observer artifact:
`.omx/research/frontier_final_rate_attack_strict_exec_20260530T2300Z/observer_revalidation.json`

Signal harvest:
`.omx/research/frontier_final_rate_attack_strict_exec_20260530T2300Z/final_rate_attack_signal_harvest.json`

Result:

- Queue health: `healthy`.
- Queue status counts: `succeeded=25`.
- Failed command count: `0`.
- Materializer observations: `13`.
- Rate-positive materializer observations: `0`.
- Max saved bytes: `0`.
- Target coverage: `archive_zip_repack_v1=3`, `fp11_source_brotli_recode_v1=2`, `packet_member_recompress_v1=3`, `packet_member_zip_header_elide_v1=3`, `selector_stream_context_recode_v1=2`.
- Post-feedback child queues executed cleanly: `operation_chain_compiler_queue`, `autonomous_chain_optimization_queue`, `repair_campaign_score_queue`, `repair_posterior_acquisition_followup_queue`.

Canonical verdict: this strict pass found no exact-ready rate-only survivor against the current CPU frontier. The result is a negative posterior for these materializer families on this archive state, not a global method retirement.

## P18/P19 Distortion-Budget Bridge

Queue artifact:
`.omx/research/scorer_region_selector_chain_strict_exec_20260530T2315Z/queue.json`

Observer artifact:
`.omx/research/scorer_region_selector_chain_strict_exec_20260530T2315Z/observer_revalidation.json`

External result root:
`/Volumes/VertigoDataTier/experiments/results/scorer_region_selector_chain_strict_exec_20260530T2315Z`

Result:

- Queue status: `9/9` steps succeeded.
- P19 PoseNet-null selected pairs: `60`.
- P18 SegNet-region waterfill selected pairs: `60`.
- Receiver patch materialized for `12` frame-1 region edits.
- Selector saved bytes: `0`.
- Repack saved bytes after selector: `0`.
- Cumulative rate saved bytes vs source: `0`.
- Exact dispatch: refused.

Blocking proofs still required before any score or budget-spend claim:

- `shell_inflate_output_change_proof_missing`
- `full_frame_inflate_parity_proof_missing`
- `runtime_consumption_proof_required_before_exact_eval`
- `inflated_output_change_proof_required_before_budget_spend_claim`
- `local_mlx_or_cpu_component_spot_check_required`
- `exact_auth_eval_required_before_score_or_promotion_claim`

Canonical verdict: the P18/P19/P11 cascade is now queue-owned and receiver-patch-producing, but remains a fail-closed local candidate surface. The next useful action is output-change proof plus local MLX/CPU component spot check, not Modal dispatch.

## PR95/HNeRV MLX Control Arm

Report artifact:
`.omx/research/pr95_mlx_bounded_long_training_20260530T2315Z/report.json`

External telemetry:
`/Volumes/VertigoDataTier/experiments/results/pr95_mlx_bounded_long_training_20260530T2315Z/telemetry.jsonl`

External checkpoints:
`/Volumes/VertigoDataTier/experiments/results/pr95_mlx_bounded_long_training_20260530T2315Z/checkpoints`

Result:

- Mode: `executed_smoke`.
- Stages executed: `8`.
- Frame cap: `32`.
- Telemetry rows: `41`.
- Source video SHA-256: `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`.
- Stage-8 checkpoint loss: `0.0022818949073553085`.
- PyTorch export artifacts were emitted for checkpoints.
- Exact dispatch: refused.

Blocking proofs still required before PR95/HNeRV MLX can be treated as a reproduction or promotion candidate:

- `source_optimizer_scheduler_qat_parity_not_yet_attested`
- `requires_pytorch_export_forward_parity_on_result_checkpoint`
- `requires_full_frame_inflate_parity_before_runtime_consumption_claim`
- `requires_segnet_posenet_loss_or_exact_scorer_response_calibration_before_dispatch_authority`
- `requires_paired_contest_cpu_and_cuda_auth_eval_before_score_claim`

Canonical verdict: PR95/HNeRV MLX has moved beyond static parity fragments into bounded local training with telemetry, checkpoint custody, and exports, but remains `[macOS-MLX research-signal]` only.

## Frontier State

No exact CPU/CUDA auth eval was dispatched or harvested in this pass, so the canonical frontier is unchanged:

- `[contest-CPU Linux x86_64]`: `0.1919853363`
- `[contest-CUDA T4]`: `0.2053300290`

## Next Queue-Owned Actions

1. Run the P18/P19 receiver patch through shell inflate output-change proof.
2. Run local MLX/CPU component spot checks on the patched candidate and keep it non-authoritative unless calibrated.
3. Feed the strict-rate negative posterior into acquisition so the next final-rate pass emphasizes unsaturated positions: entropy-before-coder transforms, receiver-side proceduralization, tensor grammar changes, and region/frame budget spending.
4. Advance PR95 MLX from 32-frame bounded smoke to a measured longer run only after export forward parity is checked on the emitted checkpoint.
