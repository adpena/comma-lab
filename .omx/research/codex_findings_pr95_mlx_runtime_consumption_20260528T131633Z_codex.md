# Codex Findings: PR95 MLX Runtime-Consumption Proof

UTC: 2026-05-28T13:16:33Z

## Scope

Queue-owned PR95/HNeRV MLX archive export consumed by the public PR95 `inflate.sh` runtime under a bounded one-pair raw-output cap.

Artifacts:

- `.omx/research/pr95_mlx_runtime_consumption_queue_20260528T131513Z/queue.json`
- `.omx/research/pr95_mlx_runtime_consumption_queue_20260528T131513Z/worker_result.json`
- `.omx/research/pr95_mlx_runtime_consumption_queue_20260528T131513Z/runtime_consumption_proof.json`
- `.omx/research/pr95_mlx_runtime_consumption_queue_20260528T131513Z/pr95_public_archive_export.json`

## Findings

- The queue executed successfully: `success_count=1`, `failure_count=0`, with no failed postconditions.
- Public PR95 runtime consumption is proven for the generated one-pair archive: `runtime_consumption_proven=true`.
- Expected raw bytes and observed raw bytes both equal `6104016`.
- Raw output SHA-256 is `d7f6455974f85d17f9d6780a0ec00e570d7ccd04970d16a322fd16dc9cc4d543`.
- Candidate archive export is `230349` bytes, SHA-256 `bf5922b27c8f1c350df728b7131f9b99dd83d83819152638a6ce156a493fb046`.
- The runtime proof artifact SHA-256 recorded in the export manifest is `f1ea67ad05bd9d1a3581825554117170375b5e901b4ddb979204ba47cf17ea24`.
- Worker elapsed time was `1.7905631659959909` seconds.
- The generated raw work directory was removed after harvest; proof JSON retains byte counts and SHA custody, avoiding rebuildable raw-output disk growth.

## Authority

This proves receiver/runtime consumption for a local PR95 MLX archive export. It is still not a contest score, not promotion authority, not rank/kill authority, and not exact-eval dispatch authority.

Remaining blockers:

- full-frame parity against a source/public runtime baseline;
- full SegNet/PoseNet scorer loss wiring for training;
- PR95 full schedule/QAT/resume fidelity;
- exact CPU/CUDA auth eval before any score claim.

## Integration

This closes the first receiver proof gap for the PR95 MLX control arm: a generated MLX archive can be consumed by the public PR95 runtime through queue-owned execution. The next closure step is full-frame inflate parity and then scaling the same queue path to source-video-trained checkpoints.
