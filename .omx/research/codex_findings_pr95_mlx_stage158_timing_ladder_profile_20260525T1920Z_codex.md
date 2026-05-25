# Codex Findings - PR95 MLX Stage 1/5/8 Timing Ladder

UTC: 2026-05-25T19:20Z
Agent: Codex
Lane: `lane_pr95_hnerv_mlx_reproduction`
Evidence grade: `[macOS-MLX research-signal]`

## What Landed

- Added the queue-owned `stage158_source_video_timing_ladder` control profile to `tools/build_pr95_mlx_optimizer_matrix_queue.py`.
- The profile emits stages 1, 5, and 8 across step counts 1, 4, and 16 with source-video RGB+YUV6 timing loss, PR95 public archive export, runtime-consumption proof, and PyTorch export parity.
- Generated and executed queue `codex_pr95_stage158_timing_ladder_20260525T191820Z`.
- All 9 local-MLX cells succeeded with no queue blockers, no definition drift, and no orphaned steps.

## Durable Artifacts

- Queue: `.omx/research/codex_pr95_stage158_timing_ladder_queue_20260525T191820Z/pr95_mlx_stage158_timing_ladder_queue.json`
- Manifest: `.omx/research/codex_pr95_stage158_timing_ladder_queue_20260525T191820Z/pr95_mlx_stage158_timing_ladder_manifest.json`
- Observation: `.omx/research/codex_pr95_stage158_timing_ladder_queue_20260525T191820Z/pr95_mlx_stage158_timing_ladder_observation.json`
- Performance: `.omx/research/codex_pr95_stage158_timing_ladder_queue_20260525T191820Z/pr95_mlx_stage158_timing_ladder_performance.json`
- Compact result table: `.omx/research/codex_pr95_stage158_timing_ladder_queue_20260525T191820Z/pr95_mlx_stage158_timing_ladder_results.json`

## Timing Signal

| Stage | Steps | Seconds/Step | Examples/Sec | Last Loss |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.037153 | 26.916 | 0.137399 |
| 1 | 4 | 0.027120 | 36.873 | 0.114611 |
| 1 | 16 | 0.024748 | 40.408 | 0.009848 |
| 5 | 1 | 0.037273 | 26.829 | 0.137399 |
| 5 | 4 | 0.027205 | 36.757 | 0.137143 |
| 5 | 16 | 0.024714 | 40.463 | 0.135934 |
| 8 | 1 | 0.038095 | 26.250 | 0.137399 |
| 8 | 4 | 0.029025 | 34.453 | 0.137348 |
| 8 | 16 | 0.026364 | 37.931 | 0.137140 |

All 9 cells proved `runtime_consumption_proven=true` and `pytorch_export_forward_parity_established=true`.

## Bug Class Extincted

The first ladder execution exposed a stale false blocker: manifests could prove PyTorch export parity while still retaining `pr95_export_forward_parity_not_established` in source-faithfulness blocker lists and the parity status `blocker` field.

Fix:

- Added reconciliation so successful export parity removes only stale export-parity blockers from manifest, runtime profile, and optimizer recipe source-faithfulness lists.
- Removed the stale parity-status `blocker` once parity is established and replaced it with a non-authority note.
- Kept exact-readiness and dispatch surfaces fail-closed: local parity remains `[macOS-MLX research-signal]`, not score authority.
- Added regression coverage in `src/tac/tests/test_run_pr95_mlx_timing_smoke.py`.

## Remaining Blockers

This is not a score claim and is not promotion authority. The remaining PR95 MLX reproduction blockers are still real:

- SegNet/PoseNet scorer loss is not yet wired into MLX training.
- Stage hparams/cosine schedules are not fully source-matched.
- QAT C1A and resume semantics are not ported.
- Full-frame inflate parity against the source runtime is not run.
- Exact CPU/CUDA auth eval remains required before any score or promotion claim.

## Next Action

Use this ladder as the timing basis for a wider PR95 reproduction queue: more source-video pairs, stage 1/5/8 longer step counts, and then a full 8-stage source-video queue once scorer-loss parity and QAT/resume semantics land.
