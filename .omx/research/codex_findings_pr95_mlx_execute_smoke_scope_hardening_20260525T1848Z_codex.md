# Codex Findings: PR95 MLX Execute-Smoke Scope Hardening

UTC: 2026-05-25T18:48:00Z
Agent: codex
Scope: PR95/HNeRV MLX long-training recovery, queue-owned local execution, false-authority metadata.

## Findings

1. The recovered PR95 MLX long-training reports exposed `source_video_frame_count`
   but not whether that count was full-video decode or a bounded `max_frames`
   smoke cap. This could make bounded smokes look more representative than they
   are.
2. The MLX memory helper used deprecated `mx.metal.get_peak_memory()` directly.
   That made clean local smoke logs noisy and created avoidable future drift
   when MLX removes the API.
3. Queue-owned local smoke checkpoints can appear under `.omx/research/` when a
   report intentionally preserves compact signal there. The JSON reports and
   checkpoint export manifests are useful; checkpoint tensors and `.pt` payloads
   are rebuildable bulk and must not be accidentally staged.
4. `dispatch_packet_ready` was blocked in scheduler runtime-policy authority
   checks, but it was not part of the default `json_false_authority`
   false-or-missing alias set. A truthy value in a generic queued artifact could
   therefore slip past the standard artifact postcondition unless every caller
   remembered to override the alias list.
5. Local-training queue compilation needed the same explicit authority surface:
   queued local-training metadata should carry `dispatch_packet_ready=false`,
   refuse non-local resource kinds, and keep source-tree hashes as metadata
   rather than fake input artifact paths.
6. Lane maturity mutation could temporarily persist exact-readiness refusal
   artifacts behind `real_archive_empirical`, `contest_cuda`, or `contest_cpu`
   gates before validation rejected them. That was a false-authority window.

## What Landed

- `register_canonical_provenance()` and `build_long_training_plan_report()` now
  emit `source_video_frame_count_scope` plus `max_frames`, with `not_decoded`
  for plan-only rows that have not decoded the source video.
- `TrainingTelemetry` and `experiment_queue_observer` surface those fields for
  PR95 MLX long-training telemetry and plan reports.
- Regression tests assert full-video, capped-smoke, and not-decoded scope
  semantics.
- `_mlx_peak_memory_bytes()` prefers `mx.get_peak_memory()` and only falls back
  to the deprecated Metal namespace if necessary.
- `.gitignore` now ignores PR95 MLX execute-smoke checkpoint payloads under
  `.omx/research/` while keeping compact queue/report/telemetry JSON and export
  manifests trackable.
- `dispatch_packet_ready` is now covered by the default queue artifact
  `json_false_authority` guard, with a regression in the experiment queue tests.
- `build_local_training_execution_queue()` now fails closed on non-local
  resources, emits `dispatch_packet_ready=false`, scopes its false-authority
  postconditions to the local-training authority map, and keeps source-tree
  SHA-256 values in metadata while tracking only real paths as input artifacts.
- `tools/lane_maturity.py` now rejects
  `exact_readiness_refusal.ready=false` evidence at mark time and during
  validation for archive empirical, contest CPU, and contest CUDA gates.

## Queue-Owned Smoke Receipt

- Queue: `.omx/research/codex_pr95_mlx_long_training_execute_smoke_20260525T1855Z/pr95_mlx_long_training_execute_smoke_queue.json`
- Worker: `tools/experiment_queue.py --queue <queue> run-worker --execute --max-steps 1 --max-parallel 1`
- Report: `.omx/research/codex_pr95_mlx_long_training_execute_smoke_20260525T1855Z/pr95_mlx_long_training_execute_smoke_report.json`
- Observation: `.omx/research/codex_pr95_mlx_long_training_execute_smoke_20260525T1855Z/pr95_mlx_long_training_execute_smoke_observation.json`
- Performance: `.omx/research/codex_pr95_mlx_long_training_execute_smoke_20260525T1855Z/pr95_mlx_long_training_execute_smoke_performance.json`
- Telemetry: `.omx/research/codex_pr95_mlx_long_training_execute_smoke_20260525T1855Z/pr95_mlx_long_training_execute_smoke_telemetry.jsonl`
- Result: 1 queue step succeeded on `local_mlx` in 1.038581959 seconds, with 8
  telemetry rows, 8 checkpoint artifacts, `source_video_frame_count=2`,
  `source_video_frame_count_scope=max_frames_cap`, and `max_frames=2`.
- Loss moved from `0.1699081063` to `0.0073727462` over the bounded queue smoke.
- Checkpoint custody: compact `.pt.export_manifest.json` manifests are committed
  under the queue artifact directory; rebuildable `.pt`, `.mlx.safetensors`,
  `.npz`, and `.latents.npy` payloads remain ignored local bulk.

## Authority Boundary

This is still `[macOS-MLX research-signal]` only. The current loss is RGB-frame
MSE, not SegNet/PoseNet scorer-faithful PR95 training. All reports retain
`score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
`ready_for_exact_eval_dispatch=false`, and `reproduction_equivalence=false`.
