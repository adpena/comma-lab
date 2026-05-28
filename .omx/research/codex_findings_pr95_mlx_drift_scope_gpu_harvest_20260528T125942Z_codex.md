# Codex Findings: PR95 MLX Drift Scope GPU Harvest

UTC: 2026-05-28T12:59:42Z

## Scope

Queue-owned harvest of the PR95/HNeRV MLX Conv2d drift-scope search for the public PR95 archive.

Artifacts:

- `.omx/research/pr95_mlx_drift_scope_queue_public_gpu_full_20260528T125320Z/queue.json`
- `.omx/research/pr95_mlx_drift_scope_queue_public_gpu_full_20260528T125320Z/worker_result.json`
- `.omx/research/pr95_mlx_drift_scope_queue_public_gpu_full_20260528T125320Z/scope_search_summary.json`
- `.omx/research/pr95_mlx_drift_scope_queue_public_gpu_full_20260528T125320Z/drift_scope_recommendation.json`

## Findings

- The queue executed one full GPU drift-scope step successfully: `success_count=1`, `failure_count=0`.
- The search evaluated 15 candidate Conv2d accumulation scopes.
- The smallest no-cliff recommendation is `preset_blocks02_kahan_fp32`, exposed to operators as `--mlx-gpu-drift-conv2d-override-preset blocks02_kahan_fp32`.
- The selected recommendation has `override_count=6`, `max_abs=0.000946044921875`, and `mean_abs=0.000037276826333254576`.
- The best raw drift-delta candidate remains `preset_blocks_refine_kahan_fp32`, with `override_count=14`, `max_abs=0.0008697509765625`, and `mean_abs=0.00003437255509197712`; it is deliberately not the default because it widens the scope without enough gain to justify becoming the normal PR95 control-arm preset.

## Authority

This is `[macOS-MLX research-signal]` only. It is not a score claim, not rank/kill authority, not promotion authority, and not exact-eval dispatch readiness.

Required blockers before score or promotion authority:

- full-frame inflate parity for the candidate/runtime pair;
- contest CPU or CUDA auth eval on a byte-closed archive;
- axis-labeled component deltas if used for a promotion decision.

## Integration

The reusable harvester is `comma_lab.scheduler.mlx_drift_scope_harvest`. It fails closed on truthy authority fields, records source-summary SHA-256, emits adoption flags, and keeps exact-readiness refused.

The default PR95 timing-smoke path already uses `blocks02_kahan_fp32`, so this harvest closes the loop from search result to reusable operator/control-arm default without turning local MLX advisory rows into score authority.
