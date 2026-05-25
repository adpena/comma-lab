# Codex Findings: PR95 Runtime-Consumption Blocker Normalization

Generated: 2026-05-25T15:16Z
Agent: Codex
Topic: PR95/HNeRV MLX reproduction lane

## Result

Fixed a false-blocker bug in the PR95 MLX timing-smoke export surface. When a
public PR95 archive export now passes runtime-consumption proof, the archive
export summary records:

- `runtime_consumption_proof_present=true`
- `runtime_consumption_proven=true`

and removes the stale blocker:

- `pr95_archive_export_is_byte_closed_but_not_runtime_consumed`

The export remains correctly fail-closed on the real remaining blockers:

- full-frame inflate parity against the source runtime is not established
- exact CPU/CUDA auth eval is still required before any score claim

## Corrected Artifact

Artifact root:

- `experiments/results/pr95_mlx_full_source_video_runtime_profile_20260525T151528Z/`

Queue state:

- healthy
- 3 succeeded
- 0 failed
- 0 orphaned
- 0 definition-drift steps

Execution:

| Stage | Module | Loss | Manifest train seconds | Runtime proof |
|---:|---|---|---:|---|
| 1 | `stage1_v328_ce` | `rgb_yuv6_mse` | 0.0341 | proven |
| 5 | `stage5_c1a_l7` | `rgb_yuv6_mse` | 0.0349 | proven |
| 8 | `stage8_muon_finetune` | `rgb_yuv6_mse` | 0.0361 | proven |

Queue telemetry:

- elapsed seconds mean: 1.5345
- elapsed seconds sum: 4.6034
- local MLX failures: 0

All three `pr95_public_archive_export.json` files now have the stale
not-runtime-consumed blocker removed and retain the full-frame parity blocker.

## Verification

- `ruff check tools/run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py`: pass
- `pytest src/tac/tests/test_run_pr95_mlx_timing_smoke.py -q`: pass
- regenerated and executed the full PR95 MLX control queue above: pass
