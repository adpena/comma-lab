# HiNeRV MLX local trainer smoke

Schema: `hi_nerv_mlx_trainer_smoke_pointer.v1`
Authority: `false_authority_macos_mlx_training_no_contest_score_claim`
Axis: `[macOS-MLX research-signal]`

Command:

```bash
.venv/bin/python experiments/train_substrate_hi_nerv_mlx_local.py --smoke --modelsize-row hi_nerv_local_tiny --num-pairs 2 --output-dir /Volumes/VertigoDataTier/pact/hinerv_mlx_local_training/codex_smoke_20260602T_now --smoke-export-archive
```

Result:

- Manifest: `/Volumes/VertigoDataTier/pact/hinerv_mlx_local_training/codex_smoke_20260602T_now/smoke_manifest.json`
- Manifest SHA-256: `6ebe0b2af224c309638db4a7c824a2be8cc2ed40cfd31a0144a177b3d8fda30e`
- Archive: `/Volumes/VertigoDataTier/pact/hinerv_mlx_local_training/codex_smoke_20260602T_now/smoke_archive_export/archive.zip`
- Archive SHA-256: `707c181486aef3acfd4c769d876d3e86ee38d7fe97e4f51879caf65958b22dfb`
- Archive bytes: `99696`
- Forward output shape: `[2, 2, 3, 384, 512]`

Blockers:

- `contest_cpu_cuda_exact_eval_not_executed`
- `hi_nerv_smoke_no_training_score`
- `official_hinerv_feature_grid_parity_not_proven`
- `full_video_mlx_scorer_replay_not_executed_for_smoke`
