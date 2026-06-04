# SNeRV Epoch 18199 Full-Video MLX Replay

False-authority local MLX replay index for the receiver-proven SNeRV checkpoint
export. This records artifact identity, the direct-cache path, and the exact
spend blocker; it is not a score claim, promotion claim, or exact-eval result.

## Artifact

- archive: `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_full600_native_temporal_lion_segw32_pose4_20260603T222429Z/checkpoint_exports/epoch018199_20260604T001141Z_ema_archive_export/snerv_checkpoint_archive_bound_package/archive.zip`
- archive_sha256: `d9289e0da2b12e2bd0c7ca41db2257c0f3b0ab81389396be4540793a2e64e962`
- archive_bytes: `444036`
- receiver_proof: `true`
- replay_dir: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_epoch18199_full600_mlx_replay_20260604T002246Z`
- candidate_cache_report: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_epoch18199_full600_mlx_replay_20260604T002246Z/candidate_cache_report.json`
- mlx_response: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_epoch18199_full600_mlx_replay_20260604T002246Z/mlx_response_gpu_full600.json`
- cpu_spend_gate: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_epoch18199_full600_mlx_replay_20260604T002246Z/mlx_response_cpu_spend_gate.json`

## Full-Video MLX Result

- axis: `[macOS-MLX research-signal]`
- n_samples: `600`
- avg_segnet_dist: `0.6795748979846636`
- avg_posenet_dist: `162.84334248860677`
- score_rate_contribution: `0.2956653461085565`
- canonical_score: `108.60700780929663`
- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`

## Verdict

The current SNeRV long-run checkpoint is blocked on both rate and distortion:
the archive is `444036` bytes and the full-video MLX response is far outside
the CPU-spend band. Exact/local CPU spend is blocked by
`mlx_response_not_within_cpu_spend_band`. The useful landing from this tranche
is the reusable SNAR1 direct-cache path; the next SNeRV score work is
score-faithful representation/training and LF payload replacement/compression,
not exact-gating this checkpoint.
