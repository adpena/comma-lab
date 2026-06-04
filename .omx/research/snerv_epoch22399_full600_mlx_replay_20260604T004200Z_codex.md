# SNeRV Epoch 22399 Full-Video MLX Replay

False-authority local MLX replay index for the receiver-proven SNeRV
checkpoint export harvested while the long run continued. This is not a score
claim, promotion claim, rank/kill signal, or exact-eval result.

## Artifact

- archive: `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_full600_native_temporal_lion_segw32_pose4_20260603T222429Z/checkpoint_exports/epoch022399_20260604T003630Z_ema_archive_export_codex/snerv_checkpoint_archive_bound_package/archive.zip`
- archive_sha256: `7d0cb10c2912cf3c8c2a659d25c5ffb479843e38a0514f154b00b84ab9b30285`
- archive_bytes: `444828`
- receiver_proof: `true`
- receiver_contract_satisfied: `true`
- export_report: `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_full600_native_temporal_lion_segw32_pose4_20260603T222429Z/checkpoint_exports/epoch022399_20260604T003630Z_ema_archive_export_codex/snerv_checkpoint_archive_export.json`
- replay_dir: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_epoch22399_full600_mlx_replay_20260604T003900Z`
- candidate_cache_report: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_epoch22399_full600_mlx_replay_20260604T003900Z/candidate_cache_report.json`
- mlx_response: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_epoch22399_full600_mlx_replay_20260604T003900Z/mlx_response_gpu_full600.json`
- cpu_spend_gate: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_epoch22399_full600_mlx_replay_20260604T003900Z/mlx_response_cpu_spend_gate.json`

## Full-Video MLX Result

- axis: `[macOS-MLX research-signal]`
- n_samples: `600`
- avg_segnet_dist: `0.7114705238739649`
- avg_posenet_dist: `163.19407329559326`
- score_rate_contribution: `0.29619270639942924`
- canonical_score: `111.84053130160859`
- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`

## Gate

- baseline_score: `0.192`
- cpu_gate_allowed: `false`
- blocker: `mlx_response_not_within_cpu_spend_band`

## Verdict

Epoch 22399 is receiver-proven but worse than epoch 18199 on the MLX advisory
surface (`111.84` vs `108.61`) while staying far over the hard byte ceilings
(`444828` bytes). Do not exact-gate or spend CPU/CUDA on this checkpoint. The
active SNeRV run should be used for optimization diagnosis and future harvests,
not promotion. The main actionable improvement from this tranche is the
selected-pair SNAR1 direct-cache decode path, which makes pair-window replay and
section-value profiling cheap enough to drive hard-pair/curriculum updates.
