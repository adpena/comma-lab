# HiNeRV Epoch 2749 Byte-Cap V2 Full-Video MLX Replay

False-authority local MLX replay index for the receiver-proven HiNeRV checkpoint
export. This records the exact artifact identity and blocker verdict; it is not
a score claim, promotion claim, or exact-eval result.

## Artifact

- archive: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_full600_pr95_pact_muon_seg16_resume_20260603T231916Z/epoch002749_ema_archive_export_20260604T000054Z_bytecap_v2/archive.zip`
- archive_sha256: `5e0c42dbf6d401a767a6ea85256dac734c88bd9515f7913fe3f72cf770e634ca`
- archive_bytes: `122074`
- receiver_proof: `true`
- replay_dir: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch2749_bytecap_v2_full600_mlx_replay_20260604T000913Z`
- candidate_cache_report: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch2749_bytecap_v2_full600_mlx_replay_20260604T000913Z/candidate_cache_report.json`
- mlx_response: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch2749_bytecap_v2_full600_mlx_replay_20260604T000913Z/mlx_response_gpu_full600.json`
- cpu_spend_gate: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch2749_bytecap_v2_full600_mlx_replay_20260604T000913Z/mlx_response_cpu_spend_gate.json`

## Full-Video MLX Result

- axis: `[macOS-MLX research-signal]`
- n_samples: `600`
- avg_segnet_dist: `0.5505764264861742`
- avg_posenet_dist: `132.72973542531332`
- score_rate_contribution: `0.08128406584343595`
- canonical_score: `91.57101908167377`
- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`

## Verdict

The 122 KB archive is rate-plausible but distortion-blocked. Exact/local CPU
spend is blocked by `mlx_response_not_within_cpu_spend_band` against the current
frontier band. This checkpoint should not be exact-gated; use it as evidence
that the byte-cap/export path works and that the remaining HiNeRV blocker is
score-faithful decoder fitting, not archive rate for this artifact.
