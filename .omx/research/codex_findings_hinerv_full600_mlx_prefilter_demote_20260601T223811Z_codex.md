# Codex Findings: HiNeRV Full-600 MLX Prefilter Demotion

UTC: 2026-06-01T22:38:11Z

## Artifact

Full-video HiNeRV MLX train/export run:

- `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_full600_real_teacher_pr95curriculum_20260601T222455Z/compact_renderer_mlx_spine_runner_report.json`

Direct full-video MLX prefilter:

- `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_full600_direct_mlx_prefilter_20260601T2228Z/candidate_cache_report.json`
- `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_full600_direct_mlx_prefilter_20260601T2228Z/reference_cache/manifest.json`
- `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_full600_direct_mlx_prefilter_20260601T2228Z/mlx_scorer_response.json`
- `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_full600_direct_mlx_prefilter_20260601T2228Z/hprc_mlx_component_neutralization_profile.json`
- `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_full600_direct_mlx_prefilter_20260601T2228Z/mlx_prefilter_coverage.json`

Repo-local coverage summary:

- `.omx/research/hinerv_full600_mlx_prefilter_coverage_20260601T223811Z_codex.json`

## Result

The 600-pair HiNeRV run is byte-closed and receiver-proven, with archive size
`72,821` bytes on the local generated archive. Receiver-proof scratch briefly
created multi-GB raw output on the SSD and then cleaned it automatically; the
training result directory returned to about 1.3 MB.

The strict singleton MLX prefilter ran over all 600 pairs:

- `batch_pairs = 1`
- `n_samples = 600`
- `candidate_cache_pairs = 600`
- `reference_cache_pairs = 600`
- `canonical_score = 90.72321618617728`
- `segnet component = 50.482592299580574`
- `pose component = 40.1921353717714`
- `rate term = 0.048488514825309637`

This is a full-video MLX signal, but it is not a local CPU replay candidate.
The profile coverage gate records:

- `has_full_video_mlx_prefilter = true`
- `local_replay_mlx_prefilter_passed = false`
- blocker: `mlx_prefilter_score_not_below_local_replay_threshold`
- blocker: `mlx_score_above_hard_demote_threshold`

## Hardening

While running this, Codex found that full-video coverage alone was still too
permissive. A profile with `batch_pairs > 1` or an obviously noncompetitive MLX
score could still be mistaken for a CPU-replay trigger.

The gate now requires:

1. full-video profile coverage,
2. singleton scorer batch shape,
3. MLX score below the local replay hard-demote threshold.

That preserves sampled and bad full-video profiles as acquisition/demotion
signal without wasting local CPU replay or exact-auth attention.

## Verdict

This tiny 8-epoch HiNeRV operating point is demoted for score movement. It
confirms the carrier can be extremely small, but distortion remains completely
unsolved at this training depth/config. Next HiNeRV work should move to native
rate-aware, pose-protected, longer staged training or a different carrier
configuration before more replay spend.
