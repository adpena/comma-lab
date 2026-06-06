# HiNeRV hard-birth REAL-VIDEO receipt — first physical servo proof

UTC: 2026-06-06T19:05:00Z · Authority: `[macOS-MLX research-signal]`, bounded
research smoke (`--allow-unscored-research-smoke`), NON-PROMOTABLE, no score
claim. Run: smoke v4 at
`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_target_region_birth_real_smoke_20260606/hinerv_witness_readiness_short_smoke_v4/`
(canonical DAG argv + `--segnet-direct-live-distillation-weight 0.5
--segnet-direct-live-target-min-ratio-floor-weight 1.0
--pose-direct-live-distillation-weight 0.25`). Source: `upstream/videos/0.mkv`,
1 pair, real MLX SegNet teacher (EfficientNet-B2 cache argmax).

## The receipt (partner-spec)

```json
{
  "actuator": "fit_target_region_birth_from_segnet",
  "pair_id": 0,
  "class_id": 0,
  "region_label": 1,
  "target_pixels": 44132,
  "frame_fraction": 0.2245,
  "total_scored_pixels_batch_local": 196608,
  "normalization_authority": "batch_local_scored_pixels",
  "raw_region_debt_local_before": 22.4467,
  "old_region_hard_ratio": 0.0,
  "new_region_hard_ratio": 0.17973,
  "target_min_region_ratio": 0.02,
  "target_min_region_ratio_reached": true,
  "old_margin_mean": 6.0097, "new_margin_mean": 0.9278,
  "old_margin_p50": 6.0922,  "new_margin_p50": 0.4706,
  "old_margin_min": 1.1719,  "new_margin_min": -0.6147,
  "hard_won_count": 7932,
  "uint8_changed_count_region": 44132,
  "uint8_delta_abs_max": 255.0,
  "float_rgb_delta_linf": 0.99999928,
  "argmax_flipped_pixels_region": 39040,
  "accepted_step_count": 1,
  "rejected_histogram": {"subquantum": 0, "pose_guard": 0, "no_progress": 3,
                          "backtracking_attempts": 8,
                          "receiver_quantum_growth_attempts": 0},
  "loss_first": 38416.30, "loss_last": 3726.49,
  "final_learning_rate": 7.8125e-06,
  "trained_groups_grad_norms": {"feature_grids": 5.71e6,
    "fine_injector": 1.16e6, "head_rgb_1": 4.22e5, "latents_fine": 4.20e4},
  "out_of_scope_bit_frozen_verified": true,
  "pose_trust": {"available": false,
    "blocker": "hinerv_target_region_birth_pose_trust_telemetry_missing"},
  "fakequant_survived": null,
  "parseback_survived": null
}
```

(`hard_won_count` = 0.17973·44132. Full payload incl. per-group sha256 table:
`hi_nerv_mlx_training/training_artifact.json` →
`...output_head_target_init_gate.metadata.scorer_domain_bootstrap.target_region_birth_actuator`.)

## What this proves

The chain *worst-region selection (exact score units) → scoped SegNet VJP →
uint8 STE → live SegNet argmax flip → admission → receipt* is PHYSICAL on the
real video with the real scorer: a fully-unsolved 44k-pixel connected region
(`target_min_ratio` was literally 0.0) gained a hard argmax island (ratio
0→0.18, margin_min crossed negative) in one accepted receiver-visible step,
with out-of-scope tensors hash-verified bit-frozen. The DAG smoke evidence now
shows `hard_birth_argmax_progress_accepted_step_count=2`,
`max_candidate_segnet_worst_debt_reduction=3.32`.

## What it does NOT prove (named first-failing surfaces)

1. **Pose trust UNGUARDED**: the spine-runner callsite does not thread a
   `pose_teacher` into the actuator (payload blocker
   `..._pose_trust_telemetry_missing`; smoke `pose_exact_delta=None`). The
   accepted update changed frame-1 by up to 255 uint8 — pose harm is
   plausible and unmeasured. NEXT: thread the existing posenet teacher into
   the actuator call (1-line at callsite) so the trust cap engages.
2. **Birth-at-init regime**: 1 epoch, renderer near-init; the region was
   trivially unsolved. The hysteresis/stability test (wins persist M steps)
   and mid-training regimes remain.
3. **fakequant/parse-back/inflate survival not yet exercised** for this
   accepted update (live-MLX only). The DAG's remaining axis trace applies.
4. **DAG `min_ratio_increase` metric** reads the bootstrap loop only
   (`max_candidate_segnet_min_ratio_increase=0.0`) — the actuator's region
   ratio lift is not yet threaded into that evidence key.

## Gate-chain fixes that made the smoke runnable (landed)

dead `--scorer-domain-bootstrap` flag removed from
`_default_hinerv_smoke_command` + `--allow-unscored-research-smoke` added
(commit `f1dbb87e5`); custody-safe payloads — substrate metadata refuses
nested authority keys even false-valued (commit `e68ee3440`). NOTE: the
canonical argv still omits the teacher-enabling flags, so as committed it
cannot produce ACCEPTED hard-birth evidence — argv update with the
empirically-verified flag set is queued (sister-coordinated; codex owns the
runner surface this session).
