# HPRC full600 native-rate ramp + MLX gate, Codex findings

Date: 2026-06-01T12:15:45Z

## Scope

Turn the HPRC proof lane into queue-owned score movement without false authority:

- full600 MLX-local HPRC train/export;
- native rate-aware curriculum, not a single 8-step smoke;
- residual-importance-driven rate collapse using the real `residual_protection.npy`;
- receiver proof before MLX scoring;
- MLX full-video prefilter before local CPU replay;
- CPU replay only for MLX-filtered survivors.

All score rows below are non-authoritative unless explicitly tagged otherwise.

## Landed automation fixes

1. `tools/build_hprc_compact_receiver_training_queue.py`
   - Added build-time refusal for missing optional artifacts:
     - native residual protection `.npy`;
     - native P19/P18 artifacts;
     - HPRC rate-collapse residual-importance `.npy`;
     - HPRC rate-collapse P19/P18 artifacts;
     - Z8 follow-up inputs;
     - MLX prefilter reference cache directory.
   - This extinguishes the failed-worker class where a queue could be emitted with phantom `posenet_null_pairs.json` / `segnet_region_waterfill.json` paths and only fail inside `transcode_hprc_rate_collapse`.

2. `tools/profile_hprc_mlx_component_neutralization.py`
   - Added `--retain-large-tensor-cache`; default now certifies and deletes owned large tensor caches after MLX scoring.
   - Cleanup manifest: `hprc_mlx_large_tensor_cache_cleanup_manifest.v1`.
   - The MLX profile still preserves small durable outputs: response JSON, component arrays, window rows, cache reports, and cleanup manifest.

## Queue run

Queue:

`/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_full600_native_rate_ramp_v1_residual_importance_20260601T120647Z/hprc_queue.json`

Status:

- `succeeded`: 6
- `skipped`: 4
- `failed`: 0

Executed steps:

- `run_local_training`: succeeded, 13.35s, `[macOS-MLX research-signal]`, numpy-portable archive export.
- `transcode_hprc_rate_collapse`: succeeded, 19.98s.
- `write_hprc_campaign_followup_report`: succeeded.
- `gate_archive_rate_before_local_replay`: passed.
- `prove_hprc_rate_collapsed_receiver`: succeeded, 132.56s.
- `run_hprc_full_video_mlx_prefilter`: succeeded, 122.88s.
- `gate_hprc_mlx_prefilter_before_local_replay`: skipped downstream CPU replay by design.

Skipped steps:

- `run_local_cpu_replay`
- `gate_exact_cpu_after_local_replay`
- `write_hprc_campaign_post_replay_report`

Reason: MLX prefilter produced a hard demotion, so CPU replay would be wasted under the current gate.

## Empirical result

Training export:

- Source archive before rate collapse: 1,094,036 bytes.
- Training backend: MLX.
- Runtime/export backend: numpy-portable.
- Curriculum preset: `hprc_native_rate_ramp_v1`.
- Remaining blockers on training artifact: Z8 residual sidecar missing, full-video P18/P19 allocator not bound to candidate, local CPU replay not executed, contest CPU/CUDA exact eval not executed.

Rate collapse:

- Best variant: `residual_tokens_dz0_qd10`.
- Best archive: 109,407 bytes.
- Archive SHA-256: `acac343b2e2cdd9e1d8748ec5eec0a8e0f4f82f64cb6d3d3140e62bee701fa5a`.
- Rate term: 0.0728496304842374.
- Bytes saved vs training export: 984,629.
- Receiver proof: passed.
- Receiver runtime SHA-256: `7e06bcfa4a8e17a7206cc949eebd4432526364ab47513a1b2b719e01b184166c`.

MLX gate:

- Axis: `[macOS-MLX research-signal]`.
- MLX advisory score: 25.54149822318198.
- Components: SegNet 0.06628670583789548, PoseNet 35.49447713761444, rate contribution 0.0728496304842374.
- Gate blockers: `mlx_score_not_below_target`, `mlx_score_above_hard_demote_threshold`.
- Local CPU replay recommendation: false.

This is a durable demotion of this 64-epoch HPRC/native-rate rung, not a demotion of the paradigm. The candidate has competitive rate after collapse but catastrophic PoseNet distortion. The next useful HPRC work must be native distortion-aware training, not more posthoc rate collapse.

## Cleanup result

The just-finished MLX profile initially left 2.6 GB of tensor caches. After landing the cleanup helper, the run was certified and cleaned:

- Cleanup manifest: `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_full600_native_rate_ramp_v1_residual_importance_20260601T120647Z/hprc_mlx_prefilter/hprc_mlx_large_tensor_cache_cleanup_manifest.json`
- Deleted bytes: 2,831,172,841.
- Final run directory size: 61 MB.
- Final MLX profile directory size: 14 MB.

## Curriculum implication

The operator's PR95 comparison is correct: this 64-epoch HPRC rung is a fast sanity rung, not a PR95-grade curriculum. PR95's 30k-epoch, tightly bound training recipe is the standard for a real control arm. HPRC needs the same seriousness:

- multi-stage full-video curriculum;
- scorer-oriented PoseNet/SegNet terms, not only RGB/grid reconstruction;
- high-resolution pose-preserving sidecar or pose-aware latent channel;
- low-resolution/interior fill with crisp class/boundary spending;
- native rate proxy throughout training;
- byte-closed export and replay gates at every promotion rung.

The current blocker is not archive rate anymore for this rung; it is PoseNet distortion under the compact receiver.

## Verification

Commands:

```bash
uv run ruff check tools/build_hprc_compact_receiver_training_queue.py tools/profile_hprc_mlx_component_neutralization.py tools/run_hprc_compact_receiver_training.py src/tac/substrates/hprc/training_adapter.py tools/gate_hprc_mlx_prefilter_for_local_replay.py src/comma_lab/local_submission_replay.py tests/test_build_hprc_compact_receiver_training_queue.py src/tac/tests/test_profile_hprc_mlx_component_neutralization.py src/tac/substrates/hprc/tests/test_training_adapter.py src/tac/tests/test_local_submission_replay.py
PYTHONPATH=. .venv/bin/pytest tests/test_build_hprc_compact_receiver_training_queue.py src/tac/tests/test_profile_hprc_mlx_component_neutralization.py src/tac/substrates/hprc/tests/test_training_adapter.py src/tac/tests/test_local_submission_replay.py -q
```

Results:

- Ruff: passed.
- Pytest: 35 passed.

## Next tranche

Build the PR95-grade HPRC curriculum lane rather than replaying more oversized or pose-collapsed candidates:

1. Add native PoseNet-aware loss and pose sidecar curriculum to HPRC.
2. Train 32/128/600-pair ladders with checkpoint resume and telemetry.
3. Preserve the current hard MLX >0.5 demotion gate.
4. Only CPU replay candidates that clear the MLX gate.
5. Keep PR95/HNeRV as the control arm for faithful long curriculum/export/archive proof.
