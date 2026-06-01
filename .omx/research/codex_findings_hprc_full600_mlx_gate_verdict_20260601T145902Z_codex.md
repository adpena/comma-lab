# HPRC full600 MLX-gated rate-collapse verdict

## Status

`hprc_full600_native_pose_guard_p18p19_20260601T144218Z` completed as a queue-owned full-video HPRC campaign. The run is non-promotable and carries no score authority.

## Evidence

- Queue root: `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_full600_native_pose_guard_p18p19_20260601T144218Z`
- Queue supervisor: `supervisor/supervisor_result.json`
- Rate-collapse report: `hprc_rate_collapse/hprc_rate_collapse_report.json`
- MLX advisory response: `hprc_mlx_prefilter/mlx_responses/baseline.json`
- MLX replay gate: `hprc_mlx_prefilter_local_replay_gate.json`
- Updated planner follow-up: `hprc_queue_followup_report.json`

## Finding

The HPRC rate axis is not the immediate blocker after residual-token collapse. The best byte-closed archive in this run is:

- variant: `residual_tokens_dz0_qd10`
- archive bytes: `217365`
- archive SHA-256: `dcb50dc5d09d7bc0f8033068054431041f5aefaeaea4b6a464855e8e92a59088`
- rate term: `0.1447344313454008`

That is below the local sub-0.19 zero-distortion archive-byte ceiling recorded by the MLX profile (`285345` bytes). The blocker is distortion. MLX advisory full600 recomputation gave:

- axis: `[macOS-MLX research-signal]`
- advisory score: `23.465335421414697`
- average PoseNet distortion: `28.23499529233217`
- average SegNet distortion: `0.06517328900285065`
- blockers: `mlx_score_not_below_target`, `mlx_score_above_hard_demote_threshold`

The queue correctly did not run local CPU replay and did not dispatch exact CPU/CUDA auth.

## System Change

Codex patched the HPRC queue loop so MLX-gated losers still emit a planner-readable follow-up report. The new report records:

- `local_replay_gate.blockers = ["mlx_prefilter_rejected_candidate_before_cpu_replay"]`
- the MLX advisory score and blockers under `local_replay_gate.mlx_prefilter_gate`
- `hprc_mlx_prefilter_gate_not_passed` as a sidecar and promotion blocker

This closes the previous automation hole where a skipped gate protected CPU time but hid the reason from downstream planning.

## Verdict

Do not spend more CPU/CUDA auth on this low-res dense residual collapse basin until the candidate first clears the full600 MLX prefilter. Reactivate when native rate-aware scorer/PoseNet training, sparse procedural protected-geometry tokens, or a Z8/HPRC residual-sidecar allocator produces:

1. `archive_zip_bytes <= 285345`
2. full600 MLX advisory score `< 0.5`
3. receiver proof present
4. exact local CPU replay gate recommended

The next HPRC score-moving work should shift from posthoc residual coarsening to native scorer-aware training and sparse protected pose/geometry tokens.
