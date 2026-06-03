# Codex Findings: NeRV Hard-Pair Tail Training Control

Date: 2026-06-03T05:08:39Z
Lane: lane_nerv_hardpair_tail_training_control_20260603
Axis: macOS-MLX research-signal, false-authority

## What Landed

- Added a robust PoseNet tail-burst detector to NeRV training telemetry feedback.
- Wired the detector into live-training control as `continue_running_queue_hardpair_prioritized_successor`, distinct from catastrophic pose instability and SegNet stagnation.
- Routed hard-pair indices from feedback rows into the HiNeRV long-training campaign command as `--prioritized-pair-indices`.
- Routed the same indices into `tools/run_compact_renderer_mlx_spine_runner.py`, where they are normalized, de-duplicated, recorded as false-authority training metadata, and passed to the real MLX score-aware trainer.
- Kept fail-closed planning: tail-burst feedback without a real hitlist/priority index source leaves `hinerv_pose_tail_burst_requires_prioritized_pair_indices`.

## Live HiNeRV Snapshot

Telemetry source:
`/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/hinerv_snerv_live_feedback_segw4_20260602T231600Z/hi_nerv_hinerv_np600_ld28_ed12_dc32_hfg_cnx_int4_mixed_ceil285000_adamw_feedback_seg_stagnation_lr2.7e-05_segw4_increase_segnet_distillation_weight_from_stagnat_preserve_pose_guard/hi_nerv_mlx_training/telemetry.jsonl`

Harvest output:
`/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/hinerv_snerv_live_feedback_segw4_20260602T231600Z/hi_nerv_hardtail_feedback_20260603T050839Z/nerv_training_telemetry_feedback.json`

Current verdict:

- last_epoch: 21921
- training_control_action: `continue_running`
- pose_instability_detected: false
- pose_instability_ever_detected: true
- pose_instability_recovered: true
- pose_tail_burst_detected: false
- pose_tail_burst_recent_p95: 4.830916023254394
- pose_tail_burst_recent_max: 5.7491455078125
- pose_tail_burst_threshold: 8.310143947601318
- seg_stagnation_detected: false
- score_claim: false
- ready_for_exact_eval_dispatch: false

This is a useful negative result: the guard is wired and does not falsely stop the current run after the recent tail has cooled. If a future full-video telemetry row shows intermittent hard-tail spikes, the planner will require a real hard-pair prioritized successor instead of silently treating a small-pair/easy-pair result as representative.

## Verification

- `uv run pytest src/tac/tests/test_nerv_candidate_feedback.py src/tac/tests/test_nerv_long_training_campaign_plan.py src/tac/tests/test_nerv_queue_training_feedback_refresh.py -q` -> 70 passed
- `uv run pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_execute_forwards_prioritized_pair_indices src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_parse_prioritized_pair_indices_arg -q` -> 2 passed
- `uv run ruff check src/tac/analysis/nerv_candidate_feedback.py src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/analysis/nerv_queue_training_feedback_refresh.py src/tac/tests/test_nerv_candidate_feedback.py src/tac/tests/test_nerv_long_training_campaign_plan.py src/tac/tests/test_nerv_queue_training_feedback_refresh.py`
- `uv run ruff check tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
- `uv run python -m py_compile src/tac/analysis/nerv_candidate_feedback.py src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/analysis/nerv_queue_training_feedback_refresh.py`

## Next Work

- Let the current HiNeRV run continue into final PR95/Muon stages unless the refreshed telemetry trips an actual stop condition.
- If tail-burst returns, run `tools/xray_hardpair_hitlist.py` on the full-video scorer axes, attach the resulting hard-pair indices to candidate feedback, and relaunch with `--prioritized-pair-indices` while preserving random full-video fill.
- Apply the same hard-pair priority route to SNeRV once the source-faithful MLX scorer-loop trainer owns real batch sampling.
