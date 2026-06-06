# Codex session summary: SNeRV v18 feedback planner

UTC: 2026-06-05T18:24:07Z

## Landed changes

- Bound SNeRV candidate-feedback selection to telemetry progress after evidence
  scope and measured pair count. This keeps a newer 2-pair partial advisory from
  outranking a 16-pair bounded proof, while preferring the v28b 16-pair row over
  the v27 row because v28b has `last_epoch=2` / `row_count=3`.
- Added the regression in
  `src/tac/tests/test_nerv_long_training_campaign_plan.py`.
- Fixed `MlxScoreAwareAdapter._add_dual_ascent_metric_aliases` so tests that
  construct an adapter without a bundle fail closed instead of raising.
- Fixed the compact HiNeRV runner report path to emit the curriculum summary it
  already referenced.

## Current planner artifact

- Plan root:
  `/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_v18_feedback_progress_rank_20260605T182036Z`
- Plan JSON:
  `/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_v18_feedback_progress_rank_20260605T182036Z/nerv_long_training_campaign_plan.json`
- Queue:
  `/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_v18_feedback_progress_rank_20260605T182036Z/nerv_long_training_campaign_queue.json`
- SNeRV LF over-ceiling reroute queue:
  `/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_v18_feedback_progress_rank_20260605T182036Z/snerv_lf_over_ceiling_reroute_queue.json`
- SNeRV LF/HF replacement queue:
  `/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_v18_feedback_progress_rank_20260605T182036Z/snerv_lf_hf_replacement_queue.json`

## v18 SNeRV row status

- Selected feedback source:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_bounded_channelmean_v28b_fastcodec3_20260605Tcodex/snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1-2-4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelmean_int8_symmetric_ceil178000_native_rate_aware_training/compact_renderer_mlx_spine_runner_report.json`
- Candidate feedback scope: `bounded_score_aware_training_telemetry`
- Measured pairs: 16
- Candidate pairs: 600
- Feedback ready: false
- SNeRV scorer tether gate: passed
- Scorer input distribution guard proof: passed
- Queue status: disabled
- Queue runnable: false

Remaining SNeRV blockers include renderer nondegeneracy, value-domain pass,
byte-closed archive export, receiver proof, full-video MLX scorer response, and
full600/hard-pair replay. The 2-pair zero-MFU smoke is retained only as a
partial-pair advisory and does not make the candidate launchable.

## Validation

- `uv run pytest src/tac/tests/test_nerv_candidate_feedback.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_recover_interrupted_report_summarizes_snerv_nested_long_training_telemetry src/tac/tests/test_nerv_long_training_campaign_plan.py::test_snerv_bounded_candidate_feedback_clears_tether_and_guard_only src/tac/tests/test_nerv_long_training_campaign_plan.py::test_campaign_plan_auto_discovers_candidate_byte_feedback_guard_split src/tac/tests/test_operator_briefing.py::test_operator_briefing_nerv_plan_auto_discovers_feedback_roots src/tac/tests/test_operator_briefing.py::test_operator_briefing_nerv_feedback_discovery_is_root_fair_for_ssd_byte_rows src/tac/substrates/_shared/mlx_score_aware/tests/test_coder_qat.py src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py -q`
  passed: 236 tests.
- `uv run ruff check` on the owned slice passed.
- `uv run python -m py_compile` on the owned slice passed.
- `git diff --check` on the owned slice passed.

## Next DAG edge

The next executable edge is still value-domain/renderer closure, not a long
launch:

1. Repair the bounded 16/32-pair renderer smoke so the receiver reconstruction
   and target/export value-domain xray pass.
2. Materialize a byte-closed archive/export row only after that nondegenerate
   smoke passes.
3. Then run full-video MLX scorer prefilter and local CPU replay gates before
   any exact CPU/CUDA dispatch.
