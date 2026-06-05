# Codex Session Summary: SNeRV Renderer Gate + HiNeRV Harvest

## Landed

- Added SNeRV renderer nondegeneracy proof extraction to `nerv_candidate_feedback_row.v1`.
- Added required SNeRV nondegenerate renderer gate to the long-training planner and queue handoff.
- Preserved native SNeRV compact skip-high value-domain gate evidence from the runner attachment.
- Hardened HiNeRV launchability so rows with hard launch blockers cannot remain `local_mlx_launch_command_ready=true`.
- Added `tools/harvest_hinerv_smoke_comparison.py`, a false-authority SSD harvester for HiNeRV runner/acquisition/export reports.
- Wired HiNeRV harvest refresh wrappers into normal long-training planner auto-discovery so harvested smoke/export rows are planner-consumed, not side reports.
- Added HiNeRV direct-live SegNet collapse-escape controls: target class-histogram tether and class-balanced hinge in the shared MLX loss path plus runner CLI/metadata forwarding.
- Added HiNeRV scorer-domain contrast-floor loss and CLI controls for SegNet last-frame RGB and PoseNet two-frame YUV6 std-ratio floors.
- Added SNeRV scorer-loop `rate_paid` byte-growth admission so tiny archive growth can be admitted only when the byte-pressured objective still improves.

## Durable Artifacts

- HiNeRV comparison harvest:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_smoke_comparison_harvest_20260605T034322Z_codex/hinerv_smoke_comparison_harvest.json`
- Planner-consumable HiNeRV feedback refresh:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_smoke_comparison_harvest_20260605T034322Z_codex/hinerv_smoke_comparison_candidate_feedback_refresh.json`
- Final SNeRV/HiNeRV campaign plan:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_renderer_gate_hinerv_feedback_20260605T0352Z_codex/nerv_long_training_campaign_plan.json`
- Final experiment queue:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_renderer_gate_hinerv_feedback_20260605T0352Z_codex/nerv_long_training_campaign_queue.json`

## Evidence

- HiNeRV harvest scanned 73 rows and emitted 49 embedded candidate-feedback rows.
- Nested HiNeRV bitstream preparation is now detected from actual `hi_nerv_bitstream_preparation.json` files; latest harvest found 60 bitstream-preparation rows.
- Current `experiments/results/hinerv_*` smoke set had 0 decoder-waterfill plan rows, so waterfill remains a source-attachment gap for this comparison surface.
- Final campaign plan consumed 49 feedback rows, attached feedback to 11 rows, emitted 36 queue experiments, and kept all 36 disabled.
- SNeRV status is `snerv_scorer_tether_smoke_gate_blocked`.
- SNeRV queue blockers include `snerv_scorer_tether_smoke_report_missing`, `snerv_renderer_nondegenerate_smoke_missing`, and `snerv_renderer_nondegenerate_smoke_min16_pairs_missing`.
- HiNeRV contrast-floor telemetry is required when enabled via `scorer_input_contrast_floor_weight`, with separate SegNet RGB and PoseNet YUV6 ratio metrics.
- Harvested HiNeRV feedback refresh artifacts are discoverable by `tools/build_nerv_long_training_campaign_plan.py --auto-candidate-feedback-root`.

## Verification

- `uv run pytest src/tac/tests/test_nerv_candidate_feedback.py src/tac/tests/test_nerv_long_training_campaign_plan.py src/tac/tests/test_harvest_hinerv_smoke_comparison.py -q` passed: 145 tests.
- `uv run ruff check tools/harvest_hinerv_smoke_comparison.py src/tac/tests/test_harvest_hinerv_smoke_comparison.py src/tac/analysis/nerv_candidate_feedback.py src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/tests/test_nerv_candidate_feedback.py src/tac/tests/test_nerv_long_training_campaign_plan.py tools/run_compact_renderer_mlx_spine_runner.py` passed.
- `uv run python -m py_compile tools/harvest_hinerv_smoke_comparison.py src/tac/analysis/nerv_candidate_feedback.py src/tac/analysis/nerv_long_training_campaign_plan.py tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_harvest_hinerv_smoke_comparison.py src/tac/tests/test_nerv_candidate_feedback.py src/tac/tests/test_nerv_long_training_campaign_plan.py` passed.
- `uv run pytest src/tac/tests/test_harvest_hinerv_smoke_comparison.py src/tac/tests/test_nerv_candidate_curriculum.py src/tac/tests/test_nerv_long_training_campaign_plan.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_scorer_loop_decoder_qat.py src/tac/tests/test_snerv_score_aware_decoder_fit_work_order.py -q` passed: 160 tests.
- `uv run pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_mlx_score_aware.py src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py::test_train_export_long_training_binds_real_scorer_teachers -q` passed: 59 tests.
- `uv run pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_private_smoke_defaults_to_full_target_hydration_for_hard_pairs src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_runner_forwards_train_time_dual_ascent_to_shared_harness src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_snerv_execute_parser_accepts_planner_gated_families src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_compact_runner_parser_accepts_hi_nerv_pr95_curriculum_total_epochs src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_main_execute_snerv_forwards_direct_live_segnet_weight -q` passed: 5 tests.

## Next Commands

```bash
uv run python tools/harvest_hinerv_smoke_comparison.py \
  --artifact-root /Volumes/VertigoDataTier/pact/experiments/results \
  --limit 250
```

```bash
uv run python tools/build_nerv_long_training_campaign_plan.py \
  --hinerv-modelsize-budget /Volumes/VertigoDataTier/pact/experiments/results/nerv_modelsize_budget_official_skipportfolio_20260604T040458Z_codex/hinerv_modelsize_budget.json \
  --snerv-modelsize-budget /Volumes/VertigoDataTier/pact/experiments/results/nerv_modelsize_budget_official_skipportfolio_20260604T040458Z_codex/snerv_modelsize_budget.json \
  --candidate-feedback-source /Volumes/VertigoDataTier/pact/experiments/results/hinerv_smoke_comparison_harvest_20260605T034322Z_codex/hinerv_smoke_comparison_candidate_feedback_refresh.json \
  --output-json /Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_renderer_gate_hinerv_feedback_next/nerv_long_training_campaign_plan.json \
  --output-md /Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_renderer_gate_hinerv_feedback_next/nerv_long_training_campaign_plan.md \
  --output-queue /Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_renderer_gate_hinerv_feedback_next/nerv_long_training_campaign_queue.json \
  --output-snerv-lf-reroute-queue /Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_renderer_gate_hinerv_feedback_next/snerv_lf_over_ceiling_reroute_queue.json \
  --max-candidates-per-family 3
```

## Remaining Blockers

- SNeRV needs a passing scorer-tether smoke report attached to the same short long-training command path.
- SNeRV then needs a bounded 16/32-pair nondegenerate renderer proof with telemetry contract passed, receiver reconstruction verified, and value-domain gates passed.
- HiNeRV waterfill exists in older/advisory roots but is not present in the latest `experiments/results/hinerv_*` smoke set consumed by the new comparison harvester.
- HiNeRV still needs a fresh bounded local CPU/MLX smoke that actually enables contrast floor plus direct-live class-balanced hinge and then passes receiver-cache quality.
- No row is promotion eligible; no score/rank/kill authority was produced.
