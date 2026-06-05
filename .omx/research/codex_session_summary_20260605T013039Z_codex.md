# Codex Session Summary - SNeRV Tether Gate, Terminal Harvest, HiNeRV Prefilter

UTC: 2026-06-05T01:30:39Z

## Landed Surfaces

- Added `tools/run_snerv_scorer_tether_smoke.py`, a tiny MLX PR95 SNeRV scorer-tether smoke that writes `snerv_scorer_tether_smoke.v1` with no score, promotion, rank/kill, or exact-eval authority.
- Added `src/tac/tests/test_snerv_scorer_tether_smoke.py`.
- Wired `--snerv-scorer-tether-smoke-report` through `tools/build_nerv_long_training_campaign_plan.py` and `src/tac/analysis/nerv_long_training_campaign_plan.py`; failed smoke reports become SNeRV queue launch blockers.
- Extended `src/tac/tests/test_nerv_long_training_campaign_plan.py` for passing/failed SNeRV tether smoke and CLI JSON consumption.
- Consumed worker landing in `tools/operator_briefing.py` and `src/tac/tests/test_operator_briefing.py`: default briefing now auto-discovers NeRV feedback roots, filters authority-leaky feedback, and emits a runnable campaign-plan command with `--auto-candidate-feedback-root`.

## SSD Artifacts

- SNeRV tether smoke report:
  `/Volumes/VertigoDataTier/pact/experiments/results/snerv_scorer_tether_smoke_20260605Tcodex/snerv_scorer_tether_smoke.json`
  SHA-256: `5bf757e83aae7a2fd9cf0d737999d2eb48a3b9363663f27df5ee46783c1d7201`
- Terminal SNeRV telemetry feedback row:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_scalarmean_hardpair_successor_fix2_20260604Tcurrent_codex/training_telemetry_feedback_20260605Tterminal_codex/nerv_candidate_training_telemetry_feedback_row.json`
  SHA-256: `6e4d22ecfa640fef8f04d74cb7ad5a091b1c63e398533a4bccd21bdfd37caa2d`
- Final queue handoff:
  `/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_snerv_terminal_tether_snar2_guard_20260605Tcodex/nerv_long_training_campaign_queue.json`
  SHA-256: `960535072f8e012ab1315b98d6ba08c48dd507b84ffbdc376feabee7dc8a70df`
- Final plan:
  `/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_snerv_terminal_tether_snar2_guard_20260605Tcodex/nerv_long_training_campaign_plan.json`
  SHA-256: `a4cb27017ffa3fca62311ece6ba13f490047314c20628d61aea82504606e0a58`
- HiNeRV high-byte arithmetic EMA prefilter export:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_bytecap_successor_final_epoch029999_ema_hi_ac_prefilter_20260605Tcodex/hinerv_checkpoint_archive_export.json`
  SHA-256: `a4b5e9acd8df0ebee6f6ab8fecebe24604ba0fc9e5e4c8ee98075193000faba3`

## Evidence

- Current SNeRV smoke report passed: both PR95 scorer tether aliases are present and both dual-ascent lambdas activate in the tiny smoke.
- The old scalar-mean run completed telemetry through epoch 29649 but produced no final runner/export report. Terminal feedback now records it as stopped and blocked by:
  - `snerv_scorer_domain_tether_missing_telemetry`
  - `snerv_posenet_yuv6_pair_distill_metric_missing_telemetry`
  - `snerv_segnet_last_frame_distill_metric_missing_telemetry`
  - `snerv_scorer_domain_tether_lambda_inactive_telemetry`
- Final regenerated campaign plan has 47 rows, 47 blocked rows, 0 launchable local rows, SNeRV smoke attached/passing, terminal feedback consumed, and SNAR2 minimization source count 2.
- HiNeRV EMA high-byte arithmetic export is receiver-proofed but not competitive: `181295` archive bytes, hard ceiling requested `178000`, full-video MLX CPU prefilter `n_samples=600`, advisory `avg_segnet_dist=0.49063136637210847`, `avg_posenet_dist=84.1670199139913`, recomputed advisory score `78.19540639252095`.

## Verification

- `uv run pytest src/tac/tests/test_snerv_scorer_tether_smoke.py src/tac/tests/test_nerv_long_training_campaign_plan.py::test_long_training_campaign_plan_consumes_passing_snerv_tether_smoke src/tac/tests/test_nerv_long_training_campaign_plan.py::test_long_training_campaign_plan_blocks_failed_snerv_tether_smoke src/tac/tests/test_nerv_long_training_campaign_plan.py::test_build_long_training_campaign_plan_cli_writes_outputs src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py::test_pr95_curriculum_snerv_dual_ascent_observes_stage_surrogate_aliases_NO_FAKE src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py::test_score_aware_telemetry_contract_rejects_stale_pr95_alias_failure src/tac/tests/test_nerv_candidate_feedback.py::test_training_telemetry_feedback_blocks_snerv_degenerate_renderer -q`
  passed: 8 tests.
- `uv run pytest src/tac/tests/test_operator_briefing.py::test_operator_briefing_nerv_plan_auto_discovers_feedback_roots src/tac/tests/test_operator_briefing.py::test_operator_briefing_nerv_plan_filters_feedback_authority_leaks -q`
  passed: 2 tests.
- `uv run ruff check tools/operator_briefing.py src/tac/tests/test_operator_briefing.py tools/run_snerv_scorer_tether_smoke.py tools/build_nerv_long_training_campaign_plan.py src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/tests/test_snerv_scorer_tether_smoke.py src/tac/tests/test_nerv_long_training_campaign_plan.py`
  passed.
- `uv run python -m py_compile tools/operator_briefing.py src/tac/tests/test_operator_briefing.py tools/run_snerv_scorer_tether_smoke.py tools/build_nerv_long_training_campaign_plan.py src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/tests/test_snerv_scorer_tether_smoke.py src/tac/tests/test_nerv_long_training_campaign_plan.py`
  passed.
- `git diff --check` for the touched slice passed.

## Next Gates

- Relaunch SNeRV only through the feedback-aware planner after the smoke report is attached; no scalar-mean successor should bypass this gate.
- Bind the current-code scorer tether smoke to the actual long-training command path, then run a short SNeRV training smoke that proves telemetry carries non-missing `snerv_posenet_yuv6_pair_distill` and `snerv_segnet_last_frame_distill` metrics plus active lambdas.
- Keep SNAR2/SNSA2 byte layout gains, but treat them as packaging evidence until renderer/scorer closure exists.
- For HiNeRV, stop treating generic latent compression as the next lever for this EMA/DC16 run; the current bottlenecks are fit-scale/scorer cache quality plus bytes over ceiling.
- No exact eval dispatch until byte-closed archive, receiver proof, full-video local prefilter, local CPU replay, compliance gates, and paired contest CPU/CUDA authority all pass.
