# Codex Session Summary - SNeRV Scorer Tether Launch Gate

## Landing

- Bound `tools/run_compact_renderer_mlx_spine_runner.py` so any nonzero SNeRV score-aware long-training launch runs `snerv_scorer_tether_smoke.v1` first and writes `snerv_scorer_tether_smoke_gate.json` beside the native export attachment.
- Failed tether smoke now returns `snerv_scorer_tether_smoke_failed_before_long_training` and never calls `train_export_snerv_mlx_native`.
- Bound actual native SNeRV long-training dual-ascent scorer tethers to strict zero targets so short prelaunch smokes prove active PosNet/SegNet lambdas instead of self-normalizing around relative first-step targets.

## Evidence

- SSD smoke report: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_tether_actual_training_smoke_strict_20260605Tcodex/compact_renderer_mlx_spine_runner_report.json`
- Gate artifact: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_tether_actual_training_smoke_strict_20260605Tcodex/snerv_mlx_native_export/snerv_scorer_tether_smoke_gate.json`
- Training telemetry: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_tether_actual_training_smoke_strict_20260605Tcodex/snerv_mlx_native_export/native_train_export/snerv_score_aware_long_training/long_training/telemetry.jsonl`
- Telemetry contract passed with `segnet_dual_metric_observed=true`, `posenet_dual_metric_observed=true`, `segnet_dual_lambda_active_observed=true`, and `posenet_dual_lambda_active_observed=true`.
- Final observed lambdas: SegNet `0.3652157992124558`, PosNet `6.0`.

## Verification

- `uv run python -m py_compile tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_train_export.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py`
- `uv run ruff check tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_train_export.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py`
- `uv run pytest src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py::test_snerv_scorer_tether_dual_targets_are_strict_before_long_training src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py::test_score_aware_telemetry_contract_accepts_live_dual_and_section_metrics src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py::test_score_aware_telemetry_contract_rejects_inactive_scorer_tether_lambdas src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_snerv_native_export_attachment_threads_mlx_prefilter_controls src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_snerv_native_export_attachment_blocks_failed_training_telemetry_contract src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_snerv_native_export_attachment_refuses_long_training_when_tether_smoke_fails src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_snerv_execute_parser_accepts_planner_gated_families src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_execute_snerv_attaches_native_mlx_export_evidence src/tac/tests/test_run_compact_renderer_mlx_spine_runner_snerv_skip_high.py -q`

## Remaining Blockers

The smoke is not a launch candidate. Remaining blockers include partial pair coverage, no exact CPU/CUDA eval, no full-video local prefilter, no local CPU replay gate, manual modelsize probe, missing scorer-loop QAT, and no byte-closed/full600 campaign readiness.
