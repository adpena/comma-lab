# Codex Session Summary: SNeRV Bounded Tether/QAT Proof

timestamp_utc: 2026-06-05T17:50:14Z
agent: codex:gpt-5
lane_family: snerv_bounded_channelmean
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Landed

- Bound the SNeRV scorer-tether/QAT proof into the executable compact-family runner path instead of leaving it as an advisory side script.
- Forced weighted coder-QAT terms to actuate during the short PR95-faithful admission proof even when the current PR95 stage marks native QAT inactive.
- Taught interrupted compact-family reports to emit planner-consumable candidate feedback rows, including recovered SNeRV scorer-domain tether health and scorer-input guard proof.
- Added recovery CLI support:
  `uv run python tools/run_compact_renderer_mlx_spine_runner.py --recover-interrupted-report-from-startup-marker --interrupted-report-recovery-reason <reason> --output-dir <run_dir> --overwrite`

## Empirical Anchors

- v28b bounded live command:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_bounded_channelmean_v28b_fastcodec3_20260605Tcodex/snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1-2-4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelmean_int8_symmetric_ceil178000_native_rate_aware_training/compact_renderer_mlx_spine_runner_report.json`
- v28b candidate feedback row:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_bounded_channelmean_v28b_fastcodec3_20260605Tcodex/snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1-2-4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelmean_int8_symmetric_ceil178000_native_rate_aware_training/nerv_candidate_byte_feedback_row.json`
- v28b result: 3 telemetry rows, final epoch 2, telemetry contract passed, scorer-domain tether passed, scorer-input guard passed, renderer nondegenerate proof still failed closed.
- v27 recovered interrupted report:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_bounded_channelmean_v27_forced_qat_20260605Tcodex/snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1-2-4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelmean_int8_symmetric_ceil178000_native_rate_aware_training/compact_renderer_mlx_spine_runner_report.json`
- v27 result: 2 telemetry rows, final epoch 1, telemetry contract passed, scorer-domain tether passed, scorer-input guard passed, renderer nondegenerate proof still failed closed.

## Still Blocked

- `byte_closed_archive_export_missing`
- `receiver_proof_missing`
- `full_video_local_prefilter_missing`
- `local_cpu_replay_gate_missing`
- `paired_contest_cpu_cuda_pass_missing`
- `snerv_training_interrupted_before_export`
- `snerv_receiver_proof_missing`
- `snerv_full_video_local_prefilter_missing`
- `snerv_local_cpu_replay_gate_missing`
- renderer nondegenerate proof still lacks receiver reconstruction and value-domain/export-domain proof.

## Next DAG Edge

Use the v28b candidate feedback row as the planner input that clears the missing PosNet/SegNet tether and scorer-input guard blockers. The next SNeRV work should target the post-training compression/export stall and then the renderer nondegeneracy/value-domain proof, not another blind scalar/shared/spectra long launch.

## Verification

- Passed: `uv run pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_compact_family_interrupted_report_preserves_false_authority_evidence src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_recover_interrupted_report_from_startup_marker_summarizes_telemetry src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_recover_interrupted_report_summarizes_snerv_nested_long_training_telemetry src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py::test_pr95_stage_4_consumes_real_coder_qat_terms_NO_FAKE src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py::test_pr95_stage_qat_uses_dual_adjusted_extra_loss_weights_NO_FAKE src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py::test_pr95_stage_can_force_weighted_qat_for_short_admission_proof -q`
- Passed: `uv run pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_coder_qat.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py::test_snerv_official_section_qat_leaves_dummy_lf_non_actuated src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py::test_official_renderer_coder_qat_selects_hfr_decoder_weights src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py::test_pr95_stage_4_consumes_real_coder_qat_terms_NO_FAKE src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py::test_pr95_stage_qat_uses_dual_adjusted_extra_loss_weights_NO_FAKE src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py::test_pr95_stage_can_force_weighted_qat_for_short_admission_proof -q`
- Passed: `uv run ruff check tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py src/tac/substrates/_shared/mlx_score_aware/adapter.py src/tac/substrates/_shared/mlx_score_aware/harness.py src/tac/substrates/_shared/mlx_score_aware/tests/test_coder_qat.py src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_train_export.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py`
- Passed: `uv run python -m py_compile tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py src/tac/substrates/_shared/mlx_score_aware/adapter.py src/tac/substrates/_shared/mlx_score_aware/harness.py src/tac/substrates/_shared/mlx_score_aware/tests/test_coder_qat.py src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_train_export.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py`
- Passed: `git diff --check -- <owned SNeRV/QAT/recovery files>`
