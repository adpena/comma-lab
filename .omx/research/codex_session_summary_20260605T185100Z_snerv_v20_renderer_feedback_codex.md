# Codex Session Summary - SNeRV V20 Renderer Feedback

## Scope

Continued the SNeRV DAG burndown from the bounded renderer/value smoke and
normal planner surfaces. All evidence remains false-authority; no score,
promotion, rank, kill, or exact-eval claim is made.

## Landed

- Patched `tools/run_compact_renderer_mlx_spine_runner.py` so bounded
  planner-row timing smokes skip expensive SNeRV scorer-loop QAT packet
  compression by default. The skip is machine-readable via
  `snerv_bounded_smoke_scorer_loop_qat_policy`, and explicit opt-in remains
  available through `--snerv-bounded-smoke-allow-scorer-loop-qat`.
- Patched `src/tac/analysis/nerv_long_training_campaign_plan.py` so
  candidate-feedback auto-discovery prefers SNeRV proof quality
  (renderer nondegenerate proof, scorer tether, scorer-input guard, clean
  direct feedback) before raw telemetry progress. This prevents older
  longer-but-blocked rows from shadowing newer proof rows.
- Added focused regression tests for bounded scorer-loop skip/opt-in and
  proof-quality feedback ranking.
- Corrected the SNeRV telemetry-contract test fixture so section-byte dual
  weight-application failure is only expected when a positive section-byte
  violation exists.

## Evidence

- Patched bounded smoke:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_candidate_channelmean_v30_bounded_no_scorer_loop_20260605Tcodex/snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1-2-4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelmean_int8_symmetric_ceil178000_native_rate_aware_training/compact_renderer_mlx_spine_runner_report.json`
- Candidate feedback row:
  `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_candidate_channelmean_v30_bounded_no_scorer_loop_20260605Tcodex/snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1-2-4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelmean_int8_symmetric_ceil178000_native_rate_aware_training/nerv_candidate_byte_feedback_row.json`
- V20 normal planner artifacts:
  `/Volumes/VertigoDataTier/pact/experiments/results/nerv_campaign_plan_v20_bounded_renderer_feedback_20260605T184810Z/`

## Current Verdict

V30 proves the actual short SNeRV command path can emit:

- non-missing PosNet and SegNet scorer-tether telemetry;
- active scorer-domain lambdas;
- scorer-input distribution guard telemetry;
- 16-pair renderer nondegenerate proof;
- receiver reconstruction/value-domain proof;
- byte-under-target dual telemetry without false byte-dual blockers.

It remains blocked, correctly, by partial-pair scope, missing full600/full-video
prefilter, missing source-forward MFU/HFR/TUB authority, missing scorer-loop best
packet materialization, and missing local CPU/exact CPU/CUDA replay gates.

## Next DAG Edge

1. Run the next SNeRV evidence step as full600 or hard-pair/full-video scorer
   prefilter, not another tiny scalar smoke.
2. Unblock official MFU/HFR/TUB trained source-forward mapping before treating
   `official_tub_lf_hf_decoder_replacement` as runnable.
3. For LF/HF families, `lf_conditioned_hf_residual_generator` and
   `joint_lf_hf_factorized_codebook` are now locally bounded-smoke ready; keep
   them false-authority until full receiver/source-forward gates close.
4. Keep HiNeRV parallel work focused on dynamic-range/scorer-input stabilization
   plus byte-closed archive/export/receiver replay.

## Validation

- `uv run ruff check ...` on owned SNeRV/planner slice passed.
- Targeted pytest bundle passed: 241 tests.
- `uv run python -m py_compile ...` passed.
- `git diff --check ...` passed.
