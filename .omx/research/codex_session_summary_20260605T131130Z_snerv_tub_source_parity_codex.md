# Codex Session Summary: SNeRV TUB Source-Parity Executable Closure

UTC: 2026-06-05T13:11:30Z
Lane: lane_snerv_tub_source_parity_executable_20260605
Authority: macOS-MLX research-signal / receiver-packet proof plumbing only; no contest CPU/CUDA score, rank, kill, promotion, or frontier claim.

## Landed

- Added `snerv_official_tub_source_fixture_binding.v1` metadata in `src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_train_export.py`.
- The native official MFU/HFR/TUB packet now removes only `snerv_official_tub_batched_temporal_context_source_forward_replay_missing` from `official_source_parity_blockers` when the executable long-training replay contract proves:
  - TUB `output_2` source fixture replay passed;
  - the score-aware train renderer was bound;
  - the trained receiver state was bound;
  - receiver official payload replay passed.
- Full MFU/HFR/upstream checkpoint parity remains blocked. `source_faithful_stack` and `snerv_official_mfu_hfr_tub_source_forward_replay_authority` remain false.
- `tools/run_compact_renderer_mlx_spine_runner.py` now preserves the TUB fixture binding and source-parity blocker list at the SNeRV native export attachment top level.

## Validation

- `uv run pytest src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py::test_official_long_training_keeps_trained_packet_with_nonrender_blocker src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py::test_official_primitives_full_video_long_training_defers_replay_gate src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py::test_official_primitives_long_training_exports_trained_official_payload src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_snerv_native_export_attachment_threads_mlx_prefilter_controls -q`
- `uv run ruff check src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_train_export.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
- `uv run python tools/lane_maturity.py validate`

## Runnable Proof Command

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
uv run python tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family snerv \
  --allow-manual-compact-family-launch \
  --output-dir "/Volumes/VertigoDataTier/pact/experiments/results/snerv_tub_source_parity_proof_${RUN_ID}_codex" \
  --num-pairs 2 \
  --epochs 1 \
  --modelsize-candidate-id auto \
  --hard-byte-ceiling 178000 \
  --snerv-official-modelsize-mparams 0.07 \
  --snerv-modelsize-control-profile contest_receiver_profile \
  --snerv-official-skip-high-mode full \
  --snerv-score-aware-long-training-epochs 1 \
  --snerv-score-aware-long-training-batch-pairs 2 \
  --snerv-score-aware-long-training-lr 0.001 \
  --snerv-score-aware-long-training-scorer-tether-smoke-steps 2 \
  --segnet-distillation-weight 1.0 \
  --pose-distillation-weight 1.0 \
  --scorer-input-contrast-floor-weight 0.01 \
  --skip-local-cpu-replay
```

Expected proof fields in the SNeRV native attachment/report:

- `snerv_official_tub_source_fixture_replay_bound: true`
- `snerv_official_tub_source_fixture_replay_passed: true`
- `snerv_official_tub_source_forward_fixture_bound: true`
- `official_source_parity_blockers` does not include `snerv_official_tub_batched_temporal_context_source_forward_replay_missing`
- full source/score authority fields remain false.
