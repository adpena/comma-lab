# Codex Session Summary - SNeRV Skip-High And HiNeRV Stage-QAT Harvest 2026-06-04T212057Z

## Scope

- Operator directive: continue all SNeRV/HiNeRV work locally on CPU/MLX; Modal remains reserved for exact auth eval of true frontier candidates.
- This continuation avoided new MLX launches because the SNeRV scalar-mean long run was active and a HiNeRV stage-QAT local claim had just produced artifacts.

## SNeRV Result

- Added reusable comparison helper:
  - `src/tac/analysis/snerv_skip_high_mode_compare.py`
  - `tools/compare_snerv_skip_high_modes.py`
  - `src/tac/tests/test_snerv_skip_high_mode_compare.py`
- Wrote artifacts:
  - `.omx/research/snerv_skip_high_mode_comparison_20260604Tcodex.json`
  - `.omx/research/snerv_skip_high_mode_comparison_20260604Tcodex.md`
- Compared existing binary profiles:
  - `scalar_mean`: archive `91445` bytes; skip-high stored shape `[1,1,1,1]`; stored raw bytes `8`; under cap but scalar-collapse risk.
  - `shared_mean`: archive `436084` bytes; skip-high stored shape `[1,3,192,256]`; stored raw bytes `1179648`; non-scalar but `258084` bytes over 178000 cap.
- Attached scalar epoch003199 local MLX prefilter:
  - score `90.86453145613247`
  - Seg term `50.48246002693971`
  - Pose term `40.31973371400145`
  - local replay admissible `false`
  - scorer input OOD `true`
- Verdict: `NO_CURRENT_SKIP_HIGH_MODE_READY_FOR_EXACT_EVAL`.

## HiNeRV Result

- Terminalized local claim `hinerv_stage_qat_runner_smoke_20260604_codex` as completed false-authority degenerate-cache-gate evidence.
- Wrote harvest artifacts:
  - `.omx/research/hinerv_stage_qat_smoke_harvest_20260604Tcodex.json`
  - `.omx/research/hinerv_stage_qat_smoke_harvest_20260604Tcodex.md`
- Harvested stage-QAT smoke:
  - output root `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_stage_qat_runner_smoke_20260604T2111Z_codex`
  - archive `106351` bytes, SHA-256 `85c1c44936560f87d0cf300392bdb8cba2cfe16abad47f7a9ba239960ca80890`
  - local sampled MLX score `94.93991807867756`
  - receiver cache gate `FUNDAMENTAL_RENDERER_OUTPUT_DEGENERATE`
  - SegNet cache dynamic range `4.9344635009765625`, std `0.517579197883606`, MAE `101.44928741455078`
  - PoseNet cache dynamic range `3.3023681640625`, std `0.43814340233802795`, MAE `69.05482482910156`
- Verdict: `BYTE_ATTRACTIVE_BUT_RENDERER_DEGENERATE_NOT_A_FRONTIER_CANDIDATE`.

## Lane State

- `lane_snerv_skip_high_mode_comparison_20260604`: L2 with `impl_complete`, `real_archive_empirical`, `strict_preflight`.
- `hinerv_stage_qat_runner_smoke_20260604_codex`: L1 with `real_archive_empirical`, `strict_preflight`, `memory_entry`.
- Registry validation: `1645 lane(s) validated cleanly`.

## Verification

- `uv run pytest src/tac/tests/test_snerv_skip_high_mode_compare.py src/tac/tests/test_xray_snerv_receiver_value_domain.py src/tac/tests/test_hprc_mlx_prefilter_coverage.py src/tac/substrates/hprc/tests/test_mlx_prefilter_coverage.py src/tac/local_acceleration/tests/test_mlx_renderer_prefilter_profile.py -q` -> 18 passed.
- `uv run ruff check src/tac/analysis/snerv_skip_high_mode_compare.py tools/compare_snerv_skip_high_modes.py src/tac/tests/test_snerv_skip_high_mode_compare.py src/tac/substrates/hprc/mlx_prefilter_coverage.py src/tac/tests/test_hprc_mlx_prefilter_coverage.py tools/xray_snerv_receiver_value_domain.py src/tac/tests/test_xray_snerv_receiver_value_domain.py` -> passed.
- `uv run python tools/lane_maturity.py validate` -> 1645 lanes clean.

## Next Local Work

1. Do not Modal-dispatch scalar_mean SNeRV or this HiNeRV stage-QAT archive.
2. Let the active SNeRV scalar-mean run continue, but treat scalar_mean as a falsification path unless a future value-domain xray disproves the collapse mechanism.
3. After MLX claims clear, run the next SNeRV skip-high smoke on a non-scalar storage mode with byte pressure, then compare by frame-1 SegNet, two-frame PoseNet, archive bytes, and scorer-input/cache quality.
4. For HiNeRV, repair renderer dynamic range before any full-video replay. The 106 KB archive is capacity signal only until receiver cache quality passes.

## Authority

All rows here are local false-authority planning/triage evidence. No score, rank, promotion, kill, or exact-eval readiness is claimed.
