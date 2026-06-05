# Codex Session Summary - SNeRV/HiNeRV MLX False-Authority Gate 2026-06-04T184607Z

## Scope

- Operator directive: continue SNeRV/HiNeRV locally on CPU/MLX; Modal only for exact auth eval of frontier candidates.
- Priority: explain the terrible local scores, exploit upstream `evaluate.py` geometry, and prevent bad local MLX artifacts from entering replay or dispatch authority.

## Crux

- The bad SNeRV checkpoint is not an exact-eval candidate. Its archive is byte-closed and receiver-proven, but local MLX prefilter evidence is component-bad and scorer-input-bad.
- The concrete SNeRV crux is `official_skip_high_mode=scalar_mean`: the receiver header stores skip-high as `[1,1,1,1]` / 8 raw bytes for a source carrier shaped `[1200,3,192,256]`, then expands it at decode. That is not lossless relative to the source skip-high carrier and it matches the saturated/clipped scorer-input diagnosis.
- HiNeRV has the same class of false-authority risk at the post-export receiver cache gate: low local score or full-video singleton coverage is not enough if `quality_gate_passed=false` or the gate reports `FIT_OR_SCALE_FAILURE`.

## Landed Artifacts

- SNeRV receiver value-domain xray:
  - `tools/xray_snerv_receiver_value_domain.py`
  - `.omx/research/snerv_receiver_value_domain_xray_epoch003199_pairs001599_20260604Tcurrent_codex.json`
- MLX scorer-input diagnosis and coverage hardening:
  - `src/tac/local_acceleration/mlx_renderer_prefilter_profile.py`
  - `src/tac/substrates/hprc/mlx_prefilter_coverage.py`
- HiNeRV/SNeRV shared local replay guard now blocks:
  - saturated or clipped scorer inputs;
  - scorer-input out-of-distribution diagnoses;
  - direct `cache_quality_gate` failures;
  - nested HiNeRV `post_export_receiver_cache_quality` failures in training metadata.
- Lane registered and marked:
  - `lane_snerv_mlx_prefilter_saturation_gate_20260604`
  - gates: `impl_complete`, `real_archive_empirical`, `strict_preflight`, `memory_entry`
  - not marked: `contest_cpu`, `contest_cuda`, `three_clean_review`, `deploy_runbook`

## Verification

- `uv run pytest src/tac/local_acceleration/tests/test_mlx_renderer_prefilter_profile.py src/tac/tests/test_hprc_mlx_prefilter_coverage.py src/tac/substrates/hprc/tests/test_mlx_prefilter_coverage.py src/tac/tests/test_xray_snerv_receiver_value_domain.py -q` -> 17 passed.
- `uv run pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_full_video_bad_mlx_score_does_not_unlock_default_cpu_replay src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_batched_full_video_mlx_prefilter_feeds_acquisition_not_replay src/tac/tests/test_hprc_mlx_prefilter_coverage.py src/tac/substrates/hprc/tests/test_mlx_prefilter_coverage.py -q` -> 11 passed.
- `uv run ruff check src/tac/substrates/hprc/mlx_prefilter_coverage.py src/tac/tests/test_hprc_mlx_prefilter_coverage.py src/tac/local_acceleration/mlx_renderer_prefilter_profile.py src/tac/local_acceleration/tests/test_mlx_renderer_prefilter_profile.py tools/xray_snerv_receiver_value_domain.py src/tac/tests/test_xray_snerv_receiver_value_domain.py` -> passed.
- `uv run python -m py_compile src/tac/substrates/hprc/mlx_prefilter_coverage.py src/tac/local_acceleration/mlx_renderer_prefilter_profile.py tools/xray_snerv_receiver_value_domain.py src/tac/tests/test_xray_snerv_receiver_value_domain.py` -> passed.
- `uv run python tools/lane_maturity.py validate` -> 1643 lanes clean.

## Live State

- Live local SNeRV MLX run remains active:
  - parent PID `45607`
  - child PID `45608`
  - output root `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_scalarmean_hardpair_successor_fix2_20260604Tcurrent_codex`
- Storage is healthy for continued local work:
  - `/Volumes/VertigoDataTier` had about 845 GiB available at handoff.

## Next Work

1. Do not Modal-dispatch the `epoch003199` SNeRV scalar-mean packet. It is under byte cap but scorer-input and skip-high value-domain blockers are real.
2. Let the live local run continue unless it emits a materially better checkpoint. Harvest only after the local profile passes scorer-input/cache-quality gates, or after a representation change replaces scalar-mean skip-high with full/shared/channel skip-high evidence.
3. For SNeRV, next implementation should compare scalar/channel/shared/full skip-high local profiles under the same upstream `evaluate.py` geometry, with frame-1 SegNet and two-frame PoseNet deltas reported separately.
4. For HiNeRV, treat `FIT_OR_SCALE_FAILURE` / `quality_gate_passed=false` as a hard local replay blocker. Focus the next smoke on cache quality repair before exact auth eval.
5. Tilde leverage remains local/control-plane only: Aurora-like optimizer timing smoke is useful; Wall Attention is only inspiration for a byte-charged SNeRV LF/TUB temporal gate; Parallax direct runtime import remains forbidden.

## Authority

All MLX rows above are `[macOS-MLX research-signal]` / false-authority. No score, rank, promotion, or exact-eval readiness is claimed. Exact CPU/CUDA authority still requires byte-closed `archive.zip` plus deterministic runtime through the official evaluator axis.
