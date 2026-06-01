# HiNeRV Coder-Aware QAT Wiring Landed

UTC: 2026-06-01T22:46:53Z
Author: Codex
Axis: [macOS-MLX research-signal], non-authority

## Finding

HiNeRV now consumes the shared decoder coder-aware QAT loss surface through the
same MLX score-aware harness used by the compact NeRV/VQ families. The runner
threads the CLI/config knobs into `_run_hi_nerv_mlx_scoreaware_smoke`, attaches
`extra_loss_terms` / `extra_loss_weights` to the `RendererBundle`, and reports
the substrate-supplied QAT metadata after the shared adapter renames it to
`substrate_supplied_score_aware_training`.

The metadata reader was a real bug class: reading
`substrate_artifact_metadata.score_aware_training.coder_aware_qat` drops the
signal on real harness artifacts, because the canonical harness owns
`score_aware_training` for its own objective summary. The runner now reads the
substrate-supplied slot first and falls back to CLI metadata if an older/test
artifact lacks it.

## Smoke

Command family:

`tools/run_compact_renderer_mlx_spine_runner.py --execute-family hi_nerv --num-pairs 2 --epochs 1 --batch-pairs 1 --coder-aware-qat --coder-qat-quant-bits 4 --coder-qat-quant-residual-weight 0.001 --coder-qat-magnitude-weight 0.0001 --coder-qat-delta-weight 0.0002`

Artifact:

`/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_qat_smoke_20260601T224653Z/compact_renderer_mlx_spine_runner_report.json`

Result:

- archive bytes: 39,640
- archive sha256: `1438b7a82618dc98d6c12d2fb23248de761d37c575d525bdedf8798838cd99d6`
- receiver proof emitted:
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_qat_smoke_20260601T224653Z/hi_nerv_mlx_training/receiver_proof/hi_nerv_mlx_receiver_proof.json`
- retained footprint: 908K on SSD
- retained raw/video artifacts: none found under the smoke output
- score claim: false
- ready for exact eval dispatch: false

Expected blockers remained:

- `contest_cpu_cuda_exact_eval_not_executed`
- `local_cpu_replay_not_run_partial_pair_coverage`
- `hi_nerv_pr95_faithful_curriculum_requires_min_8_epochs`
- `hi_nerv_real_segnet_posenet_teachers_not_both_attached`
- `full_video_mlx_scorer_replay_not_attached`
- `no_full_coverage_compact_base_candidate`
- `no_full_coverage_candidate_under_any_hard_ceiling`

## Verification

- `ruff check tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
- `pytest --import-mode=importlib -q src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
- `pytest --import-mode=importlib -q src/tac/substrates/_shared/mlx_score_aware/tests/test_coder_qat.py`

## Next

Run the next HiNeRV/SNeRV tranche with real SegNet/PoseNet teachers, >=8 epoch
PR95-faithful curriculum, coder-aware QAT enabled, singleton full-video MLX
prefilter, and local CPU replay only for MLX-filtered candidates. The immediate
score-lowering question is whether rate-aware decoder-weight training can move
the current HiNeRV full600 demotion (`MLX score 90.723...`) into a plausible
local-replay band before exact CPU/CUDA spend.
