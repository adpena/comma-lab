# Codex Findings: PR95 MLX Source-Video Training Queue Smoke

UTC: 2026-05-24T20:37:00Z
Lane: codex_pr95_mlx_source_video_training_targets_20260524

## Finding

The PR95 MLX source-video target path now works through both direct execution
and `experiment_queue.v1`. This preserves the intended authority split:
source-video RGB targets are materially closer to PR95 reproduction than
synthetic targets, but remain `[macOS-MLX research-signal]` until scorer loss,
export parity, byte-closed archive replay, and exact CPU/CUDA auth-eval gates
exist.

## Direct Smoke Evidence

- Artifact root:
  `experiments/results/codex_pr95_mlx_source_video_training_20260524T203254Z`
- Candidate:
  `pr95_hnerv_mlx_stage1_pr95_stage1_adamw_baseline_mlx_seed41_steps1_c4_source_video_rgb`
- `training_fidelity=source_video_rgb_timing_only`
- `source_video_training=true`
- `target_source.kind=pr95_source_video_rgb_pairs`
- `target_shape_n2chw=[1,2,3,384,512]`
- `synthetic_pairs=null`
- `training_pair_count=1`
- `seconds_per_step=0.02733625000109896`
- Authority remains false:
  `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`

## Queue Evidence

- Queue root:
  `experiments/results/codex_pr95_mlx_source_video_training_queue_20260524T203303Z`
- Queue id:
  `codex_pr95_source_video_training_queue_smoke`
- Worker outcome:
  `success_count=1`, `failure_count=0`, `claim_refused_count=0`
- Executed experiment:
  `local_training_0000_pr95_hnerv_mlx_stage1_pr95_stage1_adamw_baseline_mlx_seed43_steps1_c4_source_video_rgb`
- Postconditions proved all required artifacts, including:
  `manifest.json`, `representation_training_manifest.json`, and
  `source_video_training_target.json`
- `source_video_training_target.json` recorded
  `source_video_training_ready=true`,
  `training_fidelity=source_video_rgb_timing_only`, pair frame indices `[0,1]`,
  full source frame shape `[1,2,874,1164,3]`, scorer target shape
  `[1,2,3,384,512]`, and false-authority fields.

## Verification

- `.venv/bin/python -m ruff check src/tac/local_acceleration/pr95_hnerv_mlx.py src/tac/local_acceleration/pr95_hnerv_mlx_training.py tools/run_pr95_mlx_timing_smoke.py tools/build_pr95_mlx_optimizer_matrix_queue.py src/tac/optimization/representation_training_probe_integration.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_pr95_hnerv_mlx_training.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_representation_training_probe_integration.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_pr95_hnerv_mlx_training.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_representation_training_probe_integration.py -q`
  passed with `43 passed in 6.67s`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check src/tac/local_acceleration/pr95_hnerv_mlx.py tools/run_pr95_mlx_timing_smoke.py tools/build_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py`
  reported `0 violations`.
- `.venv/bin/python tools/lane_maturity.py validate`
  reported `1276 lane(s) validated cleanly`.

## Next Engineering Gap

The next non-leaf PR95/NeRV-family gap is to replace RGB timing loss with a
source-faithful MLX objective surface: scorer-preprocess-coupled loss first,
then staged schedule/QAT parity, PyTorch export-forward parity, and
byte-closed runtime replay. Queue ownership is now in place for source-backed
cells, so the next work should be scorer-loss and export closure rather than
more synthetic timing cells.
