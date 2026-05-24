# Codex Findings: PR95 MLX Source-Video Training Targets

UTC: 2026-05-24T20:30:54Z
Lane: codex_pr95_mlx_source_video_training_targets_20260524

## Finding

The previous PR95 MLX source-video preprocess smoke proved real decoded
contest-video pair loading and scorer-resolution RGB/YUV6 preprocessing, but
the MLX timing-smoke training loop still used synthetic RGB targets. That kept
PR95 local training farther from the intended source-video substrate and left
optimizer-matrix queue rows unable to request source-backed target batches.

## Landing

- `run_pr95_mlx_synthetic_timing_smoke(...)` now accepts supplied
  `target_pairs_n2chw` and records `source_video_training=true` with
  `training_fidelity=source_video_rgb_timing_only`.
- `tools/run_pr95_mlx_timing_smoke.py` adds
  `--train-on-source-video-pairs`, decodes PR95 source-video pair frames,
  preprocesses them to scorer-resolution RGB targets, writes
  `source_video_training_target.json`, and passes those targets into the MLX
  timing smoke.
- `tools/build_pr95_mlx_optimizer_matrix_queue.py` propagates
  `--train-on-source-video-pairs` into generated queue commands and cell
  identity so optimizer matrices can distinguish synthetic and source-video
  timing substrates.
- Representation manifests and candidate rows now preserve target-source
  metadata for downstream local-training and spend-triage consumers.

## Authority

This is still `[macOS-MLX research-signal]`. Source RGB targets replace the
synthetic target blocker, but they do not establish full scorer loss,
export parity, byte-closed archive authority, CPU/CUDA parity, promotion,
rank/kill, or score authority. Exact-readiness blockers now say that
source-video RGB timing targets are not full scorer-quality authority.

## Verification

- `.venv/bin/python -m ruff check src/tac/local_acceleration/pr95_hnerv_mlx.py tools/run_pr95_mlx_timing_smoke.py tools/build_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py -q`

## Remaining Gap

The next PR95/NeRV-family substrate-training gap is scorer-loss coupling:
train the MLX substrate against differentiable scorer-preprocess/scorer
surrogates or a calibrated local scorer-response surface, then export into a
byte-closed archive/runtime packet for exact CPU/CUDA evaluation. The broader
inverse-steganalysis gap remains deterministic operation-set lowering for
non-leaf cells.
