# Codex Findings: PR95 MLX Source-Video Blocker Taxonomy

UTC: 2026-05-24T22:30:57Z
Lane: `codex_pr95_mlx_source_video_blocker_taxonomy_20260524`

## Finding

PR95 MLX timing/preprocess artifacts were using a coarse
`pr95_training_loop_not_yet_source_faithful` blocker across both synthetic
timing and source-video target paths. That blurred two different states:
synthetic timing is not source-video training at all, while the source-video
path now has RGB/YUV targets but still lacks scorer-network loss, full stage
schedule parity, QAT/resume parity, export-forward parity, full-frame inflate
parity, byte-closed archive export, and exact CPU/CUDA auth eval.

## Landing

- Added named PR95 blocker constants in
  `src/tac/local_acceleration/pr95_hnerv_mlx_contract.py`.
- Source-video timing profiles now mark `source_video_target_loss_training=true`
  for source-video target-loss runs, while keeping both
  `source_faithful_training=false` and
  `full_pr95_source_faithful_training=false` until scorer/curriculum/export/auth
  parity is actually wired.
- The executable PR95 MLX optimizer backend label is now
  `implemented_mlx_local_timing_proxy`, reflecting that the backend includes
  synthetic and source-video target-loss timing modes but remains proxy-only.
- Representation-training probe integration now carries
  `source_video_training_target` signal separately from preprocess signal.
- Plan-only PR95 MLX dispatch blockers now filter optimizer-level source-video
  loader and YUV6-preprocess blockers when the queued command itself will
  produce the corresponding source-video target/preprocess signal.
- CLI postconditions now assert the more precise blocker for preprocess versus
  source-video-target runs.
- Tests/tools consume the named constants instead of cargo-culting stale blocker
  strings through optimizer registry, timing smoke, runtime-consumption proof,
  optimizer-matrix queue, representation-training integration, and PR95 MLX
  tests.

## Authority Contract

The source-video MLX path remains local substrate/training signal only. These
changes do not make PR95 MLX outputs score claims, promotion candidates,
rank/kill evidence, or exact-eval dispatch authority. They sharpen the blocker
taxonomy so queue/autopilot consumers can distinguish "no source video target"
from "source video target present but full scorer/source-faithful pipeline still
unwired."

## Verification

- `.venv/bin/python -m ruff check src/tac/local_acceleration/pr95_hnerv_mlx_contract.py src/tac/local_acceleration/pr95_hnerv_mlx.py src/tac/local_acceleration/pr95_hnerv_mlx_training.py src/tac/optimization/optimizer_scheduler_registry.py src/tac/optimization/representation_training_probe_integration.py tools/run_pr95_mlx_timing_smoke.py tools/build_pr95_mlx_optimizer_matrix_queue.py tools/prove_pr95_public_archive_runtime_consumption.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_pr95_hnerv_mlx_training.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_pr95_muon_local_training_integration.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_pr95_hnerv_mlx_training.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_pr95_muon_local_training_integration.py -q` (`61 passed`)
- Live MLX source-video RGB+YUV6 one-step smoke:
  `tools/run_pr95_mlx_timing_smoke.py --stage 1 --steps 1 --batch-size 1 --synthetic-pairs 1 --seed 51 --base-channels 4 --latent-dim 8 --train-on-source-video-pairs --source-video-loss-surface rgb_yuv6_mse --source-video-path upstream/videos/0.mkv --source-video-upstream-dir upstream --source-video-pair-index 0 --source-video-output-hw 384,512 --write-source-faithful-preprocess-smoke --write-source-video-preprocess-smoke ...`
- Live smoke artifact:
  `experiments/results/codex_pr95_mlx_source_video_contract_20260524T224504Z/manifest.json`

## Next

The next PR95 MLX substrate step is scorer-network loss wiring on the
source-video target path, followed by stage schedule/QAT resume parity and a
byte-closed export-forward parity packet before any exact-auth dispatch.
