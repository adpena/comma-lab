# Codex Findings - PR95 MLX Source Preprocess

## Verdict

Landed the first source-faithful PR95/HNeRV MLX training-preprocess surface as
local research signal only. The lane now has a native MLX eval-roundtrip/YUV6
smoke, gradient reachability proof, timing-smoke artifact wiring, and
optimizer-matrix queue propagation with execution postconditions.

## What Changed

- Added `tac.local_acceleration.pr95_hnerv_mlx_training` with NHWC MLX
  bicubic/bilinear resize, PR95 eval-roundtrip uint8 STE, RGB-to-YUV6, and a
  gradient probe.
- Wired `tools/run_pr95_mlx_timing_smoke.py` to emit
  `source_faithful_preprocess_smoke.json`, preserve the preprocess config in
  run manifests, representation manifests, and recommended execution commands,
  and refuse score/exact-readiness authority.
- Wired `tools/build_pr95_mlx_optimizer_matrix_queue.py` so matrix/DAG-owned
  PR95 optimizer sweeps can request the same preprocess smoke per cell.
- Added `json_array_contains` queue postconditions and normalized the
  preprocess smoke into harvested representation-training candidate payloads so
  downstream planners can query gradient reachability and exact-readiness
  blockers directly.
- Extended tests to cover PyTorch parity for resize/YUV/eval-roundtrip,
  edge/saturation and odd-crop YUV behavior, actual-resolution eval-roundtrip
  parity, gradient reachability, false-authority semantics, queue
  postconditions, and worker execution/harvest with the preprocess artifact
  enabled.

## Adversarial Notes

- Random resize parity was not enough; half-value rounding was checked directly
  and MLX/PyTorch both use bankers rounding on the tested values.
- The first queue integration pass had a hidden config-loss bug: plan emission
  received the small preprocess fixture shape, but the queued execution command
  would have fallen back to defaults. This was fixed by carrying
  `--source-preprocess-*` through `python_command_args`, manifest metadata, and
  harvested candidate params.
- Sidecar adversarial review found that a weak `source_faithful_preprocess_ready`
  bit was not enough queue evidence. Queue-owned execution now requires
  gradient reachability, exact-readiness refusal, a known source-faithfulness
  blocker, and false-authority fields.
- The artifact is not a score claim. It still blocks on source-video loader
  integration, scorer-loss wiring, real training-loop parity, PyTorch export
  forward parity, byte-closed archive export, and exact contest CPU/CUDA eval.
- A second adversarial pass found stale source-faithfulness wording in older
  PR95 MLX surfaces. The blocker was narrowed from "preprocess not ported" to
  "ported but scorer loss/training loop not wired," and the preprocess-ready
  flag now requires gradient reachability instead of being unconditional.
- Registry evidence is tied back to the canonical
  `lane_pr95_hnerv_mlx_reproduction` lane. The source-preprocess row is only an
  implementation sublane and carries no dispatch authority.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue.py src/tac/local_acceleration/pr95_hnerv_mlx.py src/tac/local_acceleration/pr95_hnerv_mlx_training.py src/tac/optimization/optimizer_scheduler_registry.py src/tac/optimization/representation_training_probe_integration.py tools/run_pr95_mlx_timing_smoke.py tools/build_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_pr95_hnerv_mlx_training.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_pr95_hnerv_mlx_training.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py -q` => 101 passed

## Next Integration Targets

- Replace synthetic RGB fixtures with PR95 source-video loader batches and keep
  the same MLX preprocess path under queue ownership.
- Wire the actual PR95/HNeRV MLX training loss through YUV6/scorer-compatible
  outputs, then compare step timing and exported checkpoint parity against the
  PyTorch reproduction lane.
- Promote the preprocess artifact into the broader inverse-steg/acquisition
  planner only as a calibrated feature source, not as score authority.
