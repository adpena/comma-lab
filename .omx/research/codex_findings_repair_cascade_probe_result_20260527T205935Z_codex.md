# Repair Cascade Probe Result Closure - Codex Findings

Date: 2026-05-27T20:59:35Z
Author: Codex

## What Landed

- Added `repair_cascade_mlx_probe_result.v1`, a deterministic false-authority result artifact for Cascade-C MLX-local probe specs.
- Added `tools/run_repair_cascade_mlx_probe.py` so the repair cascade queue now writes a durable result after building a probe spec.
- Updated the repair cascade queue to publish `probe_result_path`, run the result step, and require false authority plus `ready_for_exact_eval_dispatch=false` on the result artifact.
- The result records exact missing MLX-local artifact blockers, component-response axis, required measurements, learning-signal kind, and the next acquisition action.

## Authority Boundary

The result is not a score, not a budget-spend decision, not a component-response replay, and not exact-eval authority. When artifacts are missing it emits `blocked_repair_cascade_mlx_probe`; when artifacts are present it still requires a concrete MLX component-response runner before any response row can enter the waterfill/action-functional ledger.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/repair_cascade_mlx_probe_queue.py tools/run_repair_cascade_mlx_probe.py src/tac/tests/test_repair_cascade_mlx_probe_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_cascade_mlx_probe_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py::test_targeted_component_correction_materialization_requests_group_responses -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `.venv/bin/python tools/lane_maturity.py validate`
- `.venv/bin/python tools/review_gate_hook.py`

## Next Integration Hooks

- Convert `repair_cascade_mlx_probe_result.v1` rows into repair-campaign posterior learning signals.
- Build the concrete MLX component-response runner that consumes the required Cascade-C artifacts and emits targeted component-response harvest rows.
- Keep all MLX rows advisory until full receiver-consumed byte closure plus contest CPU/CUDA exact-axis confirmation exists.
