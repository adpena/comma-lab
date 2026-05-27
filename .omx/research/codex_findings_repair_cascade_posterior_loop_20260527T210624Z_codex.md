# Codex Findings: Repair Cascade Posterior Loop Closure

UTC: 2026-05-27T21:06:24Z

## Landing

Closed the next automation gap in the encoder-side repair cascade loop. The
queue-owned Cascade-C MLX probe path no longer stops at a readiness/result JSON:
it now emits a generic `repair_campaign_learning_signal.v1` row and appends it
to the shared repair stackability posterior with duplicate suppression.

## What Changed

- `repair_cascade_mlx_probe_result.v1` now preserves the cascade pipeline
  position and targeted positions, so downstream learning keeps the
  entropy-position and pixel/region/boundary/frame/pair/batch/video context.
- Added `build_repair_cascade_mlx_learning_signal(...)`, which converts a
  false-authority cascade probe result into a planner-consumable
  `repair_campaign_learning_signal.v1`.
- The cascade queue now owns the full sequence:
  spec -> result -> learning signal -> stackability posterior append report.
- The queue metadata records posterior paths and append-report schema, and the
  worker validates the posterior JSONL remains false-authority.
- Added `tools/build_repair_cascade_mlx_learning_signal.py` for direct operator
  and queue execution.

## Authority

All emitted artifacts remain research/planning signal only:

- no score claim;
- no promotion eligibility;
- no rank/kill authority;
- no budget-spend authority;
- no exact-eval dispatch authority.

MLX remains `[macOS-MLX research-signal]`; component response and exact
CPU/CUDA eval are still required before budget spend, dispatch, or promotion.

## Verification

- `ruff check` on touched cascade queue, tool, CLI, and tests passed.
- `pytest src/tac/tests/test_repair_cascade_mlx_probe_queue.py -q`:
  7 passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`:
  57 passed.

## Remaining Frontier Gap

The posterior loop now captures blocked and ready cascade probe results, but the
concrete MLX component-response runner still needs to materialize the missing
Posenet-null, SegNet-region, selector-codec, and receiver-consumption artifacts.
That runner should emit measured SegNet/PoseNet deltas and selector bits into
the same learning-signal surface, then let exact CPU/CUDA auth gates decide
budget spend.
