# Codex Findings: Repair Frontier Autonomous Floor Loop Closure

Date: 2026-05-28T15:05:38Z
Agent: codex

## Landed Slice

- The autonomous floor loop now consumes ranked stack-acquisition frontier paths as executable work selection when `--execute-local` is enabled, writing an iteration-local filtered queue that runs only the selected family/typed-response rows.
- The fractal marginal surface now carries measured MLX marginal updates for each family/stage/scope cell, including improvement per byte, delta bytes, source execution report, and false-authority markers.
- Exact-ready bridge output now emits per-candidate Lightning provider preclaim artifacts before any exact dispatch claim can be made.
- Failed receiver/runtime/archive/preclaim outcomes now produce durable rebudgeting updates that demote the responsible family, entropy stage, and fractal scope when the blocker identifies a broken layer.

## Authority Boundary

All new loop outputs remain false-authority. Frontier selection is execution routing only; MLX marginal updates are advisory only; preclaim checks do not claim exact readiness; exact CPU/CUDA dispatch still requires byte-closed archive/runtime custody and a valid lane claim.

## Verification

- `ruff check tools/run_repair_campaign_autonomous_floor_loop.py src/tac/optimization/repair_family_stack_search.py src/tac/tests/test_repair_campaign_materialization_queue.py src/tac/tests/test_repair_family_materializers.py`: passed.
- `git diff --check` on touched files: passed.
- `pytest src/tac/tests/test_repair_family_materializers.py -q`: 24 passed.
- `pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`: 10 passed.
- Review tracker: three Codex review passes marked; policy checks report 0 violations for all touched files.

## Recursive Senior Engineer Review

Pass 1, executable frontier routing: clean. The runner no longer treats frontier paths as display-only; it creates a queue-owned selection artifact and runs the selected queue when local execution is requested.

Pass 2, measured-signal preservation: clean. Every fractal cell receives measured MLX marginal rows without granting score, promotion, rank/kill, budget-spend, or exact-dispatch authority.

Pass 3, failure-loop closure: clean. Exact bridge rows now get preclaim artifacts, and failed exact/preclaim/receiver/runtime/archive blockers produce rebudgeting updates instead of dead-end reports.
