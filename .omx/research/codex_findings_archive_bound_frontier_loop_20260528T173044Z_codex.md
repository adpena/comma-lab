# Codex Findings: Archive-Bound Frontier Loop

Date: 2026-05-28T17:30:44Z
Agent: codex

## Landed Slice

- Frontier-selected queues now carry an archive-bound default contract: selected experiments must emit byte-closed candidate archives, receiver runtime proofs, and exact-bridge preclaim inputs by default.
- Measured MLX marginal rows now feed posterior budget routing directly through `repair_family_measured_mlx_posterior_budget_routing_update.v1`.
- Stack search now promotes positive measured MLX marginals toward archive-bound materialization while preserving negative-posterior demotion priority.
- The autonomous floor loop now emits entropy-stage materializer work-order bundles for before-coder, coder-boundary, and after-coder stages.
- The multi-archive runner now runs the floor loop as a bounded live-archive loop with configurable `max_floor_iterations` and terminal outcome reporting.

## Authority Boundary

All new artifacts remain false-authority. MLX marginals route local acquisition only; archive-bound defaults require custody; exact dispatch still requires exact CPU/CUDA authority, provider preclaim, lane claim, receiver/runtime custody, and the existing exact-ready gates.

## Verification

- `ruff check` on touched files: passed.
- `git diff --check` on touched files: passed.
- `pytest src/tac/tests/test_repair_family_materializers.py -q`: 24 passed.
- `pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`: 10 passed.
- `pytest src/tac/tests/test_repair_autonomous_multi_archive_runner.py -q`: 1 passed.

## Recursive Senior Engineer Review

Pass 1, routing correctness: clean. Positive measured MLX marginals now request archive-bound materialization, but negative posterior demotion still wins when the family is already demoted.

Pass 2, entropy-stage compiler contract: clean. The floor loop emits materializer work orders split by before-coder, coder-boundary, and after-coder positions without granting score or dispatch authority.

Pass 3, live archive loop closure: clean. The multi-archive runner now records bounded iteration policy and terminal outcomes for live archive campaigns, with exact-axis authority still fail-closed.
