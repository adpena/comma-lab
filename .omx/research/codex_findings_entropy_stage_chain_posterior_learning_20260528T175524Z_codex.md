# Entropy-stage chain posterior learning closure

Codex landing: make composed repair chain execution feed the autonomous
posterior loop, not only produce candidate archives.

## What changed

- Chain stage reports now preserve the source stage entropy label/order,
  fractal scope, allocated bytes, byte-transform delta, and MLX-local probe
  delta from the leaf execution report.
- `build_repair_family_stack_learning_signal_report(...)` accepts a
  `repair_entropy_stage_chain_execution_bundle.v1` and appends one
  posterior-consumable learning signal per executed chain stage.
- The bounded autonomous floor loop now writes operator-visible counts for
  stack-row learning versus entropy-stage chain learning.

## Authority boundary

The new rows remain local acquisition updates only. They can increase/demote
family/stage/scope priors and exact-handoff routing, but they cannot claim
score, promote, dispatch, rank, kill, or spend budget. Exact CPU/CUDA authority
still requires archive/runtime custody, lane claim, preclaim gate, and
contest-axis harvest.

## Verification

- `ruff check --fix` on touched files: passed
- `py_compile` on touched Python modules/tools: passed
- `pytest src/tac/tests/test_repair_family_materializers.py::test_entropy_stage_chain_executor_composes_selected_archive_stages -q`: passed
- `pytest src/tac/tests/test_repair_campaign_materialization_queue.py::test_real_archive_intake_runs_all_families_through_floor_loop -q`: passed
- `pytest src/tac/tests/test_repair_family_materializers.py -q`: 25 passed
- `pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`: 10 passed

## Review notes

The closure avoids using chain rows as exact authority. The chain signal only
changes posterior acquisition pressure. The most important follow-up is to let
the exact bridge consume composed chain candidates directly when their custody
is complete, so the promoted candidate is the final chained archive rather than
only the independent leaf archives.
