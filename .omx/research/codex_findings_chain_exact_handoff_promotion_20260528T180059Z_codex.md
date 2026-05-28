# Chain exact-handoff promotion closure

Codex landing: exact promotion now sees the composed entropy-stage repair chain
archive, not only the independent leaf candidates.

## What changed

- Chain stage reports retain runtime-consumption proof path, SHA-256, byte size,
  receiver contract kind, and receiver contract satisfaction.
- `build_repair_family_exact_handoff_plan(...)` accepts a
  `repair_entropy_stage_chain_execution_bundle.v1` and emits an additional
  archive-bound exact-handoff row for each composed chain candidate with
  complete archive/proof custody.
- The bounded autonomous floor loop now executes the chain before exact-ready
  bridge construction, so source queues and blocked exact-ready queues include
  the final chained archive.

## Authority boundary

The bridge remains fail-closed. Chain rows can become exact-ready bridge inputs
only after archive/proof custody is present; dispatch remains blocked until the
existing exact-readiness gate, lane claim, and contest CPU/CUDA axis handoff.

## Verification

- `ruff check --fix` on touched files: passed
- `py_compile` on touched Python modules/tools: passed
- `pytest src/tac/tests/test_repair_family_materializers.py::test_entropy_stage_chain_executor_composes_selected_archive_stages -q`: passed
- `pytest src/tac/tests/test_repair_campaign_materialization_queue.py::test_real_archive_intake_runs_all_families_through_floor_loop -q`: passed
- `pytest src/tac/tests/test_repair_family_materializers.py -q`: 25 passed
- `pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`: 10 passed

## Remaining pressure

The next closure is to let exact/preclaim/harvest failures write back against
the chain row identity directly: family, entropy stage, scope, candidate
archive SHA, and runtime proof SHA. That completes the chain-level demotion and
rebudget loop.
