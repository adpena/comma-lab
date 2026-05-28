# Exact failure posterior loop closure

Codex landing: exact/preclaim/receiver/runtime failures now write back into the
repair posterior with typed archive/proof/content identity.

## What changed

- Exact-ready bridge rows now carry `failure_rebudgeting_identity` with
  candidate id, chain id, family/stage/scope identity, source/candidate archive
  SHA, runtime proof SHA, and runtime content-tree SHA.
- Preclaim gates preserve that identity, so provider-claim failures do not lose
  their archive/proof context.
- Failure rebudgeting updates now expose the responsible failure surface:
  preclaim, receiver/runtime proof, runtime content tree, archive custody, or
  exact-axis handoff.
- `build_repair_family_stack_learning_signal_report(...)` now appends
  posterior-consumable exact-failure learning signals in addition to stack-row
  and entropy-stage-chain signals.

## Authority boundary

These rows only update acquisition, demotion, and rebudget routing. They still
cannot claim score, promote, dispatch, rank, kill, or spend budget. Contest
CPU/CUDA exact-axis authority remains the only score authority.

## Verification

- `ruff check --fix` on touched files: passed
- `py_compile` on touched Python modules/tools: passed
- `pytest src/tac/tests/test_repair_family_materializers.py::test_entropy_stage_chain_executor_composes_selected_archive_stages -q`: passed
- `pytest src/tac/tests/test_repair_campaign_materialization_queue.py::test_real_archive_intake_runs_all_families_through_floor_loop -q`: passed
- `pytest src/tac/tests/test_repair_family_materializers.py -q`: 25 passed
- `pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`: 10 passed

## Review notes

This closes the no-signal-loss path for local chain execution through exact
bridge failure: archive-bound candidates, composed chains, preclaim failures,
and runtime/custody blockers all flow into the same append-only posterior.
The next pressure is broader substrate coverage under this same loop:
FEC variants, header/member ordering, selector streams, range/ANS/Huffman, and
pre-coder distribution shaping.
