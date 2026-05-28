# Codex Findings: Entropy-Stage Chain Executor

Date: 2026-05-28T17:47:34Z
Agent: codex

## Landed Slice

- Added a concrete entropy-stage chain executor that consumes selected repair-family execution reports plus the entropy-stage work-order bundle, then replays each family byte transform sequentially against the archive emitted by the previous stage.
- Added an operator CLI for the chain executor.
- Wired the autonomous repair floor loop to emit `repair_family_entropy_stage_chain_execution_bundle.json` by default.
- The floor-loop summary now exposes chain count, materialized composed-candidate count, and runtime-proof-ready count.

## Authority Boundary

The composed archive candidate is still encoder-side and false-authority. It can feed exact-ready bridge preparation, but it cannot claim score, promote, rank/kill, spend budget, or dispatch exact eval until contest CPU/CUDA custody and lane-claim gates pass.

## Verification

- `ruff check` on touched files: passed.
- `py_compile` on new/edited tools and module: passed.
- `pytest src/tac/tests/test_repair_family_materializers.py::test_entropy_stage_chain_executor_composes_selected_archive_stages -q`: 1 passed.
- `pytest src/tac/tests/test_repair_family_materializers.py -q`: 25 passed.
- `pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`: 10 passed.
- `pytest src/tac/tests/test_repair_autonomous_multi_archive_runner.py -q`: 1 passed.
- `git diff --check`: passed.
- Recursive adversarial review bundle `cbb2c3fbc3b2b81a`: 3 successive clean passes, sealed.

## Review Notes

- Real archive custody now produces a composed archive-bound candidate through the floor loop.
- Synthetic queue rows without source archive custody remain fail-closed and produce zero chain candidates.
- MLX remains advisory-only; exact CPU/CUDA authority is still the only score path.
