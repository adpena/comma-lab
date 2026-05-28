# Codex Findings: Archive Entropy Substrate Gap Routing

Timestamp: 2026-05-28T18:24:38Z

## Verdict

Archive entropy substrate coverage is now solver input, not a side report.
Range/ANS/FEC/header/selector coverage emitted by byte-transform execution is
carried into repair-family stack rows, stack-search plan metadata, posterior
learning feature vectors, and the autonomous floor-loop summary.

## What Changed

- Stack rows now preserve `archive_entropy_substrate_coverage`,
  `archive_entropy_substrate_materialized_substrates`, and
  `archive_entropy_substrate_blockers`.
- Missing entropy substrates now participate in budget routing through the
  typed action `materialize_missing_archive_entropy_substrate_variant` with
  blocker `archive_entropy_substrate_materializer_gap`.
- Negative posterior demotion remains higher semantic authority than missing
  substrate expansion: known-bad families demote before range/ANS expansion.
- Floor-loop summaries expose aggregate entropy-substrate blocker count and
  blocker IDs, so autonomous runners can rebudget instead of burying the gap.

## Authority Contract

No score authority is granted. Range/ANS gaps are typed materializer blockers
until a byte-closed archive, receiver proof, and exact CPU/CUDA handoff path
exist. MLX-local and archive-analysis outputs remain advisory.

## Verification

- `.venv/bin/ruff check --fix src/tac/optimization/repair_family_stack_search.py tools/run_repair_campaign_autonomous_floor_loop.py src/tac/tests/test_repair_campaign_materialization_queue.py src/tac/tests/test_repair_family_materializers.py`
- `.venv/bin/python -m py_compile src/tac/optimization/repair_family_stack_search.py tools/run_repair_campaign_autonomous_floor_loop.py`
- `.venv/bin/pytest src/tac/tests/test_repair_family_materializers.py -q`
- `.venv/bin/pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`
