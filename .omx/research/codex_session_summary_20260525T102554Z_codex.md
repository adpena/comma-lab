# Codex Session Summary - Family Materializer Empirical Sweeps - 2026-05-25T10:25:54Z

## Scope

Harden and land the in-progress family-agnostic materializer empirical sweep
queue work so automated final-byte operations can compare multiple archive
candidates in one queue-owned run instead of staying at single leaf
materializations.

## Landed

- `tools/run_family_agnostic_materializer_sweep.py` now supports
  `archive_section_entropy_recode_v1` in addition to packet-member recompress,
  ZIP-header elide, and tensor-factorize targets.
- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py` can wrap explicit
  sweep contexts into executable `experiment_queue.v1` rows that call
  `tools/run_family_agnostic_materializer_sweep.py`.
- Sweep rows emit strict postconditions for:
  - sweep JSON schema and target kind,
  - false authority on the aggregate sweep JSON,
  - false authority on every JSONL observation row.
- Sweep activation is explicit: only sweep-prefixed context keys trigger the
  sweep path, so ordinary single-materializer contexts with generic archive
  lists remain on the single materializer tool.
- Queue telemetry carries input artifacts, pullback artifacts, and a
  `family_agnostic_materializer_sweep_command_contract.v1` contract with all
  score/dispatch authority false.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py tools/run_family_agnostic_materializer_sweep.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializer_sweep.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_work_queue_wraps_archive_section_entropy_recode_adapter src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_work_queue_wraps_family_agnostic_empirical_sweep src/tac/tests/test_family_agnostic_materializer_sweep.py -q --durations=20`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_final_byte_operation_contexts.py -q --durations=30`

Latest broad result: `135 passed in 3.21s`.

Extended materializer/observer/runner/acquisition/queue/sweep slice:
`241 passed in 7.93s`.

## Remaining Work

- Feed sweep observation JSONL into the action-functional feedback path so
  archive-class demotion and receiver-repair recommendations automatically
  update the next acquisition plan.
- Add a runner-level smoke that executes a tiny sweep queue through
  `tools/run_byte_shaving_materializer_campaign.py` once the next campaign
  runner tranche resumes.
