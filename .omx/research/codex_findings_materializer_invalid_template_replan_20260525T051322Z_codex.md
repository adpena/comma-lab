# Codex Findings: Materializer Invalid-Template Replan

UTC: 2026-05-25T05:13:22Z
Lane: codex_materializer_invalid_template_replan_20260525

## Verdict

The live high-level byte-shaving campaign failure is not a timeout or a
parallelism ceiling. It is a deterministic stale-source failure: the queued
inverse-scorer cell materializer points at
`.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/template.zip`,
which is not a strict single-member ZIP archive.

Current queue-build code already refuses new invalid templates before producing
executable rows. The remaining gap was recovery: the live campaign was built
before that guard, and the generic queue recovery path would have emitted a
rewind queue that only replays the same invalid input.

## Landed Changes

- `tools/materialize_byte_shaving_queue_recovery.py` now classifies failed
  inverse-scorer materializer source commands, validates their
  `--candidate-archive-template` with the strict single-member ZIP reader, and
  emits `queue_source_failure_diagnostics.json`.
- Non-rewindable source failures now block recovery-queue emission with a typed
  `source_failure_non_rewindable:candidate_archive_template_invalid_strict_single_member_zip:*`
  blocker.
- `tools/operator_briefing.py` consumes the source-failure diagnostics,
  reports non-rewindable counts, and refuses to recommend stale recovery queue
  initialization when the source diagnostic says the rewind would repeat the
  failure.
- The previously emitted stale recovery queue artifact was removed from the
  tracked campaign artifact set and replaced with the typed diagnostic.

## Live Artifact

- `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3/queue_source_failure_diagnostics.json`
  reports `diagnostic_count=1`, `non_rewindable_source_failure_count=1`,
  `recovery_queue_execution_recommended=false`, and the invalid-template
  blocker.
- `tools/operator_briefing.py --json` now reports
  `queue_observation_recovery_queue_count=0`,
  `source_failure_non_rewindable_count=1`, and next command:
  `.venv/bin/python tools/experiment_queue.py --queue .omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3/materializer_execution_queue.json --state .omx/state/experiment_queue_high_level_byte_shaving_runner_smoke_20260524T050723Z_campaign3.sqlite observe --tail-lines 20`

## Verification

- `.venv/bin/python -m py_compile tools/materialize_byte_shaving_queue_recovery.py tools/operator_briefing.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m ruff check tools/materialize_byte_shaving_queue_recovery.py tools/operator_briefing.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py -q` -> 39 passed
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_inverse_action_cells_refuse_invalid_candidate_archive_template -q` -> 1 passed

## Remaining Work

The next score-moving action is to repair or regenerate the materializer
contexts with a valid strict single-member candidate template, then rebuild the
source materializer queue so the inverse-scorer action cells can actually emit
byte-closed candidates. Do not execute the old recovery queue; it has been
retired as non-rewindable.
