# Codex Findings: Inverse Action Queue Feedback Unit Matching

- Timestamp: 2026-05-24T20:15:27Z
- Lane: `codex_inverse_action_queue_feedback_unit_match_20260524`
- Scope: queue performance feedback to inverse-steganalysis action functional.

## Finding

Queue performance summaries already expose materializer telemetry with
`source_unit_ids` and `source_selection_ids`, but inverse-action calibration
matched observations only by `candidate_id`. Materializer queues often use a
PacketIR/work identifier rather than the parent action-cell candidate id, so
timing and artifact-size feedback could miss the exact cell that produced the
work. That left useful denominator calibration trapped in the queue artifact.

## Landed

- Queue performance observations now preserve `candidate_ids`, `work_ids`,
  `backlog_keys`, `source_unit_ids`, and `source_selection_ids`.
- Action-functional observation selection now matches an atom by explicit
  source/provenance/compiler identifiers as well as by candidate id.
- Compiler-derived materializer unit ids such as
  `inverse_action_<atom_id>_<unit_id>_<index>` are recognized, so feedback from
  PacketIR/materializer rows can calibrate the originating action cell.
- Added regression coverage proving a materializer observation with a
  nonmatching `candidate_id` still updates the intended action cell via
  `source_unit_ids` and `source_selection_ids`.

## Authority Boundary

This remains denominator calibration for planning only. Queue performance rows
carry no scorer delta, no score claim, no promotion authority, and no exact eval
dispatch authority.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q`
  - `40 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py`
  - `All checks passed!`

## Next

- Let materializer campaign follow-up queues run the generated feedback replan
  step after local proof chains complete, then inspect whether the action
  functional priorities shift by the measured per-operation denominators.
