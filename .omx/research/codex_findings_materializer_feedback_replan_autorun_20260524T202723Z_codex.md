# Codex Findings: Materializer Feedback Replan Autorun

UTC: 2026-05-24T20:27:23Z
Lane: codex_materializer_feedback_replan_autorun_20260524

## Finding

The byte-shaving materializer campaign runner already emitted a safe
`queue_feedback_replan_followup_queue.json`, but the child queue stopped as a
paused artifact. That left the inverse-steganalysis feedback loop one manual
step short of compounding: parent queue performance could generate the next
action-functional command, but the runner did not execute that local child queue
unless an operator manually resumed it.

## Landing

- Added `--execute-queue-feedback-replan-followup` to
  `tools/run_byte_shaving_materializer_campaign.py`.
- The flag keeps the generated child queue paused by default, then explicitly
  initializes a run-scoped child state, records a `control running` event, runs a
  bounded local-only worker, observes the child queue, captures child
  performance, and records the generated feedback action-functional path in the
  campaign summary.
- The child queue remains restricted to
  `tools/build_inverse_steganalysis_action_functional.py`, retains forbidden
  exact/provider/submit flags, and enforces false-authority JSON postconditions.
- The campaign summary now records
  `queue_feedback_replan_followup_execution`, success/blockers, child state
  path, and feedback action-functional path.

## Authority

This is local queue execution only. It does not grant score, promotion,
rank/kill, exact-eval dispatch, or paid/cloud authority. Queue performance
continues to act as denominator/calibration signal for the next action
functional.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q`
- `.venv/bin/python tools/review_tracker.py policy-check tools/run_byte_shaving_materializer_campaign.py`
- `.venv/bin/python tools/review_tracker.py policy-check src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`

## Remaining Gap

Pasteur's read-only audit confirmed the larger gap: inverse-steganalysis is now
real along the action-functional -> campaign -> materializer -> feedback path,
but many high-level cells still require deterministic operation-set compilers
before they become final-rate operations. The next production target is a
broader `inverse_action_operation_set_compiler.v1` surface in `tac`, with
`comma_lab` materializer-context binding and MLX/source-video substrate training
as downstream consumers.
