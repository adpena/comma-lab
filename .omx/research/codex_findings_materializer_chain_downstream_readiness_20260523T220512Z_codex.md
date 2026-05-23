# Codex Findings: Materializer Chain Downstream Readiness

- timestamp_utc: 2026-05-23T22:05:12Z
- agent: codex
- lane_id: codex_serialized_archive_economics_guard_20260523
- research_only: false

## Finding

`materializer_chain_complete` conflated two different states:

- a chain is complete and has emitted byte-closed candidate custody;
- a candidate has cleared every downstream exact-ready blocker.

That was too strict for the current queue/DAG boundary. A materializer chain can
be complete while still correctly carrying downstream blockers such as
`candidate_inflate_output_parity_missing` or
`exact_auth_eval_required_before_score_claim`.

## Fix Landed

`materializer_chain_complete` now allows downstream `readiness_blockers` by
default. Callers that need the old strict behavior can opt in with
`forbid_readiness_blockers=true`.

This keeps materializer completion about artifact custody and lets exact-ready
promotion enforce dispatch authority separately.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  - 79 passed
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue.py src/tac/tests/test_experiment_queue.py`
  - passed

## Remaining Wire-In

The next helper should consume completed chain manifests, preserve downstream
blockers as typed exact-ready inputs, and only clear them through explicit
parity, local advisory, or auth-eval gates.
