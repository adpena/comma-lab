# Codex Findings - Queue Feedback Replan Local Policy

UTC: 2026-05-24T20:57:17Z

Lane: `codex_queue_feedback_replan_local_policy_20260524`

## Finding

The materializer campaign runner already emitted queue-owned feedback replan
artifacts, but the child queue only executed when the operator passed
`--execute-queue-feedback-replan-followup`. That kept the inverse-steganalysis
feedback loop partly manual: performance observations could produce a paused
child action-functional queue, but the queue/DAG layer did not own the safe
local follow-up policy.

## Patch

- Added `--queue-feedback-replan-followup-policy-local-autopilot` to
  `tools/run_byte_shaving_materializer_campaign.py`.
- Added a strict local-autopolicy guard that only resumes the feedback child
  when the queue is paused, local-first, local-CPU only, command-scoped to
  `tools/build_inverse_steganalysis_action_functional.py`, free of forbidden
  dispatch flags, and recursively free of truthy score/promotion/dispatch
  authority fields.
- Preserved the old explicit runner flag for manual override while recording
  the activation policy in summary and execution payloads.
- Surfaced feedback policy/execution counters in `tools/operator_briefing.py`
  so the autonomous loop is visible from the normal operator summary.
- Tightened family-agnostic materializer queue postconditions so receiver-backed
  archive/member/tensor candidates require runtime-consumption proof and
  receiver-contract satisfaction instead of passing on weak readiness-blocker
  shape alone.
- Added regression coverage for policy execution, state-rationale validation,
  unsafe child queue refusal, operator briefing visibility, and weak receiver
  manifest rejection.

## Authority

This is a local queue-feedback policy only. It does not create a score claim,
promotion path, rank/kill authority, paid dispatch authority, or exact-auth
shortcut. The child output remains an advisory action functional and must still
cross exact readiness and contest auth-eval gates before any score movement is
claimed.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  - `41 passed`
- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py tools/operator_briefing.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_byte_shaving_campaign_queue.py`
  - clean
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  - `102 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py -q`
  - `31 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_staircase_dag.py -q`
  - `24 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_staircase_dag.py -q`
  - `74 passed`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check tools/run_byte_shaving_materializer_campaign.py`
  - clean
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
  - clean
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check src/comma_lab/scheduler/byte_shaving_campaign_queue.py`
  - clean
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check src/tac/tests/test_byte_shaving_campaign_queue.py`
  - clean
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check tools/operator_briefing.py`
  - clean
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check src/tac/tests/test_operator_briefing.py`
  - clean
- `.venv/bin/python tools/lane_maturity.py validate`
  - clean

## Remaining Gap

This closes one manual loop inside the local materializer feedback path. It
does not yet make the full inverse-scorer surface globally optimal: the next
gap is to run real proof-chain queue outputs through this policy continuously
and promote the best calibrated action-functional rows into concrete
materializer/permutation candidates without losing the false-authority boundary.
