# Codex Findings: Receiver-Gated Dynamic Sparse Feedback Runner

- UTC: 2026-05-25T11:59:57Z
- Lane: `codex_receiver_gated_dynamic_sparse_feedback_runner_20260525`
- Status: integrated locally; planning-only; no score or dispatch authority

## Finding

The materializer campaign runner already harvests queue observation, local
runtime/cache identity, and receiver feedback. The missing bridge was a
standard compiler-hint artifact that lets receiver-positive materializer rows
feed the higher-level dynamic sparse action surface automatically while keeping
receiver-negative rows as repair/demotion signal.

## Landed Integration

- Added a runner-owned `dynamic_sparse_feedback_compiler_hint.json` artifact
  generated from `queue_observation.json` via
  `tools/build_dynamic_sparse_gate_compiler_hint.py`.
- Gated the default compiler channel to `receiver_proof`, so byte savings
  without receiver proof do not become positive compiler actuation.
- Added `dynamic_sparse_feedback_compiler_hint_status.json` with selected
  operation counts, receiver-positive counts, receiver-negative counts, command
  provenance, blockers, and false-authority fields.
- Threaded the hint path, status path, emitted flag, blocker list, and receiver
  feedback counts into the queue feedback replan request, child queue metadata,
  canonical response-update placeholder, and campaign run summary.
- Added regressions proving mixed receiver-positive/receiver-negative queue
  observations select only the receiver-positive operation, and
  receiver-negative-only observations emit blockers instead of a compiler hint.

## Authority Boundary

The dynamic sparse feedback hint is an inverse-action compiler planning
handoff only. It does not claim score, promotion, rank/kill, exact readiness,
paid dispatch, or contest authority. Receiver-negative rows remain explicit
blocker signal for receiver repair before bucket refill.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py --no-cache`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_emits_receiver_gated_dynamic_sparse_hint src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_blocks_dynamic_sparse_hint_without_receiver -q --durations=10 --durations-min=0.01`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_dynamic_sparse_gate_oracle.py -q --durations=30 --durations-min=0.01`
- `git diff --check`
- `.venv/bin/python tools/review_tracker.py policy-check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`

## Uncommitted Neighbor Work

Concurrent receiver-schema hardening edits appeared in
`src/tac/optimization/family_agnostic_materializers.py` and
`src/tac/tests/test_family_agnostic_materializers.py` during verification. They
are receiver-adjacent but not required for this runner bridge, so this lane
keeps them out of its commit unless the operator or owning lane routes them in.
