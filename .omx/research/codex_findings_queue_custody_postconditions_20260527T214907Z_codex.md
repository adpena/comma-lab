# Codex Findings: Queue Custody Postconditions

## Summary

Codex hardened the experiment queue postcondition evaluator so receiver/runtime
custody claims cannot pass through `json_false_authority`, `jsonl_false_authority`,
or `json_completion_contract` by carrying labels such as `proof_present` or
`receiver_contract_satisfied` alone.

The queue now revalidates custody-bearing postcondition payloads at the same
authority boundary where the worker decides whether a step succeeded:

- candidate archive record must point to a live, non-symlinked file with matching
  bytes and SHA-256;
- receiver contract satisfaction must be independently present on the payload or
  receiver verification object;
- a JSON proof path must exist and carry a success flag such as `passed`,
  `runtime_consumption_proof_passed`, or full-frame parity satisfaction;
- the proof must bind back to the candidate archive SHA when that SHA is known;
- proof files may not smuggle truthy score, promotion, rank/kill, dispatch, or
  exact-eval authority fields;
- runtime-adapter-ready claims must pass runtime adapter identity validation,
  including expected runtime-tree identity when a runtime tree is claimed.

## Why It Matters

The observer already revalidated many of these surfaces, but the queue worker
itself could still classify a custody-flavored postcondition as passing before
observer health caught the issue. That left a small but real false-authority
propagation gap: a row could say it had receiver closure without making the
artifact itself prove archive/runtime/receiver custody.

This landing moves that check into the queue's decisive postcondition path.
Planning-only rows remain allowed, but custody rows now have to carry the actual
evidence they are asking downstream consumers to trust.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue.py src/tac/tests/test_experiment_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue_observer.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
- `.venv/bin/python tools/review_gate_hook.py`
- `.venv/bin/python tools/lane_maturity.py validate`

## Commit

- `754a1ccff Harden queue custody postconditions`

## Remaining Work

The next high-EV tranche is to make repair/action-functional rows materialize
the same proof-backed custody by default: every optimizer-selected repair spend
should emit a byte-closed candidate archive, archive-bound proof JSON, MLX-local
advisory calibration record, and exact-eval refusal or handoff packet from the
same queue-owned ledger row.
