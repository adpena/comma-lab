# Codex Findings: Experiment Queue Receiver Proof Guard

UTC: 2026-05-27T21:49:56Z

## Finding

`experiment_queue` postconditions could validate false-authority JSON and JSONL
rows while still allowing a row to claim receiver/runtime custody from fields
such as `receiver_contract_satisfied` or `proof_present` without proving that a
live receiver proof file existed, was bound to the candidate archive, and itself
remained false-authority.

## Fix Landed

- Added receiver proof extraction for JSON and JSONL postcondition payloads.
- Required receiver-runtime custody claims to carry a live JSON proof artifact.
- Bound proof artifacts back to the candidate archive SHA when candidate archive
  custody is present.
- Rejected proof files that are missing, symlinks, non-JSON, blocked, failed,
  authority-leaking, or hash/byte mismatched.
- Preserved runtime adapter identity fail-closed behavior when payloads or
  proofs claim `runtime_adapter_ready`.

## Verification

- `ruff check src/comma_lab/scheduler/experiment_queue.py
  src/tac/tests/test_experiment_queue.py` passed.
- Targeted queue tests passed: `17 passed, 61 deselected`.
- Full experiment queue test file passed: `78 passed in 15.60s`.
- CLI smoke ran `tools/experiment_queue.py init`, bounded `run-worker`, and
  `observe` against a generated queue whose step emitted a candidate archive,
  receiver proof, and advisory JSON. The step succeeded and observer output
  remained healthy with false score/dispatch authority.

## Next Integration Edge

Use this guard as the enforcement layer for final-rate and portfolio-coverage
autoloops: a queue step that claims receiver/runtime custody should now fail
its postconditions unless it has real proof custody. The next gap is to make
portfolio coverage a required preflight for the final-rate autoloop campaign
mode rather than an inspectable artifact only.
