# Codex Findings: Runtime Adapter Identity False Authority

UTC: 2026-05-27T14:29:37Z

## Finding

Queue postconditions and observer revalidation still had one remaining false-authority
surface: a materializer row could claim `runtime_adapter_ready=true` with a
syntactically valid SHA-256 string, but without proving that the live runtime
adapter directory or source-native adapter file matched that identity.

This was especially risky for queue-owned rate attacks because downstream steps
use postconditions to decide whether dependencies are satisfied. A stale runtime
hash could therefore let a chain advance from "JSON says ready" rather than
"the receiver runtime currently on disk is the one that was proven."

## Fix

Landed a canonical runtime-adapter identity verifier:

- live runtime directories must be non-symlink directories;
- `inflate.sh` must exist for directory runtimes;
- the live `tree_sha256(runtime_dir)` must match the declared runtime tree hash;
- source-native file adapters must be non-symlink files with matching SHA-256;
- bare `runtime_adapter_sha256` strings are no longer sufficient identity proof.

The verifier is wired into:

- experiment queue `json_completion_contract`;
- experiment queue `materializer_chain_complete`;
- experiment queue observer materializer and proof revalidation;
- family-agnostic runtime-consumption proof verification;
- materializer-chain harvest proof loading.

## Verification

- `src/tac/tests/test_experiment_queue_observer.py`: 23 passed
- `src/tac/tests/test_materializer_chain_harvest_scheduler.py`: 60 passed
- `src/tac/tests/test_family_agnostic_materializers.py`: 43 passed
- `src/tac/tests/test_byte_shaving_campaign_queue.py`: 89 passed
- Ruff passed on touched implementation and test files
- `py_compile` passed on touched implementation files

## Next Consumer

This unblocks running the queue-owned final rate attack with less fake-readiness
risk. Materializer chains can now be harvested and observed without trusting stale
runtime adapter identity declarations.
