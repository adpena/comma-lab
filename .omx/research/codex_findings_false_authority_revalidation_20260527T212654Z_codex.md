# Codex findings: false-authority revalidation hardening

Date: 2026-05-27T21:26:54Z
Agent: Codex
research_only: false

## Findings

1. Final-rate child queue observer revalidation was identity-complete but not health-complete.
   - It checked observer schema, queue id, queue hash, and read-only mode.
   - It did not fail closed when the observer payload itself reported `healthy=false`, nonzero `blocker_count`, observer blockers, or succeeded artifact postcondition failures.
   - This could allow a queue-owned campaign run to preserve a valid observer artifact while losing the observer's negative signal.

2. Materializer-chain harvest had late-bound receiver proof paths, but those paths needed to become executable evidence after runtime context overlay.
   - Work-queue runtime context can provide `runtime_consumption_proof_path` after a chain manifest has been adapted into optimizer-row shape.
   - A carried proof path or `proof_present` flag is not sufficient; the harvest row must re-open the proof, verify archive binding, verify proof success, and bind it to live runtime identity before clearing receiver/runtime blockers.

## Landed behavior

- `frontier_final_rate_attack_autoloop` now treats child queue observation health as part of observer revalidation.
  - `observer_queue_unhealthy`, `observer_blocker:*`, `observer_blocker_count_nonzero`, and `observer_artifact_postcondition_failures_present` now make the child run invalid and stop progress credit.
  - Revalidation records now preserve `observed_healthy`, `observed_blockers`, `observed_blocker_count`, and `observed_artifact_failure_count`.
- `materializer_chain_harvest` now revalidates late-bound `runtime_consumption_proof_path` rows from runtime context.
  - Clean proofs can clear stale receiver/runtime blockers while staying `ready_for_exact_eval_dispatch=false`.
  - Missing, stale, mismatched, or unsupported proofs remain blockers and keep the row non-promotional.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py src/comma_lab/scheduler/materializer_chain_harvest.py src/tac/optimizer/materializer_chain_harvest.py src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_materializer_chain_harvest_scheduler.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q` passed with 32 tests.
- `.venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py -q` passed with 65 tests.
- Bounded queue-owned final-rate smoke `frontier_final_rate_attack_false_authority_smoke_20260527Tlocal` completed with:
  - top-level `failed_command_count=0`
  - post-feedback child `failed_command_count=0`
  - post-feedback child `failed_queue_count=0`
  - `observer_revalidation_failed_count=0`
  - `stalled_queue_count=0`
  - selected child queue `operation_chain_compiler_queue`, `queue_healthy=true`, `observer_revalidation_valid=true`, `steps_started=2`.

## Authority boundary

The smoke and this memo are infrastructure evidence only. They do not claim a score, rank, promotion, exact readiness, or dispatch authority. Frontier score remains unchanged until an exact auth-axis result lands through the canonical dispatch and review path.
