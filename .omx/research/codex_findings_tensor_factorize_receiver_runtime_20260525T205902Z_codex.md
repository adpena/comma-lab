# Codex Findings - Tensor Factorize Receiver Runtime And DFL1 Queue Ordering

UTC: 2026-05-25T20:59:02Z

## Scope

Continuation of the family-agnostic materializer hardening lane after
`tensor_factorize_v1` gained a SHA-bearing runtime-adapter proof gate, plus
the DFL1 queue-ordering blocker found by sidecar review.

## Finding

Two integration gaps were present:

1. The tensor-factorize proof gate was correct but incomplete as an executable
   system: tensor factorization had no first-class receiver runtime that could
   consume the factorized NPZ packet, reconstruct a source-runtime-compatible
   `.npy` member, delegate to the original inflate runtime, and emit the
   generic `family_agnostic_runtime_consumption_proof_v1` expected by queue
   postconditions.
2. The DFL1 materializer execution step could require full-frame inflate
   parity before the parity follow-up step had a chance to create that proof,
   deadlocking the receiver-positive path.

## Landing

Added `tac.optimization.tensor_factorize_receiver` with:

- deterministic NPZ factor packet parsing and shadow-archive reconstruction;
- source-runtime adapter compilation with runtime tree SHA custody;
- generic runtime-consumption proof emission using the existing
  `tensor_factorize_cooperative_receiver_reconstruction_proof.v1` kind;
- smoke execution against `inflate.sh`/`inflate.py` source runtimes.

Wired the runtime path through:

- `tools/run_family_agnostic_materializer.py`;
- `comma_lab.scheduler.byte_shaving_campaign_queue`;
- `comma_lab.scheduler.final_byte_operation_contexts`.

The queue now defers DFL1 full-frame parity postconditions to the dedicated
parity/handoff follow-up unless the context already provides an existing
inflate-parity proof. The materializer step still requires runtime adapter
readiness, but no longer blocks on a proof produced by a later step.

## Authority Boundary

Tensor-factorize receiver evidence and DFL1 parity-followup readiness remain
runtime/queue evidence only. They do not claim score, rank/kill authority,
promotion eligibility, or exact-eval dispatch readiness. Parser
reconstruction, generated runtime metadata, and local smoke behavior remain
below contest-auth authority.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/tensor_factorize_receiver.py src/tac/optimization/family_agnostic_materializers.py tools/run_family_agnostic_materializer.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/final_byte_operation_contexts.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py`
- Focused receiver/queue tests: 6 passed.
- `PYTHONPATH=. .venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`: 145 passed.

## Remaining Work

Next, run a bounded queue-owned materializer campaign that includes tensor
factorization with a real source runtime path, then feed receiver-positive or
receiver-negative observations back into the acquisition planner rather than
leaving the result as an isolated smoke.
