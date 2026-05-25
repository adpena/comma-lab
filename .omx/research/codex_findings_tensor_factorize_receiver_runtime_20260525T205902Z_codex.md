# Codex Findings - Tensor Factorize Receiver Runtime

UTC: 2026-05-25T20:59:02Z

## Scope

Continuation of the family-agnostic materializer hardening lane after
`tensor_factorize_v1` gained a SHA-bearing runtime-adapter proof gate.

## Finding

The proof gate was correct but incomplete as an executable system: tensor
factorization had no first-class receiver runtime that could consume the
factorized NPZ packet, reconstruct a source-runtime-compatible `.npy` member,
delegate to the original inflate runtime, and emit the generic
`family_agnostic_runtime_consumption_proof_v1` expected by queue postconditions.

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

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/tensor_factorize_receiver.py src/tac/optimization/family_agnostic_materializers.py tools/run_family_agnostic_materializer.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/final_byte_operation_contexts.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py`
- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py -q`

## Remaining Work

Next, run a bounded queue-owned materializer campaign that includes tensor
factorization with a real source runtime path, then feed receiver-positive or
receiver-negative observations back into the acquisition planner rather than
leaving the result as an isolated smoke.
