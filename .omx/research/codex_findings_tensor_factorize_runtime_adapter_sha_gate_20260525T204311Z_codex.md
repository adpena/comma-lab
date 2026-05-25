# Codex Findings: Tensor Factorize Runtime Adapter SHA Gate

Captured at: 2026-05-25T20:43:11Z

## Finding

`tensor_factorize_v1` had a useful reconstruction proof, but reconstruction
success is not the same as receiver/runtime consumption. A tensor packet can
round-trip through NumPy SVD math while still lacking a byte-closed runtime
adapter identity that the queue can trust.

## Landing

Hardened the tensor-factorize materializer path so:

- reconstruction status is reported separately as `reconstruction_passed`;
- top-level `receiver_contract_satisfied` requires both reconstruction success
  and `runtime_adapter_ready`;
- `runtime_adapter_ready` requires a declared receiver id, receiver adapter
  kind, and a valid runtime adapter SHA such as `runtime_tree_sha256`;
- queue materializer postconditions require
  `receiver_verification.runtime_adapter_sha256` for tensor-factorize rows;
- missing runtime adapter SHA keeps the manifest false-authority and blocked
  even if the contract tries to set `runtime_adapter_ready=true`.

## Authority Boundary

This is still local materializer evidence only. It does not claim score,
promote, rank/kill, or authorize exact-eval dispatch. The gate only prevents
parser/math reconstruction from masquerading as receiver-ready runtime
consumption.

## Verification

- `.venv/bin/ruff check src/tac/optimization/family_agnostic_materializers.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py::test_runtime_consumption_proof_rejects_runtime_adapter_ready_without_sha src/tac/tests/test_family_agnostic_materializers.py::test_tensor_factorize_materializer_emits_cooperative_receiver_proof src/tac/tests/test_family_agnostic_materializers.py::test_tensor_factorize_materializer_accepts_runtime_adapter_ready_contract src/tac/tests/test_family_agnostic_materializers.py::test_tensor_factorize_materializer_runtime_adapter_ready_requires_sha src/tac/tests/test_family_agnostic_materializers.py::test_tensor_factorize_materializer_cli_auto_writes_runtime_proof src/tac/tests/test_byte_shaving_campaign_queue.py::test_tensor_factorize_postconditions_require_runtime_adapter_ready -q`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`

Results: focused 6-test receiver gate passed; widened materializer/queue slice
passed with 122 tests.
