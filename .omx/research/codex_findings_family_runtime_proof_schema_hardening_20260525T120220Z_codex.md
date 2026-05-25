# Codex Findings: Family Runtime Proof Schema Hardening

- UTC: 2026-05-25T12:02:20Z
- Lane: `codex_family_runtime_consumption_proof_schema_hardening_20260525`
- Status: integrated locally; receiver authority hardening; no score claim

## Finding

Family-agnostic runtime consumption proofs need the same fail-closed schema
guard as proof kind, receiver contract, target kind, materializer id, archive
SHA, and member SHA checks. Without the explicit schema check, a malformed or
wrong-family proof object could still be evaluated through later metadata
checks, making receiver failures less diagnosable.

## Landed Integration

- Introduced `RUNTIME_CONSUMPTION_PROOF_SCHEMA` as the canonical family proof
  schema constant used by proof emitters and verifiers.
- Made `verify_runtime_consumption_proof(...)` append
  `runtime_consumption_proof_schema_mismatch` when the supplied proof schema is
  not `family_agnostic_runtime_consumption_proof_v1`.
- Centralized packet-member ZIP header-elide proof kind and receiver contract
  constants.
- Required header-elide materialization to validate proof kind, receiver
  contract kind, target kind, and materializer id before treating receiver proof
  as satisfied.
- Added regressions for wrong proof schema on packet-member recompress and
  wrong proof metadata on packet-member ZIP header elide.

## Authority Boundary

This only hardens local receiver proof classification. It does not grant score,
promotion, rank/kill, exact-readiness, paid dispatch, or contest authority.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/family_agnostic_materializers.py src/tac/tests/test_family_agnostic_materializers.py --no-cache`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py::test_packet_member_recompress_materializer_rejects_wrong_proof_schema src/tac/tests/test_family_agnostic_materializers.py::test_packet_member_zip_header_elide_materializer_rejects_wrong_proof_metadata src/tac/tests/test_family_agnostic_materializers.py::test_packet_member_zip_header_elide_materializer_emits_runtime_proof -q --durations=10 --durations-min=0.01`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py -q --durations=30 --durations-min=0.01`
- `.venv/bin/python tools/review_tracker.py policy-check src/tac/optimization/family_agnostic_materializers.py src/tac/tests/test_family_agnostic_materializers.py`
- `git diff --check`
