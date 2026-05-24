# Codex Findings: Packet-Member ZIP Header Elide Materializer

UTC: 2026-05-24T22:56:53Z
Lane: `codex_packet_member_zip_header_elide_materializer_20260524`

## Finding

The inverse-steganalysis final-byte queue had a registered
`packet_member_zip_header_elide_v1` target, but it was planning-only. That left
PacketIR/high-level operation-set rows able to name deterministic ZIP/header
savings without a byte-closed local proof chain.

## Landing

Implemented an executable family-agnostic materializer for
`packet_member_zip_header_elide_v1`.

- Preserves selected ZIP member payload bytes and member name.
- Strips deterministic elidable ZIP metadata: member extra field, member
  comment, and archive comment by default.
- Emits a `family_agnostic_runtime_consumption_proof_v1` payload-identity proof.
- Stays false-authority: no score claim, promotion, rank/kill, dispatch, or
  exact-eval readiness.
- Wires registry, final-byte context compiler, materializer work queue command
  builder, postconditions, harvestable schema list, and CLI.

## Verification

Focused verification:

```bash
.venv/bin/ruff check src/tac/optimization/family_agnostic_materializers.py tools/run_family_agnostic_materializer.py src/comma_lab/scheduler/byte_shaving_materializer_registry.py src/comma_lab/scheduler/final_byte_operation_contexts.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py
PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_candidate_queue.py -q
```

Result: ruff clean; `160 passed`.

## Remaining Integration

This closes one execution-breadth gap. Remaining registered planning-only
families should be converted similarly only where the receiver contract can be
made byte-closed and testable:

- `packet_member_reorder_v1`
- `archive_section_header_elide_v1`
- `archive_section_reorder_v1`
- tensor quantize/prune/shared-codebook targets

Do not treat this local proof-chain output as score or promotion authority.
Exact auth eval remains required for score movement.
