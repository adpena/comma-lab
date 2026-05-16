# PacketIR Section Hash-Domain Custody Hardening

Date: 2026-05-16
Owner: Codex
Status: landed

## Bug Class

PacketIR runtime-consumption closure could compare bare `sha256` section
identities without proving the hash domain. For PR106/R2 Format0D and
headerless aliases, this allowed a future proof to accidentally bind a
whole-payload, runtime-digest, or reconstructed-stream hash as if it were the
emitted member payload section bytes.

## Fix

- `pr106_sidecar_consumed_byte_proof()` now emits
  `hash_domain=pr106_sidecar_packet_ir_emitted_member_payload_section_bytes_v1`
  on every section row.
- `prove_pr106_sidecar_runtime_decode_consumption()` preserves that domain in
  `runtime_consumed_score_affecting_section_identities`, alongside provenance
  fields naming the runtime mutation-probe evidence.
- `packetir_exact_closure` rejects Format0D and headerless-alias closure when
  candidate or runtime section identities do not carry the PacketIR emitted
  section-byte hash domain.

## Verification

- `.venv/bin/python -m ruff check src/tac/packet_compiler/pr106_sidecar_packet.py src/tac/packet_compiler/pr106_runtime_consumption.py src/tac/packetir_exact_closure.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py`
- `.venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_materialize_pr106_latent_score_table_candidate.py src/tac/tests/test_research_basis.py -q`

## Reactivation Criteria

Any new PacketIR grammar or runtime-consumption proof that introduces another
alias between emitted bytes and reconstructed/runtime bytes must define a
domain-separated identity field and a closure test that rejects a mismatched
domain.
