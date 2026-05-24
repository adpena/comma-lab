# Codex Findings: PacketIR / Final-Byte Integration Boundary

UTC: 2026-05-24T16:09:15Z
Author: Codex
Status: implementation finding + integration guard

## Verdict

PacketIR should be the canonical compiler/IR boundary for concrete byte-shaped
operations, not the owner of inverse-steganalysis planning authority.

Use PacketIR now where a row already has byte-closed shape: archive sections,
packet members, tensors, sparse residual streams, PR101/FEC6 packets, and
PR106/HNeRV-style section grammars. Do not force scorer-inverse cells,
water-bucket acquisition rows, or action-functionals into PacketIR before a real
operation-set compiler has lowered them into concrete byte operations.

Subagent signal: Russell (`019e5ab9-ab42-7e90-8ad9-1587cbe7a48d`) completed a
read-only PacketIR integration audit. Its key recommendation was to integrate
PacketIR now only at concrete byte-operation boundaries and to normalize
`section_manifest` / `parser_section_manifest` / `packet_ir_manifest` vocabulary
before materializer dispatch. The agent was closed after harvest; no subagent
signal was left orphaned.

## Changes Landed

- `final_byte_operation_contexts` now normalizes `section_manifest`,
  `parser_section_manifest`, and `packet_ir_manifest` into the canonical
  `section_manifest` field expected by archive-section materializer work.
- The original manifest source key is preserved in the context so PacketIR
  provenance is not erased.
- The PacketIR operation-set lowering contract is centralized in
  `tac.packet_compiler.deterministic_compiler.packetir_operation_set_bridge_contract`;
  final-byte scheduler rows wrap that contract instead of duplicating compiler
  order/proof vocabularies.
- `selection_kind=operation_set` scheduler rows now fail closed unless the plan
  contains a matching PacketIR operation set with the expected schema, source id,
  sequence hash, compiler contract, required order, required proofs, and no
  truthy authority fields.
- Supported archive-section, packet-member, and tensor final-byte contexts now
  carry the same compiler-owned PacketIR bridge contract, not only unsupported
  rows.
- Family-agnostic runtime-consumption proofs now reject canonical false-authority
  leakage such as `score_claim`, `ready_for_exact_eval_dispatch`,
  `dispatch_attempted`, and `promotable`.
- Unsupported high-level inverse-steganalysis operation-set rows remain blocked
  and carry a PacketIR compiler bridge hint rather than receiving fake executable
  authority.
- Rust packet-compiler docs were corrected from scaffold-only/mixed signals to
  the current mixed-native state: many primitives are native and golden-vector
  protected, while remaining scaffolds stay explicit.

## Refactor Map

1. Define `packet_ir_operation_set_v1` as the durable lowered form from
   inverse-steganalysis/water-bucket portfolios into concrete byte operations.
2. Implement the compiler:
   inverse action portfolio -> operation set -> PacketIR-backed materializer
   contexts.
3. Keep queue/DAG custody and dispatch authority in `comma_lab.scheduler`.
4. Keep reusable byte grammars, deterministic compiler helpers, and Rust hot
   paths in `tac.packet_compiler` / `runtime-rs`.
5. Gate Rust lowering by Python golden vectors and exact closure proofs; Rust
   acceleration is never score or promotion authority.
6. Normalize remaining vocabulary across `section_manifest`,
   `parser_section_manifest`, `packet_ir_manifest`, and packet-member manifests
   so adapters do not drift.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py -q`
  -> 9 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_deterministic_compiler.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_cooperative_receiver_integration.py -q`
  -> 51 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_compile_dqs1_byte_shaving_plan_emits_action_summary_and_blocks_unknown_ops src/tac/tests/test_byte_shaving_campaign_queue.py::test_packet_ir_operation_set_lowers_to_materializer_backlog_rows src/tac/tests/test_byte_shaving_campaign_queue.py::test_operation_set_execution_requires_matching_packet_ir_handoff src/tac/tests/test_byte_shaving_campaign_queue.py::test_operation_set_execution_validates_packet_ir_contract src/tac/tests/test_byte_shaving_campaign_queue.py::test_packet_ir_operation_set_lowering_rejects_authority_and_bad_sequence -q`
  -> 5 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_deterministic_compiler.py::test_packetir_operation_set_bridge_contract_is_fail_closed src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_cooperative_receiver_integration.py src/tac/tests/test_optimizer_exact_readiness.py::test_promotes_family_agnostic_candidate_with_receiver_proof src/tac/tests/test_optimizer_exact_readiness.py::test_family_agnostic_runtime_proof_fails_closed_on_invalid_evidence -q`
  -> 22 passed.
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/final_byte_operation_contexts.py src/tac/tests/test_final_byte_operation_contexts.py`
  -> passed.
- `cargo test -p tac-packet-compiler` in `runtime-rs`
  -> 130 unit tests, 41 golden-vector parity tests, doctest pass/ignored as
  expected.
- `cargo doc -p tac-packet-compiler --no-deps` in `runtime-rs`
  -> passed after eliminating rustdoc link warnings in touched/native compiler
  docs.

## Remaining Gap

The frontier-moving gap is still the actual compiler from high-level
inverse-steganalysis acquisition portfolios into byte-closed PacketIR/materializer
operations. Until that compiler exists, the system must fail closed with a
bridge hint instead of pretending the planner signal is executable.
