# Codex Findings: Archive-Section Receiver Contract Handoffs

UTC: 2026-05-26T03:32:24Z

## Verdict

`archive_section_header_elide_v1` and `archive_section_reorder_v1` were registered
planning families, but they still collapsed into generic unsupported
context/work-queue gaps. That lost the useful signal: the optimizer could tell
that archive-section header elision and reorder were promising classes, but not
what receiver/runtime proof must be built next.

This landing wires both families into the final-byte context compiler and
materializer work queue as explicit receiver-contract handoffs. They remain
non-executable and false-authority until a byte-closed adapter, section contract,
and runtime consumption proof exist.

## Implemented

- `src/comma_lab/scheduler/final_byte_operation_contexts.py`
  - Routes `archive_section_header_elide_v1` and `archive_section_reorder_v1`
    through the archive-section context compiler instead of unsupported rows.
  - Carries `archive_path`, `section_manifest`, receiver contract hints, and
    valid `runtime_consumption_proof` paths into context rows.
  - Emits typed missing-context blockers for absent header-elision/order
    contracts and runtime proof.
- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py`
  - Adds typed dispatch blockers for archive-section header-elide and reorder.
  - Emits `archive_section_receiver_contract_work_order.v1` telemetry with
    required fields, next adapter name, and false-authority flags.
  - Prevents these rows from falling back to
    `materializer_work_queue_adapter_missing:*`.
- `src/tac/tests/test_final_byte_operation_contexts.py`
  - Covers missing-contract handoff behavior.
  - Covers the positive proof-signal path where a valid runtime proof is
    preserved into queue telemetry while execution remains blocked.

## Live Bridge Evidence

Generated artifact:
`.omx/research/frontier_rate_attack_feedback_cycle_20260526T_archive_section_contract_handoffs/initial_refresh/operation_materializer_bridge.json`

Observed current rows:

- `archive_section_entropy_recode_v1`: executable local materializer row,
  still non-promotional and false-authority.
- `byte_range_entropy_recode_v1`: blocked on `schema_manifest`,
  `beam_probe_reports`, and `source_runtime_dir`.
- `archive_section_header_elide_v1`: blocked on missing `archive_path`,
  `section_manifest`, `header_elision_contract`, and
  `runtime_consumption_proof`, plus byte-closed adapter requirement.
- `archive_section_reorder_v1`: blocked on missing `archive_path`,
  `section_manifest`, `section_order_contract`, and
  `runtime_consumption_proof`, plus byte-closed adapter requirement.

The feedback-cycle artifact also preserved the rate-budget-to-correction bridge:
`targeted_component_correction_acquisition.json` estimates
`0.0002756656065925789` rate-credit score units, while keeping budget spend
blocked on SegNet/PoseNet component behavior rows, component eval, and exact
auth eval. This is the intended bridge: freed bytes become candidate repair
budget, not score or dispatch authority.

## Eureka JSON Snapshot

The initial cycle had no active local CPU eureka prior. The post-followup JSON
is active and contains:

- `decoder_q_pairset_drop_one`: 26 candidates
- `decoder_q_pairset_drop_two`: 22 candidates
- best conservative gap vs auth frontier: `2.4999999999886224e-06`
- best projected gap vs auth frontier: `-5.000000000143778e-07`

This remains local advisory acquisition prior only. It is useful for selecting
follow-up search geometry, not for promotion, rank/kill, or paid dispatch.

## Verification

- `ruff check` on touched scheduler/test files: pass.
- `pytest src/tac/tests/test_final_byte_operation_contexts.py::test_final_byte_context_compiler_wires_archive_section_contract_handoffs src/tac/tests/test_final_byte_operation_contexts.py::test_final_byte_context_compiler_preserves_archive_section_contract_proof_signal -q`: 2 passed.
- `pytest src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 132 passed.

## Remaining Work

The next executable step is not another advisory row. It is to build byte-closed
receiver adapters for:

1. `archive_section_header_elide_v1`: implicit-constant/header reconstruction
   contract plus runtime consumption proof.
2. `archive_section_reorder_v1`: section order-independence contract plus
   runtime lookup/remap proof.
3. `byte_range_entropy_recode_v1`: schema manifest, beam probe reports, and
   source runtime receiver proof.

Only after those receiver adapters exist should these families spend the
freed-rate budget on targeted SegNet/PoseNet correction work.
