# PacketIR Format0D Section-Consumption Review

Date: 2026-05-16
Owner: Codex
Status: verified-current-main

## Review Trigger

Adversarial review flagged a possible Format0D false-authority class:
`runtime_all_score_affecting_sections_consumed` could be true if the runtime
only consumed the extra PR101 sidecar stream while ignoring the base Format0C
section.

## Current Main Finding

Current `main` already contains the structural fix:

- `runtime_sidecar_section_consumption_probes(...)` independently probes
  `base_format0c_sidecar_payload` and `extra_pr101_ranked_no_op_payload`.
- `runtime_sidecar_decode_consumption_claim` requires both section probes.
- `runtime_consumed_score_affecting_sections` records each Format0D section
  independently.
- `test_pr106_runtime_consumption_rejects_format0d_without_base_section_probe`
  monkeypatches the base probe to false and verifies the all-sections claim
  fails closed.

The follow-up offset-identity concern is also already covered:

- Format0D closure compares candidate/runtime SHA-256, hash domain, byte
  length, and section offset.
- `test_packetir_exact_closure_rejects_format0d_runtime_offset_mismatch`
  shifts the runtime offset and expects closure to reject the section identity.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py src/tac/tests/test_packetir_exact_closure.py -q`

Result: `64 passed`.

## Reactivation Criteria

If a future PacketIR format introduces multiple score-affecting streams, it
must get the same independent per-section mutation probes and offset-bound
closure identity before any all-sections-consumed claim is accepted.
