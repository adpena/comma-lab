# L5 v2 temporal side-info byte-range hardening - 2026-05-16

## Trigger

L5/Cathedral adversarial review found that
`byte_closed_temporal_sideinfo_consumption` proved parser consumption,
output change, and nonnegative mutated byte offsets, but did not bind those
offsets to the parsed temporal-sideinfo section. A mutation elsewhere in the
archive could therefore satisfy the gate.

## Landing

The L5 v2 gate semantics now require:

- a parsed section byte range (`section_offset` plus section length);
- all `mutated_byte_offsets` to be absolute archive offsets inside that range;
- `section_sha256` / parsed-section identity;
- baseline and mutated archive SHA-256 pair with actual archive change;
- runtime-tree SHA-256 for the inflate/parser implementation under test.

This keeps the gate byte-closed: the proof must show that temporal side-info
bytes are emitted, parsed from a known archive range, mutated in that same
range, and observed to change full-frame inflate output under the recorded
runtime.

## Tests

- `test_l5_v2_sideinfo_consumption_binds_mutation_to_parsed_section_range`
- `test_l5_v2_sideinfo_consumption_requires_archive_runtime_section_identity`

## Boundary

This remains a dispatch/proof gate, not a score claim. Promotion still requires
paired exact CPU/CUDA evidence and full exact-eval custody.
