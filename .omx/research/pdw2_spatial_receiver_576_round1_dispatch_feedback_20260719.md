# Task #576 dispatch round-1 feedback

Round 1 ended on a transport/max-output failure and is incomplete. Preserve the useful new
module, tests, and equation builder, but close these exact defects before returning:

1. `_validate_packet` currently calls `bytes(encode_target=target)`. That raises `TypeError` for
   every valid packet and is not caught. Remove the bogus branch; strict `decode_pdw2` plus exact
   `encode_pdw2(target) == bytes(packet)` is the only check.
2. `_vector_fields_from_target` must select representatives with two distinct classes. Sorting
   all candidates and taking the first two can choose the same class. Its fallback attempts to
   reshape four scalars to a full field and is invalid. Use a deterministic analytical or bounded
   search, retain the first representative per distinct class, then broadcast only after two
   distinct classes exist.
3. Do not hash a real n600 feature field by materializing `np.ascontiguousarray(field)` in RAM.
   The probe must stream the read-only memmap pair-by-pair and increment content and label hashes.
4. Complete `tools/probe_pdw2_spatial_receiver.py`. It must accept packet and quotient `.npy`,
   support `--pair-caps 24,600`, stream without label output, record peak RSS, build the executable
   non-identifiability witness, and find a canonical coefficient mutation whose partition changes
   on the real field. Canonical JSON only; exact authority fields from the spec.
5. Apply the minimal integration edit to
   `src/tac/boundary_math/integer_plane_emitter_byte_close.py`: import the exact blocker constant
   and use it in the receiver-consumed refusal/receipt. Do not enable consumed mode or change the
   archive schema.
6. Direct Python outputs should use stable JSON-compatible lists or tests should assert tuples
   consistently. Run the tests; do not leave an unexecuted mismatch.
7. Equation provenance must say `[macOS-CPU advisory]`, not `[contest-CPU advisory]`; hardware is
   `macos_arm64`. State explicitly that the non-identifiability implication is a direct proof with
   no imported theorem/paper. Keep the broader `packet + counted spatial generator + scorer-free
   RGB pullback` family open.
8. Run the exact focused pytest command, Ruff, and py_compile. Do not commit and do not edit parent-
   owned receipt/DAG/findings/state files.

Review provenance: parent fresh-eyes review of dispatch round 1; implementation remains
`recovery-written-UNREVIEWED` until these defects close and tests pass.
