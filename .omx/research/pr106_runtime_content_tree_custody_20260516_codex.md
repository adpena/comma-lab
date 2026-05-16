# PR106 Runtime Content-Tree Custody - 2026-05-16

## Summary

PR106 runtime consumption proofs carried `runtime_source_tree_sha256`, but exact
closure and exact-eval custody also reason about `runtime_content_tree_sha256`.
That created a source-vs-content ambiguity: local runtime-consumption proofs
could be valid parser/apply evidence while missing the content-tree field needed
to bind them cleanly to exact eval and full-frame parity packets.

## Fix

- `pr106_runtime_source_manifest()` now emits both:
  - `runtime_source_tree_sha256`: existing mode-inclusive source manifest hash.
  - `runtime_content_tree_sha256`: path/bytes/content-SHA manifest hash.
- `prove_pr106_sidecar_runtime_decode_consumption()` accepts
  `expected_runtime_content_tree_sha256` and fails closed on malformed or
  mismatched values.
- `tools/prove_pr106_sidecar_runtime_consumption.py` exposes
  `--expected-runtime-content-tree-sha256`.
- The stale PR106 R2 PR101-grammar runtime source-tree constant in
  `tools/all_lanes_preflight.py` and its test fixture was refreshed to the
  current deterministic manifest hash produced by
  `pr106_runtime_source_manifest()`.

This is a custody hardening only. It does not create a score claim or promotion
path.

## Verification

- `src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py` asserts the
  content-tree field is deterministic and that malformed/mismatched expected
  content-tree hashes block runtime-consumption claims.

## Reactivation Criteria

Reopen if a PR106 runtime proof, exact-closure packet, or exact-ready queue row
can bind a runtime by source-tree hash alone while omitting content-tree custody.
