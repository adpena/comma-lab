# Codex Findings: PR106 Runtime Custody Import Shadow

UTC: 2026-05-26T03:07:47Z

## Finding

The PR106/R2 sidecar runtime-consumption gate was failing on a stale
`runtime_source_tree_sha256`, but the deeper bug class was stronger than stale
custody: the runtime manifest did not include the vendored `src/tac/**` helper
files now imported by `submissions/pr106_latent_sidecar_r2/inflate.py`, and the
runtime import harness could resolve an already-imported repository `tac`
package instead of the submission-local vendored package.

This made the proof weaker than its claim. It could prove parser/runtime
sidecar consumption while not hashing every runtime source file involved in the
execution path, and while not guaranteeing the loaded helper was the same helper
the contest runtime would use from an empty `PYTHONPATH`.

## Fix Landed

- `pr106_runtime_source_manifest()` now includes deterministic `src/tac/**/*.py`
  vendored helper files in the runtime source/content tree.
- `_runtime_import_context()` now shadows existing `tac` modules during runtime
  imports and restores them afterward, forcing submission-local import
  resolution for vendored runtime helpers.
- PR106/R2 expected runtime tree custody was refreshed to the new stronger tree
  hash after proving runtime consumption passes with the vendored helper in the
  manifest.
- Regression coverage verifies both vendored helper custody and that monkeypatching
  the repository helper cannot affect the loaded submission runtime helper.

## Validation

- `src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py`
- `src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py`
- focused touched-surface stack: 97 passed
- `tools/lane_maturity.py validate`: 1372 lanes validated cleanly

## Follow-Up

Audit other source-runtime proofs for the same class: any proof that imports a
submission-local package namespace must shadow already-loaded repo packages and
hash every vendored helper file it can execute.
