# L5 v2 TT5L Dry-Run Exact-Cell Coverage Hardening

Date: 2026-05-17
Author: codex
Authority: implementation hardening; `score_claim=false`;
`promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`;
`ready_for_provider_dispatch=false`; `dispatch_attempted=false`.

## Problem

The TT5L side-info Lightning dry-run verifier already checked required cells,
paired-axis identity, local archive custody, runtime markers, queue metadata,
and state/stdout agreement. Two exact-coverage edges remained: a bundle with
duplicate or extra `(variant, axis)` cells would fail only indirectly through
cell-count or missing-cell symptoms, and the verifier did not reload the source
paired-axis plan to prove the bundle cell identities still matched the plan on
disk.

For the L5 v2 side-info effect curve, the required surface is exactly five
variants by two axes. Any duplicate, extra, or keyless cell is a custody bug:
the downstream effect curve must not infer ten clean paired cells from a bundle
that contains ambiguous rows or a stale source plan join.

## Patch

`src/tac/optimization/l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py`
now computes explicit cell-key coverage before executing dry-run commands:

- `missing_cells`
- `duplicate_cells`
- `extra_cells`
- `key_missing_indices`

Those fields are emitted under the verification artifact's `coverage` object
and become global fail-closed blockers:

- `source_bundle_missing_cells:...`
- `source_bundle_duplicate_cells:...`
- `source_bundle_extra_cells:...`
- `source_bundle_cell_key_missing_indices:...`

This turns the audit requirement "all ten cells present, no duplicates/extras"
into executable verifier behavior rather than a reader-side inference.

The verifier also now reloads `bundle["source_plan"]`, verifies its SHA-256,
schema, exact 5x2 coverage, and per-cell identity fields, then emits
`source_plan` and per-cell `source_plan_cell` summaries in the dry-run
verification artifact. Mismatches fail closed before any dry-run command result
can imply custody.

## Test

Added a focused regression in
`src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py`
that appends both a duplicate required cell and an unexpected extra cell. The
verifier now marks the artifact not ready and records both exact blockers. The
fake bundle fixture now writes a source paired-axis plan so source-plan custody
is exercised in the normal positive path too.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py -q
.venv/bin/python -m py_compile src/tac/optimization/l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py
git diff --check -- src/tac/optimization/l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py
```

Result:

- `10 passed`
- combined Catalog #315 + dry-run custody + execution-bundle sweep: `76 passed`
- `py_compile` clean
- `git diff --check` clean for the touched verifier files

No provider dispatch was attempted. This is custody hardening only.
