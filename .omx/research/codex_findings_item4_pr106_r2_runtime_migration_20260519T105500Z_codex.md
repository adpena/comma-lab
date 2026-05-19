# Codex Findings: ITEM4 PR106 R2 Runtime Migration

Timestamp: 2026-05-19T10:55:00Z
Agent: codex
Task: `codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518::ITEM_4`

## Verdict

ITEM4 is now closable for the OP-1/OP-2/OP-5 reviewability batch.

Phase A added the shared helper and warn-only LOC budget gate. This pass
migrated one real oversized runtime, `submissions/pr106_latent_sidecar_r2`, to
a vendored shared helper while preserving contest-runtime closure.

## Landed Changes

- `submissions/pr106_latent_sidecar_r2/inflate.py` now imports
  `tac.substrates._shared.inflate_runtime.select_inflate_device` from its
  archive-local `src/` tree and wraps the returned string as `torch.device`.
- Added a vendored minimal helper package at
  `submissions/pr106_latent_sidecar_r2/src/tac/substrates/_shared/` so
  `inflate.sh` works with an empty external `PYTHONPATH`.
- `inflate.py` dropped from 214 physical lines to 192 lines, moving it from
  `hard_budget` to `default_budget`.
- `submissions/pr106_latent_sidecar_r2/inflate.sh` now defaults to `python3`
  and no longer appends an empty `PYTHONPATH` segment.
- `tac.submission_inflate_loc_budget` now detects real shared-helper imports
  with AST parsing, so docstring/prose mentions no longer count as adoption.
- `test_inflate_select_device_parity` now covers the migrated PR106 R2 runtime
  and clears vendored `codec` / `model` modules between submission imports.

## Proofs

- Empty external `PYTHONPATH` import resolved `tac` to
  `submissions/pr106_latent_sidecar_r2/src/tac/__init__.py` and the helper to
  `submissions/pr106_latent_sidecar_r2/src/tac/substrates/_shared/inflate_runtime.py`.
- Synthetic old-vs-new inflate proof exercised sidecar parsing, device
  selection, batching, interpolation, rounding, and raw writing without writing
  multi-GB contest outputs.
- Parity artifact:
  `experiments/results/pr106_latent_sidecar_r2_runtime_migration_20260519T104500Z_codex/parity_proof.json`
  with SHA-256
  `2ebb4636c293d3bdebed5fb4bf65632bdc2988590bf670230b61b23f7c9f892c`.
  A compact tracked copy is
  `.omx/research/pr106_latent_sidecar_r2_runtime_migration_parity_20260519T104500Z_codex.json`.
  It reports `byte_identical=true` and identical raw SHA-256
  `d08aaeab9fc7913c5b0d88ce310643667dd441eee99af385cf6276aa4d6624ce`.
- LOC audit now reports `submissions/pr106_latent_sidecar_r2/inflate.py` as
  192 lines, `default_budget`, `shared_runtime_helper_adopted=true`; hard
  violations dropped from 14 to 13.

## Verification

```bash
.venv/bin/pytest \
  src/tac/tests/test_pr106_latent_sidecar.py \
  src/tac/tests/test_substrate_inflate_runtime.py \
  src/tac/tests/test_check_295_submission_inflate_empty_pythonpath.py \
  src/tac/tests/test_check_328_submission_inflate_py_loc_budget.py \
  src/tac/tests/test_inflate_select_device_parity.py -q

.venv/bin/ruff check \
  src/tac/submission_inflate_loc_budget.py \
  src/tac/tests/test_check_328_submission_inflate_py_loc_budget.py \
  src/tac/tests/test_inflate_select_device_parity.py \
  submissions/pr106_latent_sidecar_r2/inflate.py \
  submissions/pr106_latent_sidecar_r2/src/tac/substrates/_shared/inflate_runtime.py \
  submissions/pr106_latent_sidecar_r2/src/tac/__init__.py \
  submissions/pr106_latent_sidecar_r2/src/tac/substrates/__init__.py \
  submissions/pr106_latent_sidecar_r2/src/tac/substrates/_shared/__init__.py

PYTHONPATH= PACT_INFLATE_DEVICE=cpu .venv/bin/python \
  submissions/pr106_latent_sidecar_r2/inflate.py

.venv/bin/python tools/audit_submission_inflate_py_loc_budget.py --repo-root . --json
```

## Remaining Risk

This does not strict-flip Catalog #328 because 13 hard-budget findings remain.
The remaining hard findings should be handled by future migrations or explicit
tracked waivers, not by loosening the gate.

This pass does not claim score movement. It is reviewability/runtime-closure
hardening only.
