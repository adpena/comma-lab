# Cross-Paradigm Atom Adapters - 2026-05-06 Codex

## Scope

Implemented a planning-only adapter layer for common meta-Lagrangian atoms
across HNeRV rate recode, WR01 wavelet, categorical/openpilot mask planning,
LA-pose planning, and foveation planning.

## Artifacts

- `src/tac/optimization/cross_paradigm_atoms.py`
- `src/tac/tests/test_cross_paradigm_atoms.py`
- `tools/build_cross_paradigm_atom_ledger.py`

## Contract

Each adapter emits deterministic rows with:

- `atom_id`
- `family`
- `pareto_scope`
- `byte_delta`
- `expected_seg_dist_delta`
- `expected_pose_dist_delta`
- `confidence`
- `interaction_assumptions`
- `archive_manifest_path`
- `archive_manifest_sha256`
- `research_basis_ids`
- `evidence_grade`
- `dispatch_blockers`

The ledger builder calls
`tac.optimization.meta_lagrangian_allocator.build_atom_ledger` and then
reattaches adapter assumptions/source blockers to allocator rows. No GPU
dispatch is attempted or enabled.

Adapters now attach research-basis ids by family/paradigm, for example
`lapose_2026`, `foveated_telepresence_2025`, `fridrich_stc_2011`, and
`yousfi_onehot_jpeg_2020`. These ids are provenance for planning math only:
they are not score evidence and do not remove archive, runtime, or exact-CUDA
blockers.

## Evidence

- [empirical:src/tac/tests/test_cross_paradigm_atoms.py] Adapter fixtures cover
  all five required paradigms, direct allocator ingestion, cross-paradigm
  ledger preservation, deterministic ordering, fail-closed validation, and the
  optional CLI.
- [empirical:local-test] `.venv/bin/python -m pytest
  src/tac/tests/test_cross_paradigm_atoms.py
  src/tac/tests/test_meta_lagrangian_allocator.py -q` passed with 17 tests.
- [empirical:local-lint] `.venv/bin/python -m ruff check
  src/tac/optimization/cross_paradigm_atoms.py
  tools/build_cross_paradigm_atom_ledger.py
  src/tac/tests/test_cross_paradigm_atoms.py` passed.

## Dispatch Status

No lane claim was made and no GPU dispatch was attempted. These atoms are
planning inputs only and remain blocked on byte-closed archive manifests,
stack interaction review, and exact CUDA auth eval before any score claim.
