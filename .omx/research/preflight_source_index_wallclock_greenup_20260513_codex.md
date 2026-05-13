# Preflight source-index wall-clock greenup (2026-05-13)

## Scope

Operator goal: keep Pact score-lowering work moving by reducing preflight wall
clock while preserving exact-eval protections. This tranche targeted measured
`tac.preflight --scope all` hotspots, not proxy score logic.

## Changes

- Catalog #168 AST assign/annassign guard now prefilters candidate files before
  AST parsing. It uses `rg`/`SourceIndex` candidates containing `ast.Assign`
  instead of parsing every Python file under `src/tac`, `tools`, and
  `experiments`.
- KL-distill roundtrip guard now scans only files mentioning
  `kl_distill_segnet_only`, plus the explicit `src/tac/segmap_renderer.py`
  sentinel. The score-protective rule is unchanged: raw renderer pairs remain
  forbidden.
- `SourceIndex` text-fact schema bumped to `v26` with the new candidate needles,
  so persistent caches cannot silently omit the fast-path tokens.
- `preflight_runtime_refs.check_test_imports_resolve_to_disk` now accepts
  package/namespace directories as Python does. This clears false positives for
  valid `submissions.robust_current` contest-runtime imports without weakening
  missing-helper detection.

## Evidence

- `src/tac/tests/test_check_168_ast_walker_handles_assign_and_annassign.py`:
  27 passed.
- `src/tac/tests/test_preflight_meta_bugs.py::TestKlDistillUsesRoundtrippedFrames`
  plus Catalog #168 tests: 33 passed.
- `src/tac/tests/test_preflight_runtime_refs.py`: 19 passed.
- `compileall` passed for modified implementation and test files.
- `check_test_imports_resolve_to_disk`: 0 live violations.
- `tac.preflight --scope dev`: passed; timing artifact
  `.omx/research/artifacts/preflight_dx_profiles_20260513_codex/preflight_dev_timing_after_index_and_importfix.json`.
- `tac.preflight --scope all --allow-slow-preflight`: passed; timing artifact
  `.omx/research/artifacts/preflight_dx_profiles_20260513_codex/preflight_all_timing_after_index_and_importfix.json`.

## Timing Notes

- Catalog #168 focused test file improved from 8.43s before the candidate
  prefilter to 2.97s after.
- Live Catalog #168 check improved from about 7.64s to about 1.91s.
- Live KL roundtrip check now runs in about 0.06s in isolation.
- Full release preflight still exceeds the 30s DX target at about 57.82s. The
  remaining top hotspots are `check_no_proxy_metric_drives_decision`,
  `preflight_filename_contract`, `check_test_files_imports_resolve`,
  `check_no_compromised_lightning_supply_chain`, `check_codebase_drift`,
  `check_profile_keys_have_resolvers`, `preflight_loader_format_safety`, and
  `check_pytest_collection_clean`.

## Score-Lowering Relevance

This does not claim a score movement. It removes dev-velocity drag and false
release-sweep noise from gates that protect exact-eval score-lowering work:
eval-roundtrip KL training, static AST guard correctness, and contest-runtime
import custody. The next performance tranche should continue consolidating
broad source scans into candidate-filtered `SourceIndex` queries without
downgrading strict coverage.
