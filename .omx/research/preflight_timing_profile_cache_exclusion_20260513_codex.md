# Preflight Timing-Profile Cache Exclusion

Date: 2026-05-13
Author: codex
Scope: full preflight wall-clock / cache correctness

## Finding

`preflight_all(verbose=False, wall_clock_budget_s=30)` still timed out on the
current tree when no clean-cache hit was available:

- observed failure: `full preflight exceeded 30s wall-clock DX budget`
- complete slow profile with explicit override:
  `python -m tac.preflight --scope all --allow-slow-preflight`
- measured wall time: about `47.4s`
- measured serial step time: about `110.4s`
- slowest recorded checks included `check_pytest_collection_clean`,
  `check_test_files_imports_resolve`,
  `check_no_compromised_lightning_supply_chain`,
  `preflight_filename_contract`,
  `check_no_proxy_metric_drives_decision`, and `check_codebase_drift`

The immediate false-cache-miss bug was that generated timing profiles under
`reports/preflight*_timing*.json` were included in the full-preflight clean
fingerprint. Running a timing profile changed the fingerprint, preventing the
next unchanged run from using the clean cache.

## Fix

`_preflight_all_fingerprint_paths()` now excludes
`reports/preflight*_timing*.json` from the clean-cache source fingerprint.
This preserves custody-sensitive result/status manifests while keeping
generated timing diagnostics from invalidating the cache they are meant to
debug.

Regression coverage:

- `test_preflight_all_clean_cache_ignores_generated_timing_profiles`

## Verification

- `pytest src/tac/tests/test_preflight_all_clean_cache.py::test_preflight_all_clean_cache_ignores_generated_timing_profiles src/tac/tests/test_probe_inflate_shell_output_parity.py src/tac/tests/test_hnerv_lowlevel_packer.py`
  passed: `17 passed in 1.82s`.
- `ruff check src/tac/tests/test_preflight_all_clean_cache.py tools/probe_inflate_shell_output_parity.py src/tac/hnerv_lowlevel_packer.py src/tac/tests/test_hnerv_lowlevel_packer.py src/tac/tests/test_probe_inflate_shell_output_parity.py`
  passed.
- After one explicit slow full-preflight pass repopulated the clean cache,
  `preflight_all(verbose=False, wall_clock_budget_s=30)` completed in about
  `3.17s` wall time.

## Remaining Performance Work

The cache fix prevents timing diagnostics from poisoning the fast path, but the
cold full release sweep is still about `47s` on this machine. Further work
should target the recorded cold-run hotspots with source-index equivalence
tests, especially pytest collection, test import resolution, Lightning
supply-chain scanning, filename-contract scanning, proxy-metric scanning, and
codebase drift scanning.
