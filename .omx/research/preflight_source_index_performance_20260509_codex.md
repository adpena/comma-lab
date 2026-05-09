# Preflight Source-Index Performance Slice (2026-05-09)

research_only=true

## Scope

Ownership: preflight/DX performance only.

Changed surfaces:
- `src/tac/preflight.py`
- `src/tac/tests/test_undeployed_artifact_producers.py`

No remote dispatch was launched.

## Change

Check 39 (`check_undeployed_archive_artifact_producers`) previously scanned
the same producer files once per archive artifact and rescanned deployment
surfaces once per producer. The check now builds:

- one producer map for all known archive artifact names;
- one deployment-reference text set for remote lane scripts and deploy
  registries;
- optional reuse of the active `SourceIndex` for file inventory and source
  reads inside the normal preflight source-index context.

The single-artifact helper remains as a compatibility wrapper.

## Timings

Baseline commands run before the patch:

- default CLI: `/usr/bin/time -p .venv/bin/python -m tac.preflight`
  - `real 9.81`
- exhaustive profile with clean shortcut disabled:
  - `PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 .venv/bin/python tools/profile_preflight_latency.py --surface preflight-all-codebase --preflight-timeout-s 180 --top 20 --max-step-records 40`
  - total `57.599s`
  - Check 39 appeared in top 20 at `7.707s`
- focused hot-check profile:
  - Check 39 `0.805756s`

After patch:

- focused Check 39 profile:
  - `PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 .venv/bin/python tools/profile_preflight_latency.py --surface preflight-checks --preflight-check check_undeployed_archive_artifact_producers --preflight-timeout-s 30 --preflight-check-timeout-s 30 --top 5 --max-step-records -1 --json-out .omx/tmp/preflight_check39_after_final.json`
  - Check 39 `0.361s`
- broader focused hot-check profile:
  - `PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 .venv/bin/python tools/profile_preflight_latency.py --surface preflight-checks --preflight-check check_test_files_imports_resolve,preflight_filename_contract,preflight_loader_format_safety,check_pytest_collection_clean,preflight_dead_resolvers,check_codebase_drift,check_undeployed_archive_artifact_producers,check_profile_keys_have_resolvers,check_pose_projection_train_inference_parity,check_no_raw_zip_extractall --preflight-timeout-s 120 --preflight-check-timeout-s 30 --top 20 --max-step-records -1 --json-out .omx/tmp/preflight_hot_after_final.json`
  - Check 39 `0.382975s`
- exhaustive profile rerun with clean shortcut disabled:
  - total `65.645s`
  - Check 39 no longer appears in top 20

The exhaustive total is not an apples-to-apples improvement claim because
parallel partner work changed the working tree during this slice. The measured
owned hot check improved from `0.805756s` to `0.361s` in the same focused
profiler surface and no longer ranks as a full-preflight top-20 hotspot.

## Verification

- `.venv/bin/python -m py_compile src/tac/preflight.py src/tac/source_index.py`
- `.venv/bin/python -m pytest src/tac/tests/test_source_index.py src/tac/tests/test_undeployed_artifact_producers.py src/tac/tests/test_preflight_codebase_drift_scan_scope.py -q`
  - `26 passed`
- `/usr/bin/time -p .venv/bin/python -m tac.preflight`
  - passed; last warm run `real 10.99`

## Residual Hot Spots

Latest exhaustive profile top residuals:

- `preflight_filename_contract`: `13.803s`
- `check_test_files_imports_resolve`: `11.705s`
- `preflight_loader_format_safety`: `11.305s`
- `check_pytest_collection_clean`: `9.550s`
- `preflight_dead_resolvers`: `9.027s`
- `check_codebase_drift`: `9.007s`
- `check_profile_keys_have_resolvers`: `8.292s`

Next highest-EV DX pass: reduce repeated AST/text work across filename
contract, loader safety, and test-import resolution without changing their
strict semantics.

## Solver Wire-In

This is a DX/preflight scalability landing, not a score-affecting codec lane.

- sensitivity-map contribution: N/A
- Pareto constraint: N/A
- bit-allocator hook: N/A
- cathedral autopilot dispatch hook: N/A
- continual-learning posterior update: N/A
- probe-disambiguator: N/A
