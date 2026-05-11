# Preflight parallel coverage audit

Generated: 2026-05-11T01:25:00Z

## Question

Does the reduced wall-clock all-lanes preflight still run every protective gate,
or did parallel scheduling narrow the coverage surface?

## Verdict

The reduced wall-clock path preserves coverage. `tools/all_lanes_preflight.py`
uses the same constructed step list for serial and parallel execution:

- `26` gate checks
- `3` lane dry-run checks
- `29` total rows
- same ordered identity and status under `--jobs 1` and `--jobs 8`
- all rows passed in both runs

The comparison checked `(section, number, name, passed, status,
forensic_only, local_smoke_only)` for every row. There were zero mismatches.

## Evidence

Serial profile:

- command:
  `.venv/bin/python tools/all_lanes_preflight.py --jobs 1 --timings --timings-json .omx/research/artifacts/preflight_coverage_serial_jobs1_20260511_codex.json`
- wall: `6.809072s`
- serial sum: `6.808469s`
- workers: `1`
- passed: `29`
- failed: `0`

Parallel profile:

- command:
  `.venv/bin/python tools/all_lanes_preflight.py --jobs 8 --timings --timings-json .omx/research/artifacts/preflight_coverage_parallel_jobs8_20260511_codex.json`
- wall: `2.919316s`
- serial sum: `13.317235s`
- workers: `8`
- estimated speedup: `4.561765x`
- passed: `29`
- failed: `0`

Canonical strict developer preflight also passed:

- command:
  `.venv/bin/python -m tac.preflight --scope dev --timings-json .omx/research/artifacts/tac_preflight_dev_coverage_20260511_codex.json`
- scope: `dev`
- wall: `8.595854s`
- recorded strict checks: `23`
- status: `passed`

## Protection Surfaces Still Covered

The all-lanes operator gate still covers:

- dispatch CLI/shell hazards;
- reverse-engineering tree custody and release manifest custody;
- hidden-gem registry/readiness;
- semantic label contract;
- engineered-correction readiness;
- HNeRV frontier scorecard and low-level repack proof;
- tooling-consolidation inventory;
- recovered remote lane canonicalization;
- untracked source/no-signal-loss disposition;
- orphan/preserved-orphan/recovery custody;
- release index/worktree split;
- nested gitlink custody;
- staged public release hygiene;
- cross-paradigm frontier inventory;
- PR91/HPM1 fail-closed custody;
- frontier monolithic archive layout;
- Omega-OPT anchor discipline;
- eval loader drift diagnostic;
- A2 packet ladder closure;
- Phase A custody/discoverability;
- Modal image build-order guard;
- Apogee intN dry-run;
- Omega-W-V3 dry-run;
- PR106 sidechannel dry-run.

The `tac.preflight --scope dev` strict bug-class catalog still covers the
bounded developer loop and explicitly warns that release/custody checks are not
included in dev scope. Release, dispatch, and frontier claims still require the
operator all-lanes gate and/or explicit `--scope all --allow-slow-preflight`
where appropriate.

## New Regression Protection

`src/tac/tests/test_all_lanes_preflight_timing_profile.py` now includes a
parallel scheduler regression test proving that `_run_steps_with_budget(...,
max_workers>1)` executes every submitted step exactly once, preserves
gate/lane sort order in results, and keeps pass/fail semantics intact.

## Remaining Speed Targets

The parallel all-lanes gate is below the 30s crash budget. The next optimization
targets are still the slowest broad scans:

1. Gate #8 tooling consolidation inventory;
2. Gate #0 dispatch CLI/shell hazards;
3. Gate #3 semantic-label contract;
4. Gate #10 untracked source inventory;
5. Gate #22 eval loader drift diagnostic.

These should be optimized by further shared `SourceIndex` facts, narrower
candidate filters, and eventually native/SIMD scanners only when golden
conformance tests prove identical findings.
