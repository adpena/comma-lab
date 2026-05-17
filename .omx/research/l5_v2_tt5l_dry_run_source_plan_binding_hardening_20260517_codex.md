# L5 v2 TT5L Dry-Run Source-Plan Binding Hardening

Date: 2026-05-17
Author: codex
Authority: implementation hardening; `score_claim=false`;
`promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`;
`ready_for_provider_dispatch=false`; `dispatch_attempted=false`.

## Problem

The TT5L side-info Lightning execution bundle already copied
`source_spec_command_sha256` from the paired-axis plan into every cell, and the
bundle builder checked that relationship at build time. The dry-run verifier,
however, consumed the execution bundle as its only command authority. That left
a false-authority edge: if a stale or hand-edited bundle drifted from the
authoritative paired-axis plan, dry-run verification could prove launcher parse
and queue custody without independently re-proving that each cell still matched
the source plan's `spec.command` SHA.

For L5 v2, the paired-axis plan is the source of truth for the ten exact-eval
cells. The dry-run verifier must bind back to it directly.

## Patch

`src/tac/optimization/l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py`
now loads the bundle's `source_plan`, validates:

- source plan path exists;
- source plan SHA-256 matches `source_plan_sha256` from the bundle;
- source plan schema is the Lightning paired-axis plan schema;
- source plan cell coverage is exact: no missing, duplicate, extra, or keyless
  `(variant, axis)` cells.

For each dry-run cell, the verifier now compares the matching source-plan cell:

- `source_spec_command_sha256` equals source-plan `command_sha256`;
- archive SHA and byte count match;
- `pair_group_id`, `run_id`, and `local_artifact_dir` match.

The verification artifact now emits `source_plan` status at top level and
`source_plan_cell` for every cell. The regenerated artifact reports all ten
cells with `source_plan_cell.matched=true`.

## Regression

Added a focused test where a bundle cell's `source_spec_command_sha256` is
edited away from the source plan. The verifier now returns
`all_dry_runs_passed=false` with
`source_spec_command_sha256_mismatch_source_plan`.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py -q
.venv/bin/python tools/verify_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py --repo-root .
.venv/bin/python -m py_compile src/tac/optimization/l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py
git diff --check -- src/tac/optimization/l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py .omx/research/l5_v2_tt5l_dry_run_source_plan_binding_hardening_20260517_codex.md
```

Result:

- `10 passed`
- dry-run verifier: `passed_cell_count=10/10`, `all_dry_runs_passed=True`
- `py_compile` clean
- `git diff --check` clean for the touched files

No provider dispatch was attempted. This is dry-run custody hardening only.
