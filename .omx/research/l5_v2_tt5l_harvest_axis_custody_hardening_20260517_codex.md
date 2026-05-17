# L5 v2 TT5L Harvest Axis-Custody Hardening

Date: 2026-05-17
Author: codex
Authority: implementation hardening; `score_claim=false`;
`promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`;
`ready_for_provider_dispatch=false`; `dispatch_attempted=false`.

## Problem

The TT5L side-info harvest bridge already preserved all ten
`variant x axis` cells from the Lightning paired-axis plan and blocked missing
artifacts, archive byte/SHA mismatches, raw `score_axis` mismatch, and raw
`pair_group_id` / `run_id` mismatch before source-metadata fallback.

One bridge-local gap remained: `ready_for_effect_curve_build` could rely on the
downstream effect-curve validator to reject mismatched `eval_device`,
`inflate_device`, or hardware custody. That downstream guard is useful, but the
harvest bridge itself is the first post-harvest custody surface and should not
call a cell builder-ready when the harvested artifact's CPU/CUDA device
evidence is contradictory.

## Patch

`src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_harvest.py` now reuses
`tac.exact_eval_custody.validate_exact_eval_evidence()` for every harvested
artifact before setting the cell ready for effect-curve build. The bridge
requires:

- expected axis match
- expected archive SHA match
- artifact path + artifact SHA
- hardware
- auth-eval command
- log path
- inflate/eval device evidence
- inflated outputs manifest
- raw output aggregate SHA

Validator blockers are surfaced as
`harvested_exact_eval_custody_<blocker>:<variant>:<axis>`.

## Test

Added a regression where a `contest_cuda` trained cell carries
`score_axis=contest_cuda` but CPU `inflate_device`, CPU `eval_device`, and CPU
hardware. The harvest bridge now records:

- `harvested_exact_eval_custody_hardware_not_cuda:trained:contest_cuda`
- `harvested_exact_eval_custody_inflate_device_not_cuda:trained:contest_cuda`
- `harvested_exact_eval_custody_eval_device_not_cuda:trained:contest_cuda`

and keeps `ready_for_effect_curve_build=false`.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py -q
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py -q
.venv/bin/python -m py_compile src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_harvest.py src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py
.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py --repo-root .
git diff --check -- src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_harvest.py src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py
```

Result:

- `9 passed`
- bridge + downstream effect-curve sweep: `23 passed`
- `py_compile` clean
- harvest bridge CLI: `cell_count=10`, `harvested_exact_eval_artifact_count=0`,
  `missing_exact_eval_artifact_count=10`, `ready_for_effect_curve_build=false`,
  `blockers=10`
- `git diff --check` clean for the touched harvest files

No provider dispatch was attempted. The current canonical harvest artifact
still has zero harvested exact-eval artifacts; this patch only hardens the
first bridge that will consume them.
