# L5 v2 side-info curve variant dispatch expansion

- date: 2026-05-16
- operator scope: L5 v2 staircase priority, bug hunting, arbitrariness removal
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false

## Finding

The TT5L side-info effect curve is a 10-cell lattice:

- axes: `contest_cpu`, `contest_cuda`
- variants: `zero`, `random_lsb`, `shuffled`, `trained`, `ablated`

Before this patch, the schedule correctly named the aggregate
`measure_tt5l_sideinfo_effect_curve` measurement, but the paired dispatch plan
would expand that aggregate into one CPU/CUDA work unit. That was an
actuation-arbitrariness bug: one paired archive cannot cover the five required
side-info variants.

## Change

- `src/tac/optimization/l5_v2_measurement_schedule.py` now records the side-info
  curve builder tool, required variants, required cells, and aggregate output
  artifact in the schedule row.
- `src/tac/optimization/l5_v2_paired_measurement_dispatch_plan.py` now expands a
  side-info curve aggregate row into five paired CPU/CUDA work units, one per
  side-info variant.
- Each variant work unit carries:
  - `work_unit_id`
  - `source_measurement_id`
  - `sideinfo_variant`
  - the two exact cells it is responsible for
  - variant-specific lane ids, pair group ids, output roots, harvest commands,
    and non-executable paired Modal command templates.

The current live schedule still fails closed on missing C1/Z5/TT5L paired probe
observations, so the current durable dispatch plan remains at three active
probe-filling work units. The side-info expansion is now ready for the next
first-match transition after those probe observations become eligible.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_paired_measurement_dispatch_plan.py -q`
  - result: `16 passed in 0.34s`
- `.venv/bin/python -m ruff check src/tac/optimization/l5_v2_measurement_schedule.py src/tac/optimization/l5_v2_paired_measurement_dispatch_plan.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_paired_measurement_dispatch_plan.py`
  - result: `All checks passed!`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_paired_measurement_dispatch_plan.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_operator_briefing.py -q`
  - result: `139 passed in 59.79s`
- `.venv/bin/python -m py_compile src/tac/optimization/l5_v2_measurement_schedule.py src/tac/optimization/l5_v2_paired_measurement_dispatch_plan.py tools/build_l5_v2_lattice_measurement_schedule.py tools/build_l5_v2_paired_measurement_dispatch_plan.py`
  - result: clean
- synthetic eligible probe intake check:
  - active rule: `measure_tt5l_sideinfo_effect_curve`
  - active measurement ids: `['measure_tt5l_sideinfo_effect_curve']`
  - expanded work unit count: `5`
  - variants: `['zero', 'random_lsb', 'shuffled', 'trained', 'ablated']`
- current operator briefing check:
  - next action: `populate_and_evaluate_c1_z5_tt5l_probe_observations`
  - `tt5l_architecture_lock_allowed`: `false`
  - `tt5l_sideinfo_effect_curve_artifact_valid`: `false`
  - paired plan work units: `3`
  - paired plan stale: `false`

## Next Frontier Action

Close the current first-match blocker by filling the paired exact C1/Z5/TT5L
probe observations. After they become eligible, regenerate the schedule and
dispatch plan; the TT5L side-info effect curve gate will then produce five
variant-paired work units instead of one aggregate placeholder.
