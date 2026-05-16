# L5 v2 side-info effect curve paired-axis requirement

- date: 2026-05-16
- scope: L5 v2 lattice measurement schedule
- code: `src/tac/optimization/l5_v2_measurement_schedule.py`
- tests: `src/tac/tests/test_l5_v2_measurement_schedule.py`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

The TT5L side-info effect-curve measurement required only `contest_cuda`, even
though its output can steer architecture lock and first-anchor planning. That
creates an unpaired-axis authority leak: a CUDA-only usefulness curve could
shape the L5 v2 staircase before CPU/CUDA custody exists.

## Change

`measure_tt5l_sideinfo_effect_curve` now uses the default paired
`["contest_cpu", "contest_cuda"]` axis requirement and carries an explicit
blocker:

`requires_paired_cpu_cuda_sideinfo_effect_curve_before_architecture_lock`

The markdown renderer now includes each measurement's `required_axes`, so the
operator-facing schedule preserves the paired-axis contract.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_staircase_v2.py -q`
