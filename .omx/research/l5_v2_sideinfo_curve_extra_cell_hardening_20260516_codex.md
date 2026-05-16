# L5 v2 side-info curve extra-cell hardening

- schema: `l5_v2_sideinfo_curve_extra_cell_hardening_v1`
- created_at_utc: `2026-05-16T22:16:36Z`
- surface: `src/tac/optimization/l5_v2_measurement_schedule.py`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

The L5 v2 side-info effect-curve validator already required paired
`contest_cpu`/`contest_cuda` evidence for the five canonical TT5L variants:
`zero`, `random_lsb`, `shuffled`, `trained`, and `ablated`.

The adversarial gap was not a missing-cell gap. It was an extra-cell gap:
an artifact could include all required cells and also include an extra declared
variant or observed cell such as `oracle` or `macos_cpu`. That extra cell did
not make the current code false-positive on score authority by itself, but it
left a path for arbitrary, unregistered side-info evidence to travel with a
valid architecture-lock curve.

## Fix

`validate_l5_v2_sideinfo_effect_curve()` now rejects:

- declared `required_variants` outside the canonical set;
- observed axes outside `contest_cpu` and `contest_cuda`;
- observed variants outside the canonical five-cell variant set.

This keeps the L5 v2 lattice exact rather than permissive: additional
side-info experiments must become explicit new schedule variants, not silent
extra cells inside the architecture-lock artifact.

## Regression

`src/tac/tests/test_l5_v2_measurement_schedule.py::test_l5_v2_schedule_rejects_extra_sideinfo_axes_and_variants`
constructs a curve with all required cells plus an extra `macos_cpu/oracle`
cell and verifies the scheduler stays on
`measure_tt5l_sideinfo_effect_curve` with explicit blockers:

- `tt5l_sideinfo_effect_curve_variants_extra:oracle`
- `tt5l_sideinfo_effect_curve_observed_axes_extra:macos_cpu`
- `tt5l_sideinfo_effect_curve_observed_variants_extra:oracle`

This is a no-score, no-promotion hardening change. It prevents arbitrary
side-info cells from becoming hidden evidence in the L5 v2 staircase.
