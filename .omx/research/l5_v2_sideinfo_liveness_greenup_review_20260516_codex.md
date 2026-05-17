# L5 v2 side-info liveness greenup review

Date: 2026-05-16

Scope: TT5L side-info liveness contract, effect-curve gating, trainer export
guard, materialized work-unit readiness, and focused regression tests.

Verdict: clean for this scoped landing.

## Review Notes

- The reusable liveness implementation lives in
  `src/tac/substrates/time_traveler_l5_autonomy/archive.py` so trainer export
  and L5 readiness inspection share one source of truth.
- The export guard still avoids arbitrary sparsity thresholds. It refuses empty
  and all-zero side-info, while recording pair and section coverage so sparse
  hard-pair mechanisms remain testable rather than prematurely rejected.
- The side-info effect-curve contract now requires liveness evidence for every
  cell, and active side-channel variants must prove nonzero side-info before
  the schedule can route toward an anchor packet.
- The regenerated schedule remains fail-closed: C1/Z5/TT5L paired observations
  are still the active measurement path, and the current TT5L side-info curve is
  invalid because it is incomplete and the observed trained CUDA cell has
  checked all-zero side-info liveness.
- Focused tests, lint, whitespace checks, and review-tracker policy passed
  before review ingestion.

## Greenup Results

### experiments/train_substrate_time_traveler_l5_autonomy.py -- CLEAN
### src/tac/optimization/l5_staircase_v2.py -- CLEAN
### src/tac/optimization/l5_v2_measurement_schedule.py -- CLEAN
### src/tac/optimization/l5_v2_sideinfo_effect_curve.py -- CLEAN
### src/tac/substrates/time_traveler_l5_autonomy/archive.py -- CLEAN
### src/tac/substrates/time_traveler_l5_autonomy/tests/test_time_traveler_archive.py -- CLEAN
### src/tac/tests/test_l5_staircase_v2.py -- CLEAN
### src/tac/tests/test_l5_v2_measurement_schedule.py -- CLEAN
### src/tac/tests/test_l5_v2_sideinfo_effect_curve.py -- CLEAN
