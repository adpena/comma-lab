# Prediction-Band Axis And Anchor Custody Hardening

Date: 2026-05-16
Operator directive: L5/L5-v2 and Cathedral rigor, adversarial review, and no
false rank/dispatch authority.

## Finding

Adversarial audit found that `validate_prediction_band()` could accept a rank
reward even when the band axis, baseline axis, and landed empirical-anchor axis
were not the same evidence space. It also accepted a landed anchor without
checking anchor artifact/custody fields.

Impact: a band could mix `[contest-CUDA]`, `[contest-CPU]`, MPS/macOS advisory,
or other proxy evidence while still returning `valid_for_rank_reward=True`.
That violates the repo's apples-to-apples axis discipline and can create false
score-lowering authority.

## Patch

- Added `prediction_band_baseline_axis_mismatch` when baseline axis differs from
  the band axis.
- Added landed-anchor checks for:
  - axis presence and axis match;
  - finite numeric score;
  - archive/runtime SHA custody;
  - artifact path presence.
- Added regression tests for baseline-axis mismatch, landed-anchor axis
  mismatch, and malformed landed-anchor custody.

## Verification

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_prediction_band.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_substrate_composition_matrix.py \
  src/tac/tests/test_autopilot_dispatch_ranking.py
# 104 passed

.venv/bin/python -m ruff check \
  src/tac/optimization/prediction_band.py \
  src/tac/tests/test_prediction_band.py
# All checks passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/tac/optimization/prediction_band.py \
  src/tac/tests/test_prediction_band.py
```

## Follow-Up

Remaining audit item: exact-eval authorization should require archive/runtime
custody, not only a dispatch-packet hash, before self-authorization.
