# L5 v2 Prediction-Band Dispatch Readiness Hardening

Date: 2026-05-16
Operator directive: prioritize L5/L5-v2 staircase rigor, bug hunting,
adversarial review, source fidelity, and no false dispatch authority.

## Finding

`l5_v2_dispatch_readiness()` previously set `ready_for_dispatch` from gate
artifact validity alone. That made a fully evidenced gate packet look
dispatch-ready even while the embedded L5 v2 prediction band remained
rank-blocked by missing baseline custody and missing empirical anchors.

Impact: gate evidence is useful for diagnostic/probe work, but it is not enough
to authorize score/rank dispatch or rank reward. Treating those as the same
readiness bit risks turning a source-backed prior into operational authority.

## Patch

- Added `ready_for_gate_probe_dispatch` for artifact-backed diagnostic/gate
  readiness.
- Added `ready_for_score_or_rank_dispatch`, which requires both valid gate
  evidence and a rank-clean prediction-band verdict.
- Kept legacy `ready_for_dispatch` conservative by aliasing it to
  `ready_for_score_or_rank_dispatch`.
- Added `prediction_band_not_dispatch_ready` plus namespaced prediction-band
  blockers into the readiness blocker list.
- Added a regression proving valid gate artifacts do not unlock score/rank
  dispatch while the prediction band is still blocked.

## Verification

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_l5_staircase_v2.py
# 13 passed

.venv/bin/python -m ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_staircase_v2.py
# All checks passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_staircase_v2.py
```

## Follow-Up

Next hardening target from the same audit: prediction-band validation should
reject baseline-axis mismatches and malformed landed anchors before any band can
influence rank reward.
