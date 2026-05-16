# Cathedral Prediction-Band Rank Reward Hardening - 2026-05-16

## Finding

The Cathedral dispatch ranker suppressed EIG when a
`prediction_band_verdict` explicitly denied rank reward, but a row carrying a
`prediction_band` with no verdict could still contribute expected information
gain. That made uncustodied numeric bands eligible for orthogonal-pair reward
through an omission.

Classification: planning-ranker rigor bug. No score claim. No lane promotion.

## Fix

- Added a row-level helper requiring that any row with a prediction band also
  carry a verdict before it can contribute rank reward.
- Applied the helper to both singleton and orthogonal-pair candidates.
- Added focused tests for singleton suppression and orthogonal-pair suppression
  when a component has a band but no verdict.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_autopilot_dispatch_ranking.py::test_prediction_band_rank_reward_requires_literal_true_verdict \
  src/tac/tests/test_autopilot_dispatch_ranking.py::test_prediction_band_rank_reward_requires_verdict_when_band_present \
  src/tac/tests/test_autopilot_dispatch_ranking.py::test_orthogonal_pair_suppresses_eig_when_component_band_lacks_verdict
```

Result: 3 passed.

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/autopilot_dispatch_ranking.py \
  src/tac/tests/test_autopilot_dispatch_ranking.py
```

Result: all checks passed.
