# Cathedral Autopilot Loader Fail-Closed Hardening

Date: 2026-05-16
Author: codex + cathedral worker
Commit containing code: `da31349c1`
Scope: `tools/cathedral_autopilot_autonomous_loop.py`, loader tests

## Finding

The L5/Cathedral adversarial review found several authority-ingestion bugs in
the autopilot loader surface:

- JSON strings such as `"false"` could be truthy-coerced for review fields.
- Malformed substrate-composition alpha state could silently collapse to `{}`.
- Composition alpha fallback matching used substring checks, allowing `c1` to
  match `c10`.
- Substrate-composition ranking artifacts were not schema-enforced before
  ingestion.
- Rank-blocked prediction bands could carry positive EIG into stale ranking
  artifacts if consumers did not re-check rank authority.

## Fix

`da31349c1` hardened the autopilot loaders:

- Added strict optional/literal JSON bool parsing for `license_ok`,
  `sideinfo_consumed`, `exact_duplicate`, ranking envelope booleans, and
  score/promotion authority flags.
- Made malformed existing substrate-composition alpha JSON fail closed instead
  of returning an empty index.
- Replaced substring fallback matching with exact `__x__` pair-key parsing.
- Required canonical `tac_autopilot_dispatch_ranking_v1` schema for substrate
  composition ranking ingestion and candidate-substrate mapping.
- Zeroed EIG and appended `prediction_band_rank_reward_suppressed` when a
  stale ranking row has a prediction-band verdict that does not allow rank
  reward.

## Verification

Worker verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -q
.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py -q
```

Combined parent verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_prediction_band.py \
  src/tac/tests/test_substrate_composition_matrix.py \
  src/tac/tests/test_l5_staircase_v2.py -q
```

Result: `237 passed`.

## Residual Status

No known issue remains for the bounded loader-hardening scope. Autonomous
dispatch still remains operator-gated and planning-only rows still carry
`score_claim=false`, `promotion_eligible=false`, and
`ready_for_exact_eval_dispatch=false`.
