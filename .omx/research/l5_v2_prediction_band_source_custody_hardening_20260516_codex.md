# L5 v2 Prediction-Band Source Custody Hardening

Date: 2026-05-16
Author: codex
Scope: L5 v2 staircase, cathedral autopilot Pareto rows, prediction-band rank authority

## Finding

The adversarial review found a custody asymmetry in prediction bands:

- `BandSource.local_ledger_paths` only needed to be non-empty.
- `per_substrate_pareto_rows()` called `validate_optional_prediction_band()` without an artifact base directory.

That combination could both suppress valid source-backed bands and allow bogus
ledger references to look source-backed. For L5 v2, this is a rank-authority
bug: source-backed theory can guide dispatch planning, but only repo-local
ledger evidence should clear the source-custody part of validation.

## Fix

- `validate_prediction_band()` now validates every `local_ledger_paths` entry:
  - rejects transient `/tmp`, `/var/tmp`, and `/private/tmp` paths;
  - rejects paths outside the artifact base directory;
  - rejects missing ledger files;
  - emits index annotations for the failing source ledger.
- `per_substrate_pareto_rows()` now passes a repo-root artifact base directory
  into prediction-band validation by default.
- `l5_v2_prediction_band_verdict()` validates against the repo root so the L5
  v2 source ledgers clear source custody while baseline and empirical-anchor
  blockers remain visible.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_prediction_band.py \
  src/tac/tests/test_substrate_composition_matrix.py \
  src/tac/tests/test_l5_staircase_v2.py -q
```

Result: `103 passed`.

## Current L5 v2 Band Classification

The L5 v2 prediction band remains valid for dispatch planning but rank-blocked:

- `prediction_band_baseline_missing`
- `prediction_band_baseline_custody_missing`
- `prediction_band_baseline_artifact_missing`
- `prediction_band_empirical_anchor_missing`

There are no source-ledger blockers for the canonical L5 v2 source ledgers.
