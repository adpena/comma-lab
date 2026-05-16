# L5 v2 smoke harness historical-band authority cleanup — 2026-05-16

## Scope

L5 v2 false-authority cleanup for the Time-Traveler L5 macOS-CPU advisory
smoke harness and the older C2/Z7 mature L5 campaign memo.

## Finding

The TT5L harness had correctly retired the active `[0.150, 0.170]` planning
band, but the public JSON/verdict surface still used `predicted_band` and
`in_predicted_band` vocabulary. That vocabulary can leak false authority into
operator review, even when `score_claim=false` and `promotion_eligible=false`.

## Fix

- Renamed the optional re-analysis CLI surface from `--predicted-band-*` to
  `--historical-band-*`.
- Renamed emitted summary fields from active-prediction language to historical
  retired-band language:
  - `historical_band_low`
  - `historical_band_high`
  - `active_historical_band`
  - `in_historical_band`
  - `retired_historical_band_low`
  - `retired_historical_band_high`
  - `retired_historical_band_active`
  - `retired_band_origin`
- Retained the old `[0.150, 0.170]` numbers only as
  `tt5l_additive_prediction_retired_20260516`.
- Appended a supersession note to
  `.omx/research/campaign_lane_c2_z7_mature_predictive_receiver_l5_20260514.md`
  that makes the mature L5 bands historical planning priors only.

## Evidence

Focused verification command:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_smoke_time_traveler_l5_autonomy.py \
  -q
```

Expected pass condition: the smoke harness tests prove the no-active-band
default, advisory non-promotability flags, and historical-band vocabulary.

## Authority Boundaries

- No score claim.
- No promotion eligibility.
- No dispatch readiness claim.
- No family retirement or falsification from macOS-CPU advisory output.
- L5 v2 architecture lock remains blocked on filled probe observation intake
  plus paired exact contest-axis evidence.
