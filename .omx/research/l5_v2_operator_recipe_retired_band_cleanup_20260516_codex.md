# L5 v2 TT5L operator recipe retired-band cleanup — 2026-05-16

## Scope

Follow-up to the TT5L smoke-harness historical-band cleanup. The Modal A100
operator recipe already had `predicted_band: null`, but its surrounding prose
still used active-prediction language.

## Fix

- Added `prediction_band_rank_reward_suppressed: true`.
- Added explicit `retired_predicted_band: [0.150, 0.170]`.
- Changed `council_phase_5_prediction` to `null`.
- Preserved the old `0.160` only as `retired_council_phase_5_prediction`.
- Reworded first-anchor risk so returned contest-CUDA results are treated as new
  information, not as pass/fail against a retired band.

## Authority Boundary

The recipe remains dispatchable only through the existing operator authorization
and claim lifecycle. These fields do not make a score claim, rank reward,
promotion claim, or exact-eval readiness claim. TT5L still requires Dykstra
score-axis sanity, side-info proof, probe-disambiguator evidence, paired-axis
plan evidence, and paired CPU/CUDA exact anchors before architecture lock.
