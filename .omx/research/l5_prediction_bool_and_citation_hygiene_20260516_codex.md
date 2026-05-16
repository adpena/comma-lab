# L5 prediction boolean and citation hygiene (2026-05-16)

## Problem

Two smaller rigor issues remained after the L5/Cathedral custody hardening:

- JSON-style prediction-band parsing used Python truthiness for `planning_only`
  and `score_claim`, so string booleans such as `"false"` were accepted as
  ordinary values instead of malformed input.
- The Time-Traveler L5 composition row cited `TeNeRV` and `TeCoNeRV` in its
  literature-anchor string, but those sources are not part of the canonical
  `time_traveler_l5_v2` research-basis stack.

## Fix

- `prediction_band_from_mapping()` now requires literal JSON booleans for
  `planning_only` and `score_claim`.
- The TT5L row literature-anchor text now names only sources represented in the
  canonical L5-v2 basis family.

## Authority

This is parser/citation hygiene only. It creates no score, promotion, dispatch,
or literature-authority claim. External sources remain design prompts until the
repo has byte-closed archive/runtime evidence and exact CPU/CUDA custody.
