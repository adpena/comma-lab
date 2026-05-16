# Cathedral race-mode prediction-band suppression hardening - 2026-05-16

## Trigger

L5/Cathedral adversarial review found that race-mode trimmed to raw negative
`predicted_score_delta` before applying prediction-band rank suppression. A
candidate whose prediction band had already been suppressed could still survive
the race-mode subset selection and reach HALT-event emission.

## Landing

Race-mode now uses an effective negative-delta predicate:

- prediction-band suppressed candidates are not race-mode credible;
- Z1 empirical revision is applied before the negative-delta test;
- continual-posterior correction is applied when present.

This aligns race-mode candidate selection with the same proof-aware ranking
semantics used by `rank_candidates`.

## Tests

- `test_one_iteration_race_mode_applies_prediction_band_suppression_before_trim`

## Boundary

The loop still does not dispatch by itself unless the explicit operator-
authorized mode and claim lifecycle gates are active. This patch only changes
which planning rows are eligible to be surfaced in race-mode.
