# Prediction-Band Custody Guard - 2026-05-16

## Summary

Kant's read-only Cathedral audit found that literature/source-scope metadata is
now guarded, but the numeric `predicted_delta_alone_band` values that feed
EV-per-dollar ranking still lacked their own custody model. This patch adds a
reusable prediction-band validator and threads its verdict through the
composition matrix, the direct autopilot ranker, and the composition-ranking
JSON bridge.

## Guarded Failure Mode

Without this guard, a row could cite a paper honestly while still letting an
uncustodied numeric band influence clean rank authority. The new rule is:

- nonzero prediction bands without a `prediction_band` custody payload emit
  `prediction_band_missing`;
- parsed bands must declare subject, kind, finite bounds, axis, baseline
  custody, source ledgers and research-basis ids, uncertainty, supersession
  status, and empirical-anchor status;
- `score_claim=true` inside a prediction band is a hard blocker;
- zero-delta rows can remain annotation-only.

## Integration

- `src/tac/optimization/prediction_band.py` is the canonical schema and
  validator.
- `src/tac/optimization/substrate_composition_matrix.py` appends prediction-band
  blockers to `ParetoRow.dispatch_blockers`.
- `src/tac/optimization/autopilot_dispatch_ranking.py` preserves the verdict in
  serialized dispatch rows.
- `tools/build_composition_ranking_json.py` appends the same blockers to
  generated Cathedral ranking JSON rows, making them operator-review rows
  instead of clean dispatch candidates.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_prediction_band.py src/tac/tests/test_substrate_composition_matrix.py src/tac/tests/test_build_composition_ranking_json.py src/tac/tests/test_autopilot_dispatch_ranking.py -q`
- `.venv/bin/python -m ruff check src/tac/optimization/prediction_band.py src/tac/optimization/substrate_composition_matrix.py src/tac/optimization/autopilot_dispatch_ranking.py tools/build_composition_ranking_json.py src/tac/tests/test_prediction_band.py src/tac/tests/test_substrate_composition_matrix.py src/tac/tests/test_build_composition_ranking_json.py src/tac/tests/test_autopilot_dispatch_ranking.py`
- `.venv/bin/python -m py_compile src/tac/optimization/prediction_band.py src/tac/optimization/substrate_composition_matrix.py src/tac/optimization/autopilot_dispatch_ranking.py tools/build_composition_ranking_json.py src/tac/tests/test_prediction_band.py`

## Follow-Up

Populate rank-active rows with real `prediction_band` payloads only when their
axis, baseline, uncertainty, source ids, and empirical anchors are known. The
parallel source-provenance audit should decide which missing paper/OSS records
must be added before any L5/Cathedral prediction band becomes clean rank reward.
