# Prediction-Band Exact-Eval Anchor Closure Hardening

Date: 2026-05-16
Owner: codex
Scope: `src/tac/optimization/prediction_band.py`

## Finding

`validate_prediction_band()` allowed a landed empirical anchor to grant rank
reward from only axis, score, archive/runtime SHA, and artifact path. That was
too thin: a score stub with hashes could look like exact-eval evidence without
sample count, hardware, command/log trace, component distances, archive bytes,
or formula closure.

## Fix

- Require landed empirical anchors to carry:
  - `n_samples`
  - `hardware`
  - `auth_eval_command`
  - `log_path`
  - `archive_bytes`
  - `seg_dist`
  - `pose_dist`
  - `score`
- Recompute the official contest formula from component fields and archive
  bytes before allowing rank reward.
- Add regressions for thin score stubs and formula mismatches.

## Evidence

- `.venv/bin/python -m pytest src/tac/tests/test_prediction_band.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_substrate_composition_matrix.py src/tac/tests/test_build_composition_ranking_json.py src/tac/tests/test_autopilot_dispatch_ranking.py -q`
- `.venv/bin/python -m ruff check src/tac/optimization/prediction_band.py src/tac/tests/test_prediction_band.py`
- `.venv/bin/python -m py_compile src/tac/optimization/prediction_band.py src/tac/tests/test_prediction_band.py`
- `git diff --check -- src/tac/optimization/prediction_band.py src/tac/tests/test_prediction_band.py`

## Result

Prediction bands can still remain visible as planning priors, but they no
longer receive Cathedral/autopilot rank reward from hash-only score stubs.
