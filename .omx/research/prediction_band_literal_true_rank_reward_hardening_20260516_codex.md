# Prediction-Band Literal-True Rank-Reward Hardening

Date: 2026-05-16
Owner: codex
Scope: Cathedral autopilot ranking and autonomous-loop intake

## Finding

Some prediction-band rank-reward checks treated
`prediction_band_verdict.valid_for_rank_reward` with Python truthiness. If a
JSON producer emitted `"false"` as a string, the row could keep positive
expected-information-gain rank reward despite carrying a malformed verdict.

## Fix

- Require the typed literal `true` value for prediction-band rank reward.
- Suppress EIG and EV/$ for malformed/non-true prediction-band verdicts in the
  dispatch ranker.
- Mirror the same strict semantics in JSONL/autonomous-loop intake.

## Evidence

- `.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_autonomous_loop.py src/tac/tests/test_autopilot_dispatch_ranking.py -q`
- `.venv/bin/python -m ruff check tools/cathedral_autopilot_autonomous_loop.py src/tac/optimization/autopilot_dispatch_ranking.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py src/tac/tests/test_autopilot_dispatch_ranking.py`
- `.venv/bin/python -m py_compile tools/cathedral_autopilot_autonomous_loop.py src/tac/optimization/autopilot_dispatch_ranking.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py src/tac/tests/test_autopilot_dispatch_ranking.py`
- `git diff --check -- tools/cathedral_autopilot_autonomous_loop.py src/tac/optimization/autopilot_dispatch_ranking.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py src/tac/tests/test_autopilot_dispatch_ranking.py`
