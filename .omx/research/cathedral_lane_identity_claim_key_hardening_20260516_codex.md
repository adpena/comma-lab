# Cathedral Lane Identity Claim-Key Hardening

Date: 2026-05-16
Owner: Codex
Status: landed

## Bug Class

Substrate-composition ranking rows carried campaign lane IDs only as prose-like
`campaign_metadata`. Cathedral converted those rows into `CandidateRow` objects
without structured `lane_id` or `claim_keys`, so an active claim on a canonical
campaign lane could be missed unless the candidate ID happened to match.

This was especially risky for L5 v2 / TT5L rows because campaign rows are
planning-only but still flow through the autonomous loop's conflict check.

## Fix

- `RankedDispatchCandidate` now exposes `lane_id` and `claim_keys`.
- Singleton rows propagate their canonical Pareto `lane_id`.
- Pair rows propagate component lane IDs as `claim_keys`.
- The Cathedral substrate-composition loader recovers lane IDs from legacy
  `campaign_metadata` entries and adds them to claim keys.
- The same loader now preserves `mdl_tier_c_density` and
  `predicted_dispatch_risk` so demotion/risk signals survive the canonical
  ranking source.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/autopilot_dispatch_ranking.py tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_autopilot_dispatch_ranking.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`
- `.venv/bin/python -m pytest src/tac/tests/test_autopilot_dispatch_ranking.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -q`

## Reactivation Criteria

If a future loader consumes dispatch candidates from a new source, it must carry
structured lane identity or recover exact claim keys before the autonomous loop
can evaluate dispatch conflicts.
