# Probe-Disambiguator Source-Scope And Rank-Reward Hardening

Date: 2026-05-16
Owner: codex
Scope: Cathedral autopilot autonomous-loop intake

## Finding

`load_candidates_from_probe_disambiguator_output()` accepted read-only
`autopilot_rows` from probe-disambiguator artifacts, but only preserved
`literature_anchor`. The row lost `source_supports`, `paper_claim_scope`,
`pact_must_prove`, and `decode_complexity_evidence`, creating a no-signal-loss
gap between paper/source fidelity review and the autonomous-loop halt surface.

The same loader also did not mirror JSONL candidate intake's prediction-band
rank-reward suppression. A probe row could carry an unvalidated
`prediction_band` while keeping positive `expected_information_gain`.

## Fix

- Preserve probe-row source-scope fields on `CandidateRow`.
- Suppress positive EIG when a probe row carries prediction-band material
  without a `prediction_band_verdict.valid_for_rank_reward=true` receipt.
- Add `prediction_band_rank_reward_suppressed` to blockers and notes for the
  probe row, matching JSONL candidate intake behavior.

## Evidence

- `.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -q`
- `.venv/bin/python -m ruff check tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`
- `.venv/bin/python -m py_compile tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`
- `git diff --check -- tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`

## Follow-Up

Probe-disambiguator producers should emit these source-scope fields whenever
they cite literature or OSS sources. Rows lacking the fields remain read-only
planning rows, but they should not become operator-authorized dispatch packets
without separate exact archive/runtime custody.
