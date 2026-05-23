# Codex Findings: Byte-Shaving Scorer-Response Unit Wire-In

UTC: 2026-05-23T08:50:30Z
Lane: `lane_codex_byte_shaving_scorer_response_unit_wire_in_20260523`
Agent: Codex

## Finding

The byte-shaving signal surface builder correctly guarded scorer-response
dataset refs with `scorer_response_planning_value_for_target`, but those refs
were not converted into campaign units. A scorer-response/MLX calibration input
could therefore preserve normalized signal as metadata while the campaign
planner ranked only pre-existing queue/master-gradient units.

That was an orphaned-signal bug: the normalized full-video MLX/scorer-response
surface was safe, but not yet actionable by the byte-shaving campaign planner.

## Landing

Added `scorer_response_row` as a planning unit kind. The planner now supports
explicit score-delta units, including zero-byte quality-only rows, while keeping
all score/dispatch/promotion authority false.

The signal-surface builder now converts each validated scorer-response row into
a planning unit with:

- `planning_value_accessor=scorer_response_planning_value_for_target`
- `planning_value_scope=normalized_full_video`
- normalized full-video projected delta, scorer gain, and byte-budget margin
- an explicit `materialize_scorer_response_candidate` operation
- blockers requiring byte-closed materialization, runtime consumption proof, and
  exact auth eval before any score claim

Rows with raw/window deltas that disagree with the normalized full-video
objective now become units whose ranked campaign value follows the normalized
full-video projected delta only.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_byte_shaving_campaign.py
.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py src/tac/tests/test_scorer_response_dataset.py
.venv/bin/ruff check src/tac/optimization/byte_shaving_campaign.py src/tac/optimization/byte_shaving_signal_surface_builder.py src/tac/tests/test_byte_shaving_signal_surface_builder.py
git diff --check
```

Results:

- focused byte-shaving suite: 14 passed
- normalized/planning downstream suite: 150 passed
- ruff: passed
- diff check: clean

## Integration Notes

This makes scorer-response and MLX calibration signal visible to the same
campaign/autopilot surface that ranks queue and master-gradient units. The next
consumer should use these normalized scorer-response units to seed bounded
local CPU/MLX materialization sweeps, then promote only byte-closed archives
through locality controls and exact contest auth eval.

No `.gitignore` change was needed in this landing: no new generated artifact
root or large local cache namespace was introduced.
