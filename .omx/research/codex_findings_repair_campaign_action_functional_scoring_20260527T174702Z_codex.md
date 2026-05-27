# Codex Findings: Repair Campaign Action-Functional Scoring

UTC: 2026-05-27T17:47:02Z

## Summary

Advanced the Week 2 repair optimizer surface from generic objective ranking
toward an executable action-functional ledger. The default repair campaign
scorer already selected local MLX-ready rows under receiver-closed byte credit,
but scored rows did not explicitly preserve the full evidence bundle needed to
reason about operator chains.

The scorer now emits first-class action-functional fields on every campaign
score row and selected allocation:

- per-op archive byte delta;
- SegNet and PoseNet response terms;
- receiver-proof custody status with exact missing artifacts;
- hard legal/runtime constraints;
- existing entropy-position, interaction, stackability, and byte-credit fields.

Rows still remain false-authority. Missing receiver proof or exact-axis custody
does not block local MLX advisory execution, but it is named explicitly before
any budget spend, exact dispatch, or promotion can be considered.

## Landed Integration

- `src/tac/optimization/repair_campaign_scorer.py`
  - Added extraction of `allocation_action_term.T_i.archive_byte_delta_vs_baseline`.
  - Added component-response term extraction for SegNet/PoseNet local response
    signals.
  - Added receiver proof status records for runtime-consumption proof,
    receiver-consumed candidate archive, component replay manifest, and exact
    axis component response.
  - Added hard legal/runtime constraint propagation into score rows and selected
    optimizer allocations.
- `src/tac/tests/test_repair_campaign_scorer.py`
  - Expanded the scorer fixture to include allocation action terms and local
    MLX component response terms.
  - Asserted that selected rows and allocations preserve bytes delta,
    SegNet/PoseNet response, receiver-proof missing artifacts, and runtime
    constraints.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/repair_campaign_scorer.py src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_score_queue.py`
- `.venv/bin/python -m py_compile src/tac/optimization/repair_campaign_scorer.py src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_score_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_score_queue.py -q`
  - `8 passed`
- `.venv/bin/python tools/lane_maturity.py validate`
  - `1440 lane(s) validated cleanly`
- `.venv/bin/python tools/review_gate_hook.py`
  - passed

## Remaining Scope

The next repair optimizer move is to bind these action-functional rows into
actual stackability probe execution and posterior updates by default, so
positive and negative MLX/local results update acquisition policy automatically.
