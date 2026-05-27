# Codex Findings: Repair Campaign Blocked-Signal Preservation

UTC: 2026-05-27T17:56:07Z

## Finding

Selected repair allocations preserved the typed action-functional fields, but
blocked optimizer rows were lower fidelity. That was a signal-loss risk: missing
MLX custody, receiver proof gaps, component-response terms, per-op byte deltas,
and hard legal/runtime constraints could be seen in the raw score rows but were
not preserved in the optimizer's blocked-allocation surface.

## Landing

Blocked allocation rows now preserve the same planner-relevant fields as selected
rows:

- family, candidate, acquisition, entropy position, targeted dimensions, and
  operation levels;
- requested bytes, objective delta, campaign score, and per-op byte delta;
- component-response terms;
- receiver proof status and exact missing artifact names;
- execution-gate missing artifacts;
- hard legal/runtime constraints;
- interaction and stacking terms.

The autonomous child-queue dependency constants were also made explicit, and the
repair campaign score child now names its waterfill-child dependency through a
precomputed child-run-step map. That prevents ordering signal from depending on
local action insertion order.

## Authority

Blocked rows remain false-authority and cannot claim score, spend budget,
promote, rank/kill, or dispatch exact eval. They are planner signal only.

## Verification

- `ruff check src/tac/optimization/repair_campaign_scorer.py src/tac/tests/test_repair_campaign_scorer.py src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `py_compile` on the same four files
- `pytest src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_frontier_rate_attack_feedback.py::test_frontier_feedback_cli_writes_valid_followup_queue -q`
