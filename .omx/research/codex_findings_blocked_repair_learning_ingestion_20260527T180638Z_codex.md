# Codex Findings: Blocked Repair Learning Ingestion

UTC: 2026-05-27T18:06:38Z

## Finding

The repair campaign queue learned from positive stackability probes but still
lost blocked-allocation signal. A blocked allocation carried useful information
about missing local MLX custody, exact receiver proof gaps, exhausted byte
credit, or non-improving component response, but the queue did not append that
negative evidence to the repair posterior.

## Landing

Blocked repair allocations now emit deterministic false-authority learning
signals:

- source score report hash and blocked-row identity;
- blocker and missing-artifact names;
- requested bytes, campaign score, per-op byte delta, component-response terms,
  entropy position, targeted dimensions, and operation levels;
- acquisition policy recommendation such as materializing missing MLX custody,
  waiting for receiver-closed byte credit, or decreasing priority after
  non-improving local response.

The repair campaign score queue now owns two additional steps after scoring:

1. `build_repair_campaign_blocked_learning_signals`
2. `append_blocked_repair_campaign_learning_posterior`

Those steps run before the stackability worker path and keep all authority
fields false. Positive selected rows still use the replay-bundle path; blocked
rows use a no-replay deterministic identity so the posterior can learn without
pretending a local MLX stackability replay occurred.

## Authority

The new signals are local planning/acquisition updates only. They do not grant
score claim, promotion, rank/kill, budget spend, or exact dispatch authority.

## Verification

- `ruff check` on the new tools and touched repair modules/tests
- `py_compile` on the new tools and touched repair modules/tests
- `pytest src/tac/tests/test_repair_campaign_score_queue.py -q`
- `pytest src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_repair_campaign_stackability_queue.py src/tac/tests/test_repair_campaign_scorer.py -q`
