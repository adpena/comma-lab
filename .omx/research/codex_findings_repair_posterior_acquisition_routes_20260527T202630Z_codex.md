# Codex Findings: Repair Posterior Acquisition Routes

UTC: 2026-05-27T20:26:30Z
Agent: Codex

## Verdict

The repair campaign continual-learning loop now has a queue-owned producer and a
scorer-owned consumer for blocked repair posterior rows. Frozen or blocked
repair-campaign allocations can append deterministic posterior rows through the
repair campaign score queue, and the scorer projects posterior acquisition
policies back into explicit follow-up routes.

This remains false-authority. The posterior, follow-up routes, local MLX
signals, and score queue outputs do not claim score, authorize spend, promote,
rank/kill, or dispatch exact eval.

## What Changed

- `repair_campaign_score_queue` now derives and records a deterministic
  stackability posterior lock path beside the configured posterior JSONL.
- The score queue's blocked-learning append step passes both `--posterior-path`
  and `--lock-path`, so blocked allocation signals are preserved in the same
  queue-owned posterior that future scoring reads.
- `repair_campaign_scorer` now groups posterior rows by
  `recommended_acquisition_policy` and emits
  `repair_campaign_posterior_acquisition_followup.v1` rows.
- Follow-up rows carry policy, priority, activation action, target queue
  artifact key, required evidence surface, family ids, typed response ids,
  missing artifact mass, blocker mass, and false-authority guards.

## Mathematical Object

The posterior route is an acquisition-policy projection over typed repair
ledger observations:

`policy -> {families, typed_response_ids, missing_artifact_total, blocker_total,
priority, activation_action, queue_artifact_key, required_evidence_surface}`.

This makes the repair optimizer consume interaction evidence as an operational
object instead of leaving blocked rows as passive diagnostics. In particular,
activation-plan-only rows can now increase priority for targeted component
response harvest, while MLX-custody, receiver-closed byte credit, exact-axis
handoff, and missing-artifact blockers route to the appropriate next queue
surface.

## Verification

- `.venv/bin/ruff check src/tac/optimization/repair_campaign_scorer.py src/tac/tests/test_repair_campaign_scorer.py src/comma_lab/scheduler/repair_campaign_score_queue.py src/tac/tests/test_repair_campaign_score_queue.py`
- `.venv/bin/python -m py_compile src/tac/optimization/repair_campaign_scorer.py src/comma_lab/scheduler/repair_campaign_score_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_scorer.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_score_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py -q`
- `.venv/bin/python tools/lane_maturity.py validate`
- Review policy checks on all four touched Python files: 0 violations.

## Remaining Work

The next useful slice is to make these acquisition follow-up routes directly
actuate queue selection in the frontier final-rate autoloop, while preserving
the false-authority boundary until exact CPU/CUDA evidence exists.
