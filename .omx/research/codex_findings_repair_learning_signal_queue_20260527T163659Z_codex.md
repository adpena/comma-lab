# Codex Findings: Repair Learning Signal Queue

Date: 2026-05-27T16:36:59Z

## Verdict

Repair stackability results now become queue-owned planner signal instead of
stopping at a local probe or replay artifact. Ready repair rows execute as a
three-step local chain: stackability probe, deterministic replay bundle, and
false-authority learning signal for acquisition/posterior planning.

## Landed Integration

- `repair_campaign_learning_signal` builds a typed local planning update from a
  score report, stackability probe, and replay bundle.
- `tools/build_repair_campaign_learning_signal.py` writes the signal as a
  deterministic JSON artifact and supports the queue-facing
  `--learning-signal-out` flag.
- `repair_campaign_stackability_queue` now records `learning_signal_path` and
  `learning_signal_schema` metadata for ready rows.
- Ready queue rows now add `build_repair_campaign_learning_signal` after the
  replay-bundle step with schema, false-authority, and dispatch-false
  postconditions.
- `tac.optimization` now exposes the learning-signal schema constants, error,
  and builder through the package's lazy export surface.
- The focused test executes all three queue steps and checks that replay
  identity, local planning update readiness, and feature-vector signal survive
  into the learning artifact.

## Safeguards

- The signal requires score report, probe, and replay bundle schemas to match.
- The signal fails closed on any truthy authority field in its inputs or output.
- The local planning update is explicitly `[macOS-MLX research-signal]` and
  cannot claim score, spend budget, promote, rank/kill, or dispatch exact eval.
- The posterior update remains blocked until exact-axis component response and
  receiver runtime materialization exist.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/repair_campaign_learning_signal.py tools/build_repair_campaign_learning_signal.py src/comma_lab/scheduler/repair_campaign_stackability_queue.py src/tac/tests/test_repair_campaign_stackability_queue.py`
- `.venv/bin/python -m py_compile src/tac/optimization/repair_campaign_learning_signal.py tools/build_repair_campaign_learning_signal.py src/comma_lab/scheduler/repair_campaign_stackability_queue.py src/tac/tests/test_repair_campaign_stackability_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_stackability_queue.py src/tac/tests/test_repair_campaign_score_queue.py -q`
  - `8 passed`

## Remaining Work

The signal is queue-owned and planner-consumable, but it is not yet appended to a
canonical posterior ledger. The next implementation step is an append-only repair
posterior writer with deterministic row ids, duplicate suppression by replay
identity, and an acquisition-rule consumer that changes future repair-family
priority from these artifacts automatically.
