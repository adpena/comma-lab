# Codex Findings: Repair Campaign Continual-Learning Loop

UTC: 2026-05-27T17:53:28Z

## Finding

The repair campaign path had a real automation break between queue-owned scoring
and queue-owned learning. The refresh flow emitted `repair_campaign_score_queue`,
and that score queue could build `repair_campaign_stackability_queue`, but the
generated stackability queue was still a second manual hop for replay bundle
generation, learning-signal emission, and posterior append.

## Landing

`repair_campaign_score_queue` now owns the full local advisory follow-through:

- assert the repair-waterfill work order exists;
- score the typed waterfill ledger into the default repair campaign scorer;
- build the selected-allocation stackability queue;
- validate the generated stackability queue;
- run the generated stackability queue in bounded local mode, producing the
  replay bundle, learning signal, posterior append report, and deterministic
  worker result.

The autonomous chain parent also binds `repair_campaign_score_queue` as a child
queue whenever refresh produced it, and its score-child run step depends on the
repair-waterfill child run step. This keeps the encoder-side order explicit:
waterfill allocation first, campaign scoring second, stackability learning third.

## Authority

All new rows remain false-authority:

- no score claim;
- no promotion eligibility;
- no rank/kill authority;
- no budget spend;
- no exact-dispatch readiness.

The path is local/MLX advisory custody only until exact CPU/CUDA auth-axis
payloads and receiver-closed archive/runtime artifacts exist.

## Verification

- `ruff check src/comma_lab/scheduler/repair_campaign_score_queue.py src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `py_compile` on the same four files
- `pytest src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_repair_campaign_stackability_queue.py -q`
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py::test_frontier_feedback_cli_writes_valid_followup_queue -q`
