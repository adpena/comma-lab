# Codex Findings: Queue Custody And Repair Execution

UTC: 2026-05-27T21:12:54Z

## Landing

Advanced the four-week objective on two axes:

- false-authority/custody propagation: queue observers now downgrade
  `json_completion_contract` postconditions unless the artifact itself proves
  archive custody, runtime adapter identity, and receiver/runtime proof custody;
- executable repair optimization: repair campaign score queues can now run the
  source repair-waterfill queue when the work order has not materialized yet,
  then continue into scoring, blocked-signal posterior append, Cascade-C MLX
  probing, and stackability probing.

## Behavioral Changes

- `experiment_queue_observer` adds required archive/runtime/receiver custody
  revalidation for completion-contract artifacts. A syntactically passing JSON
  postcondition no longer carries custody authority by itself.
- `repair_campaign_score_queue` treats a missing-but-declarable waterfill work
  order as executable local work, not manual babysitting: validate source queue,
  run bounded worker, assert the work order exists, then score.
- The repair campaign Cascade-MLX child worker now runs all five current child
  steps, including learning-signal emission and posterior append.
- Scheduler exports include the repair-cascade result and learning-signal
  helpers so normal consumers do not need private imports.

## Authority Boundary

All new queue and observer outputs remain false-authority:

- no score claim;
- no promotion eligibility;
- no rank/kill authority;
- no budget-spend authority;
- no exact-eval dispatch authority.

The new observer guard is deliberately stricter than the worker postcondition:
the worker can mark a command successful, but the observer will not treat the
artifact as custody-positive unless the artifact and proof files withstand
independent revalidation.

## Verification

- `ruff check` on touched observer, repair queue, feedback, export, and tests
  passed.
- `pytest src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_repair_cascade_mlx_probe_queue.py -q`:
  48 passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`:
  57 passed.

## Remaining Gap

The next high-value closure is concrete MLX component-response materialization:
PoseNet-null bottom-decile artifacts, SegNet class-region masks, selector codec
bit ledgers, and receiver-consumption proof files must be emitted as actual
artifacts so the waterfill/action-functional scorer learns measured response
curves rather than blocked custody rows.
