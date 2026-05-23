# Codex Findings - DQS1 End-Of-Queue Harvest - 2026-05-23T11:49:57Z

## Scope

Adversarial follow-up during the DQS1 local-first autopilot run after post-harvest
retention execution was automated.

## Finding

`pairset_drop_two_r029_017_p0259_0242` completed all seven local steps, but the
autopilot failed at harvest because every candidate in the current action summary
already had local advisory evidence. `build_dqs1_harvest_result()` treated "no
safe next candidate" as a fatal reroute error, which prevented harvest artifacts
and retention execution from being written for the completed final candidate.

Failure class: end-of-queue was encoded as an exception instead of a terminal
state.

## Landing

`src/comma_lab/scheduler/dqs1_local_first_harvest.py` now converts the specific
`no safe DQS1 local-first candidate found` reroute condition into a valid harvest
with `rerouted_queue=None`. If an output queue path was supplied, the queue is
left untouched. Other reroute errors still fail closed.

Regression coverage in `src/tac/tests/test_dqs1_local_first_queue_builder.py`
verifies that an observe-only harvest with all candidates consumed:

- returns the harvest record;
- creates no exact-auth request;
- returns `rerouted_queue=None`;
- preserves the queue file hash.

## Empirical Anchor

After the fix, the already-completed candidate harvested and compacted cleanly:

- candidate: `pairset_drop_two_r029_017_p0259_0242`
- local score: `[macOS-CPU advisory] 0.19203861709818362`
- conservative projected contest score: `0.19203111709818363`
- eureka trigger: `false`
- exact-auth request: `false`
- no reroute available
- post-harvest retention: 5 certified rows moved, 0 blocked,
  14,649,816,858 bytes moved
- local candidate directory after move: 1.3M
- SSD cold-store candidate directory: 14G

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_autopilot.py src/tac/tests/test_artifact_retention.py`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/dqs1_local_first_harvest.py src/tac/tests/test_dqs1_local_first_queue_builder.py tools/run_dqs1_local_first_autopilot.py src/tac/tests/test_dqs1_local_first_autopilot.py`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `git diff --check`

All passed before commit.

## Remaining Work

The current DQS1 action summary is exhausted for safe local-first rerouting. Next
work should either regenerate a richer action summary from updated learned
signals, broaden the candidate generator beyond this consumed action set, or
switch to the next frontier-relevant family while reusing the autopilot,
retention, and eureka contracts.
