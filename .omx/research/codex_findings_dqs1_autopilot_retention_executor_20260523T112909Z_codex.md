# Codex Findings - DQS1 Autopilot Retention Executor - 2026-05-23T11:29:09Z

## Scope

Follow-up hardening for the DQS1 local-first queue after raw-retention planning
landed. The remaining failure class was that certified retention execution still
depended on a manual `tools/compact_experiment_artifacts.py --execute --action
move` command after each harvested candidate.

## Finding

The autopilot had a narrow delete-only scratch helper for
`local_cpu_advisory_work/inflated` and `extracted`, but it did not use the
canonical `comma_lab.artifact_retention` planner/executor. That left locality
raw trees, MLX caches, cold-store copy verification, and JSONL execution
journals outside the autonomous loop.

Failure class: the queue could produce and plan retention signal, but the
actuator that should consume that signal was still operator-memory dependent.

## Landing

`tools/run_dqs1_local_first_autopilot.py` now executes post-harvest retention for
the just-harvested candidate through `build_retention_plan()` and
`execute_retention_plan()`. The new path:

- targets only the harvested candidate root;
- re-plans immediately before execution;
- fails closed if any blocked retention candidate is present;
- writes `.omx/research/dqs1_artifact_retention_<candidate>_<stamp>.json`;
- writes the paired JSONL journal from `execute_retention_plan()`;
- supports `--retention-action delete|move`;
- supports external cold storage through `--retention-cold-store-root`;
- can include certified MLX caches via `--include-mlx-cache-retention`;
- keeps score/promotion/exact-dispatch authority false.

The autopilot output was also compacted so large orphaned queue details no longer
flood stdout.

## Empirical Anchor

Live autopilot command:

```bash
.venv/bin/python tools/run_dqs1_local_first_autopilot.py \
  --execute \
  --max-candidates 1 \
  --max-total-steps 7 \
  --max-steps-per-worker 7 \
  --idle-sleep-seconds 0 \
  --max-idle-cycles 0 \
  --retention-action move \
  --retention-cold-store-root /Volumes/APDataStore/pact-tertiary/artifact_cold_store \
  --retention-min-bytes 1 \
  --min-free-disk-gb 40
```

Result:

- candidate: `pairset_drop_two_r028_023_p0257_0440`
- local score: `[macOS-CPU advisory] 0.19203961709818362`
- conservative projected contest score: `0.1920321170981836`
- eureka trigger: `false`
- exact-auth request: `false`
- post-harvest retention: 5 certified rows moved, 0 blocked,
  14,649,816,858 bytes moved
- local candidate directory after move: 1.3M
- SSD cold-store candidate directory: 14G

Queue rerouted to `pairset_drop_two_r029_017_p0259_0242`.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_dqs1_local_first_autopilot.py src/tac/tests/test_artifact_retention.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
- `.venv/bin/python -m ruff check tools/run_dqs1_local_first_autopilot.py src/tac/tests/test_dqs1_local_first_autopilot.py`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `git diff --check`

All passed.

## Remaining Work

Run the DQS1 queue through the new autopilot path for the remaining local-first
candidates. The next engineering target is to persist compact autopilot run
summaries with `--summary-out` or equivalent, then use those summaries as a
typed acquisition-history surface for learned queue ordering and exact-auth
dispatch thresholds.
