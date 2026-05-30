# Codex Findings: Queue Native-Consumer Routing

- date_utc: `2026-05-30T18:41:38Z`
- agent: `codex`
- scope: `experiment_queue.v1 fleet scanner, native queue/report artifacts`
- authority: local telemetry only; no score claim, no promotion claim

## Finding

The queue fleet had already reached `actionable_count=0`,
`invalid_queue_count=0`, and `needs_recovery_count=0`, but
`NON_EXECUTABLE_QUEUE_ARTIFACT` rows were still passive. The recurring passive
schemas were materializer work queues, optimizer candidate queues, exact-ready
queues, queue validation reports, queue observations, performance summaries,
fleet status reports, and queue summaries.

This was not a queue-health failure, but it was an automation/signal-custody
gap: fleet telemetry knew these artifacts were not executable
`experiment_queue.v1` files, yet did not expose their native consumers.

## Landing

The fleet scanner now emits typed `experiment_queue_fleet_native_consumer_hint.v1`
metadata for every non-executable artifact schema seen in the bounded live scan.
Materializer work queues route to `tools/build_materializer_execution_queue.py`;
optimizer candidate source queues route to
`tools/build_materializer_submission_closure.py`; exact-ready queues route to
`tools/build_materializer_exact_eval_consumer.py`; report artifacts are marked
known report-only. All hints preserve false-authority fields.

## Live Probe

Command:

```bash
.venv/bin/python tools/queue_fleet.py --scan-limit 160 --row-limit 0 status --format json
```

Result:

- `queue_count=160`
- `actionable_count=0`
- `ready_to_supervise_count=0`
- `needs_recovery_count=0`
- `invalid_queue_count=0`
- `non_executable_artifact_count=90`
- `native_consumer_artifact_count=90`
- `known_native_consumer_artifact_count=90`

## Next Wiring

The next autonomous step is to let the fleet supervisor optionally consume
`next_native_consumer_commands` under a separate bounded mode, then rescan and
supervise newly emitted execution queues. That should stay distinct from normal
`next_supervise_commands` so report-only artifacts and exact-dispatch gates never
become false actionability.
