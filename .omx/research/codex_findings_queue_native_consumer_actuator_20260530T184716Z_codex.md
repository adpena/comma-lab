# Codex Findings: Queue Native-Consumer Actuator

- date_utc: `2026-05-30T18:47:16Z`
- agent: `codex`
- scope: `tools/queue_fleet.py consume-native`
- authority: local queue/artifact routing only; no score claim, no promotion claim

## Landing

`tools/queue_fleet.py` now has a bounded `consume-native` subcommand. It selects
non-executable artifacts with typed `native_consumer_command` hints, records the
selected child commands, optionally executes them under the same fleet lock
pattern used by `supervise` and `init-missing`, writes JSONL child events, and
rescans afterward. The mode is separate from normal queue supervision so
report-only artifacts and paused exact-dispatch gates do not become false
actionability.

## Live Plan Probe

Command:

```bash
.venv/bin/python tools/queue_fleet.py --scan-limit 40 --row-limit 0 consume-native --output-dir /tmp/pact_queue_native_plan --max-artifacts 2
```

Result:

- `schema=experiment_queue_fleet_native_consumer_run.v1`
- `execute=false`
- `selected_count=2`
- `completed_child_count=2`
- `failed_child_count=0`
- selected consumers: `byte_shaving_materializer_work_queue`,
  `byte_shaving_materializer_work_queue`
- `score_claim=false`

The fixture test also covers `--execute`, proving the actuator can turn a
`byte_shaving_materializer_work_queue.v1` artifact into an `experiment_queue.v1`
without granting score or dispatch authority.

## Next Wiring

The next local-first automation step is a bounded loop:

1. `queue_fleet.py consume-native --execute` for materializer work queues.
2. `queue_fleet.py init-missing --execute` for newly emitted execution queues.
3. `queue_fleet.py supervise --execute` for local proof chains.
4. rescan and repeat until only terminal/report-only/paused-exact-dispatch rows
   remain.

That loop should stay exact-auth cold unless a later exact-readiness/eureka gate
emits an explicitly authorized dispatch candidate.
