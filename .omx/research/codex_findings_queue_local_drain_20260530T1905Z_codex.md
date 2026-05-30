# Codex Findings - Queue Local Drain Automation - 2026-05-30T1905Z

## Scope

Implemented and exercised a queue-owned local drain loop for the current final-rate
attack fleet. This is local-only custody and telemetry, not score authority.

## Landed Behavior

- `tools/queue_fleet.py drain-local` now runs bounded cycles over:
  - native non-executable artifact consumers;
  - missing experiment queue state initialization;
  - bounded local queue supervision.
- Reports use `experiment_queue_fleet_local_drain_run.v1`, compact status
  snapshots, JSONL child events, a fleet lock, and false-authority fields.
- Existing materializer execution queues are treated as already consumed rather
  than rebuilt.
- `byte_shaving_materializer_work_queue.v1` artifacts with zero executable rows
  are preserved as blocked signal and no longer advertise runnable native
  consumer commands.

## Live Proof

- Plan-only dry run:
  - `.omx/research/queue_fleet_local_drain_plan_20260530T185604Z/fleet_local_drain_result.json`
  - selected 2 native consumers, 2 missing-state inits, 2 local supervisors;
    `failed_child_count=0`.
- Bounded execute proof:
  - `.omx/research/queue_fleet_local_drain_exec_20260530T185704Z/fleet_local_drain_result.json`
  - generated one materializer execution queue, initialized one queue, ran one
    local supervisor tick; `failed_child_count=0`.
- Bug-class proof:
  - `.omx/research/queue_fleet_local_drain_exec_20260530T185808Z/fleet_local_drain_result.json`
  - exposed `no executable materializer work rows` from a materializer work queue
    that was incorrectly advertised as runnable.
- Repaired execute proof:
  - `.omx/research/queue_fleet_local_drain_exec_20260530T190139Z/fleet_local_drain_result.json`
  - after refusal hardening, drained the next executable native artifact,
    initialized its generated queue, and ran another local supervisor tick;
    `failed_child_count=0`.

## Tests

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/queue_fleet.py tools/queue_fleet.py src/tac/tests/test_queue_fleet_tool.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_queue_fleet_tool.py`

Result: 20 queue-fleet tests passed.

## Frontier

No contest score authority changed in this pass. Current recorded frontier remains:

- `[contest-CPU Linux x86_64]`: `0.1919853363`, archive `b7106c9bdbb8...`
- `[contest-CUDA T4]`: `0.2053300290`, archive `9cb989cef519...`

## Next

Run `drain-local --execute` with larger caps once the current patch lands, then
wire the terminal local drain outputs into the existing exact-readiness and
eureka-gate surfaces. Recovery queues should remain typed and explicit until
their blockers are converted into safe queue-owned recovery actions.
