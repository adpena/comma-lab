# Codex Findings - DQS1 Queue Dependency Guard And Locality Timeout

UTC: 2026-05-23T16:05:33Z
Queue: `dqs1_pairset_local_first`
Authority axis: local queue control / false authority

## Finding

The DQS1 local-first queue allowed
`pairset_drop_one_rank024_pair0112.local_cpu_contest_drift_eureka` to run from
stale SQLite success state even though its new Vertigo-root
`local_cpu_advisory.json` dependency was missing. The step failed fast with:

```text
LocalCPUContestDriftError: /Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first/materialized/drop_rank024_pair0112/local_cpu_advisory.json: could not load JSON
```

Root cause: dependency readiness only checked whether the required step state was
`succeeded`; it did not re-check the required step's postconditions against the
current queue definition and artifact root.

## Landing

- `ready_steps(...)` now accepts `repo_root` and re-validates postconditions for
  succeeded dependency steps before allowing downstream readiness.
- `run_queue_worker(...)`, queue CLI `next/status/init/control/rewind/reconcile`,
  and the queue observer pass the repo root into readiness/summary paths.
- Added a regression test where a dependency artifact is deleted after success;
  the downstream step is no longer ready.

## Queue Actions

1. Ran a bounded local worker tranche (`max_steps=2`, `max_parallel=2`).
   - `dqs1_scheduler_preflight.proactive_cleanup`: succeeded.
   - `pairset_drop_one_rank024_pair0112.local_cpu_contest_drift_eureka`: failed
     because `local_cpu_advisory.json` was absent at the current Vertigo root.
2. Rewound both affected candidate chains from `build_bridge_plan` with reason:
   `stale_success_artifacts_after_storage_root_migration`.
3. Re-ran bounded local worker tranches:
   - `build_bridge_plan`: rank023 and rank024 succeeded.
   - `plan_packet`: rank023 and rank024 succeeded.
   - `materialize`: rank023 and rank024 succeeded.
4. Launched `locality_controls` for rank023 and rank024 in parallel.
   - Both hit the 900s queue timeout.
   - Neither emitted `locality_controls.json`.
   - No child locality/inflate processes remained after timeout.

## Current Queue State

Latest observation:

- status counts: `failed=2`, `queued=6`, `succeeded=8`
- failed steps:
  - `pairset_drop_one_rank023_pair0440.locality_controls`
  - `pairset_drop_one_rank024_pair0112.locality_controls`
- ready steps: none
- running steps: none
- definition drift: 0 changed, 0 missing

Timeout logs:

- `.omx/state/experiment_queue_logs/dqs1_pairset_local_first/pairset_drop_one_rank023_pair0440/locality_controls/20260523T155007Z_e273a41507340ba9.log`
- `.omx/state/experiment_queue_logs/dqs1_pairset_local_first/pairset_drop_one_rank024_pair0112/locality_controls/20260523T155007Z_c155a48cb5f2f77c.log`

Both logs contain:

```text
[experiment-queue] timeout after 900s
```

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_experiment_queue.py \
  src/tac/tests/test_dqs1_local_first_queue_builder.py

.venv/bin/python -m ruff check \
  src/comma_lab/scheduler/experiment_queue.py \
  src/comma_lab/scheduler/experiment_queue_observer.py \
  tools/experiment_queue.py \
  src/tac/tests/test_experiment_queue.py
```

Result: `57 passed`; ruff clean.

## Remaining Gap

The next bottleneck is not DAG scheduling. It is locality-control runtime:
selective inflate/control work did not complete inside the 900s queue timeout
for the rebuilt Vertigo-root rank023/rank024 candidates. Next work should either:

- profile `tools/run_decoder_q_selective_runtime_locality_controls.py` on these
  two materialized candidates;
- split locality control into smaller observable substeps;
- raise timeout only with a timing artifact and clear expected wall-clock; or
- add a cheaper preflight that catches pathological selective-runtime inflation
  before launching full locality controls.
