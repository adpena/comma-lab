# Codex Findings: Materializer Execution Queue Telemetry

**UTC**: 2026-05-23T17:20:19Z
**Lane**: `lane_codex_materializer_work_queue_executor_bridge_20260523`

## Finding

The byte-shaving materializer backlog could now build executable
`byte_shaving_materializer_work_queue.v1` rows, but those rows still stopped at
JSON planning. They were not yet runnable by the shared `experiment_queue.v1`
worker, which meant byte-range proof chains could not use the existing pause,
rewind, dependency, concurrency, logging, and state machinery.

The queue worker also lacked a reusable throughput signal. Completed steps
recorded elapsed time and logs, but artifact footprint and per-resource
performance were not normalized into a consumer-facing summary for learned
sweeps, MLX/local CPU routing, Dask-style executors, or Rust hotspot decisions.

## Landed

- Added `build_materializer_execution_queue(...)`, which compiles executable
  materializer work rows into local-only `experiment_queue.v1` experiments.
- Added CLI support in `tools/build_byte_shaving_campaign_queue.py` via
  `--materializer-execution-queue-out`.
- Added false-authority metadata to materializer execution experiments so local
  proof-chain execution cannot become score, promotion, rank/kill, or dispatch
  authority.
- Added step telemetry fields to `experiment_queue.v1`: declared artifact paths,
  recursive footprint accounting, postcondition artifact accounting, and log
  byte accounting.
- Added `queue_performance_summary(...)` plus `tools/experiment_queue.py
  performance` so completed queue telemetry is queryable by resource kind and
  step.
- Preserved compatibility with the legacy `json_file_key_equals` postcondition
  spelling while normalizing new materializer rows to `json_equals`.
- Kept DQS1 retention planning in the `local_io_heavy` resource lane with longer
  timeout, avoiding CPU-lane contention for cleanup/retention work.

## Authority

This is a local execution and telemetry surface only. It does not grant score,
promotion, rank/kill, or exact-eval authority. Exact contest CPU/CUDA auth eval
and lane-dispatch claims remain required before any score or frontier claim.

## Next Integration

The next scheduler layer should consume `experiment_queue_performance_summary.v1`
to choose concurrency/resource placement, detect scorer-bound versus I/O-bound
families, and feed a Dask adapter only as an executor beneath the durable
`experiment_queue.v1` authority contract.
