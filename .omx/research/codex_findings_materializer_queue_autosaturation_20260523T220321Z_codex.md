# Codex Findings: Materializer Queue Autosaturation

- timestamp_utc: 2026-05-23T22:03:21Z
- agent: codex
- lane_id: codex_materializer_queue_autosaturation_20260523
- research_only: false

## Finding

`tools/build_byte_shaving_campaign_queue.py` still defaulted generated local
CPU work queues to a fixed concurrency of 2. That serialized the byte-shaving
materializer path on machines that can safely run more local CPU work in
parallel, directly opposing the current shortest-wall-clock queue/DAG goal.

## Fix Landed

The CLI now accepts `--local-cpu-concurrency auto` and makes it the default.
`auto` resolves to `os.cpu_count()` with a fail-closed minimum of 1. Explicit
positive integers remain supported, and invalid/zero values still terminate the
tool. The generated build summary records both:

- `local_cpu_concurrency`
- `local_cpu_concurrency_requested`

This preserves observability while letting the scheduler use the available
machine by default.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  - 36 passed
- `.venv/bin/python -m py_compile tools/build_byte_shaving_campaign_queue.py`
  - passed
- `.venv/bin/python -m ruff check tools/build_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py`
  - passed
- `.venv/bin/python tools/lane_maturity.py validate`
  - passed

## Remaining Wire-In

The next throughput boundary is dynamic resource control: queue builds should
learn and persist measured CPU, MLX, disk, and materializer throughput, then
cap concurrency by live resource telemetry rather than static counts alone.
