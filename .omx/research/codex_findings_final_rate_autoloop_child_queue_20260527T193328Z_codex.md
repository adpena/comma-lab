# Codex findings: final-rate autoloop child queues

- Lane: `frontier_final_rate_attack_autoloop_child_queue_20260527`
- Scope: close the manual break between final-rate materializer execution,
  post-execute feedback refresh, and generated follow-up queues.

## What changed

- Added reusable `comma_lab.scheduler.frontier_final_rate_attack_autoloop`
  helpers for bounded post-feedback child queue execution.
- The final-rate queue builder now defaults to running generated local child
  queues after feedback refresh, with `--skip-execute-post-feedback-queues`
  available for dry custody-only runs.
- Child queue runs preserve `validate`, `init`, `run-worker`, and `observe`
  command streams as sidecar logs, while the parent custody report stores
  checksummed stream references instead of embedding large nested JSON.
- The report now records `steps_started`, `progress_made`, and
  `stalled_queue_count` so a valid queue that starts zero steps cannot be
  mistaken for advanced work.

## Empirical anchors

- Focused tests: `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`
  passed with 25 tests.
- Feedback discovery tests:
  `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q -k "materializer_feedback_discovery_accepts_final_rate_signal_harvest or materializer_feedback_default_discovery_scans_research_candidates_only"`
  passed with 2 selected tests.
- Live autoloop smoke:
  `.omx/research/frontier_final_rate_attack_autoloop_progress_smoke_20260527Tlocal/execution_report.json`
  had `failed_command_count=0`, selected 2 post-feedback child queues, advanced
  `operation_chain_compiler_queue` with 2 started steps, and explicitly marked
  `autonomous_chain_optimization_queue` as stalled because the worker started
  0 steps while 15 steps remained queued.
- Broader all-executable run:
  `.omx/research/frontier_final_rate_attack_all_exec_autoloop_20260527Tlocal/execution_report.json`
  had `failed_command_count=0`, 9 materializer observations, 0 rate-positive
  observations, and automatic child queue custody.

## Remaining blocker

The final-rate materializer layer is now queue-owned through first follow-up
execution, but the current frontier archive did not expose rate-positive bytes
under archive repack, member recompress, or header elision in this bounded run.
The next score-moving work is upstream of these leaves: compile combined
pre-entropy transforms, byte-range/section recoders, targeted repair budget
waterfill, and substrate/receiver-specific materializers into executable child
queues that start work rather than only validate planning state.

