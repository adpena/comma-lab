# Targeted Component Paired Reference Response - Codex Findings - 2026-05-26T05:53Z

## Verdict

Receiver-closed rate wins now flow into targeted SegNet/PoseNet correction work
as a paired source-reference measurement problem instead of an isolated byte
shave. The queue remains false-authority: local CPU and MLX component responses
can recommend local acquisition, but cannot claim score, promote, rank/kill, or
dispatch exact eval.

## Landed Integration

- `frontier_rate_attack_feedback.py` now carries source archive/runtime custody
  from receiver-closed submission closure/source queues into the receiver-closed
  correction budget, targeted correction acquisition rows, work orders, queue
  metadata, and response harvest rows.
- Targeted component queues now emit one shared
  `local_cpu_reference_advisory` per candidate when a real source archive and
  source inflate path are available, then pass that paired reference into every
  grouped harvest row for that candidate.
- The paired local CPU harvest path derives SegNet/PoseNet deltas from candidate
  versus receiver-closed source advisory outputs when the candidate advisory does
  not already contain explicit component deltas. Receiver-closed source
  reference mode keeps `correction_rate_delta_score_units = 0.0` so the saved
  rate credit is not double-counted as correction spend.
- Source-reference runtime resolution is fail-closed against candidate-runtime
  confusion: closure-local `source_runtime_dir` is not enough to unlock a source
  reference. The resolver requires explicit source/reference runtime fields from
  the source queue, and the regression fixture poisons the closure with a
  candidate runtime to catch drift.
- `tools/harvest_frontier_targeted_component_correction_response.py` accepts
  `--reference-local-cpu-advisory` and `--reference-role`, preserving the same
  false-authority semantics in CLI and queue-owned harvest flows.

## Live Queue Proof

Generated:
`.omx/research/frontier_rate_attack_feedback_refresh_20260526T055307Z_paired_component_reference/`

Observed summary:

- `operation_count`: 30
- `followup_signal_operation_count`: 11
- `queue_executable_operation_count`: 4
- `receiver_closed_saved_bytes_total`: 414
- `targeted_component_correction_acquisition.row_count`: 10
- Top operation families include DFL1+merge+header-elide chaining, packet member
  merge, pair-frame geometry requests, registered multisurface materializer
  program, drop-many beam/waterfill, learned multi-drop, DFL1, and inverse-scorer
  null-direction masked variants.

The generated targeted queue validates and carries a paired reference for the
packet-member merge candidate:

- reference archive: `submissions/robust_current/archive_correct.zip`
- reference inflate: `submissions/robust_current/inflate.sh`
- shared reference output:
  `experiments/results/frontier_targeted_component_correction/frontier_feedback_component_correction/packet_member_merge_cbe7d79124ba/shared_component_response/reference_local_cpu_advisory.json`

## Verification

- `ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py tools/harvest_frontier_targeted_component_correction_response.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` -> 28 passed
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_rate_attack_feedback_refresh_20260526T055307Z_paired_component_reference/targeted_component_correction_queue.json validate` -> valid, 2 experiments, 21 steps
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_rate_attack_feedback_refresh_20260526T055307Z_paired_component_reference/receiver_repair_queue.json validate` -> valid, 4 experiments, 8 steps
- `.venv/bin/python tools/lane_maturity.py validate` -> 1379 lane(s) validated cleanly
- Review tracker policy clean for the three touched Python files

## Remaining Gap

This closes a queue-owned measurement bridge for spending freed rate budget on
targeted corrections. It does not itself execute the long local CPU reference
and candidate advisory runs, nor does it make MLX/local advisory evidence a
score or dispatch authority. The next high-EV step is to run the generated
targeted queue far enough to harvest paired component deltas, then let accepted
responses feed the materialization queue for concrete correction operators.

