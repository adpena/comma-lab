# Codex Findings: 5D Extended Operator Refresh Autowire

UTC: 2026-05-27T04:45Z

## Verdict

The 5D extended operator apparatus is now wired into the frontier refresh and
autonomous parent queue when a populated pair-frame scorer-geometry 5D canvas is
available. The receiver remains decode-only; the new queue is local
encoder-side planning and acquisition only, with all score, promotion,
rank/kill, and exact-dispatch authority fields false.

## What Landed

- `tools/build_frontier_rate_attack_feedback_refresh.py` now accepts
  `--pair-frame-5d-canvas`, discovers populated 5D canvas manifests when not
  supplied explicitly, emits `pair_frame_5d_extended_operator_queue.json`, and
  publishes operator commands to validate/init/run it.
- `src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py` emits the
  same queue from normal refresh artifact writing when the report carries a
  valid 5D canvas path.
- `src/comma_lab/scheduler/frontier_rate_attack_feedback.py` binds
  `pair_frame_5d_extended_operator_queue` as an autonomous child queue whenever
  that artifact exists, alongside the materializer execution and repair
  waterfill queues.
- `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py`
  adds a false-authority coverage audit for sparse/biased 5D canvases and emits
  typed densification work orders rather than letting zero fanout disappear.
- `src/comma_lab/scheduler/pair_frame_5d_extended_operator_queue.py` carries
  the coverage audit inside each operator experiment metadata, so the queue row
  explains why a sparse live canvas produced no candidates.
- `tools/audit_5d_canvas_coverage.py` exposes the audit as a CLI for operator
  and queue-consumer flows.

## Live Proof

Fresh refresh output:

`experiments/results/pair_frame_5d_canvas_build123_20260527T032000Z/refresh_with_5d_extended_queue_autowire_20260527T0455Z/`

The refresh emitted:

- `pair_frame_5d_extended_operator_queue.json`
- `autonomous_chain_optimization_queue.json`
- `pair_frame_5d_canvas_coverage_audit.json`
- eight `pair_frame_5d_extended_operator_outputs/*_extended_candidates.json`
- `pair_frame_5d_extended_operator_queue_worker_result.json`

Validation:

- 5D child queue validated: 8 experiments, 8 steps.
- Autonomous parent queue validated: 3 experiments, 21 steps.
- Autonomous parent queue includes child keys:
  `operation_materializer_execution_queue`,
  `pair_frame_5d_extended_operator_queue`, and
  `repair_budget_waterfill_queue`.
- The coverage audit is now a named refresh artifact and is summarized in
  `feedback_refresh_report.json` with
  `coverage_verdict=densification_required` and
  `coverage_work_order_count=5`.
- Worker fired all eight extended operators:
  replace-one, replace-many, merge-pair, reorder-pair, drop-frame,
  synthesize-frame, motion-conditional, and temporal-coherence.
- Worker result: `steps_started=8`, `success_count=8`, `failure_count=0`,
  `stop_reason=max_steps_reached`.

## Zero-Candidate Interpretation

All eight live operator outputs emitted `candidate_count=0`. This is not a
method retirement and not a score claim. The coverage audit classifies the live
canvas as `densification_required` with these blockers:

- `missing_cpu_cuda_axis:contest_cpu`
- `receiver_runtime_diversity_missing`
- `pair_coverage_below_grouped_search_floor:0.001667<0.050000`
- `frame_coverage_below_grouped_search_floor:0.000833<0.050000`
- `no_negative_predicted_delta_cells`
- `only_single_feasible_pair_anchor`
- `only_single_feasible_frame_anchor`

The corresponding machine-readable work orders are:

- `populate_missing_paired_cpu_cuda_axis_anchors`
- `densify_pair_coverage_for_grouped_search`
- `densify_frame_coverage_for_masked_and_feathered_search`
- `acquire_negative_delta_cells_before_operator_fanout`
- `populate_receiver_runtime_mode_diversity`

## Verification

- `.venv/bin/ruff check src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py tools/audit_5d_canvas_coverage.py src/comma_lab/scheduler/pair_frame_5d_extended_operator_queue.py src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py tools/build_frontier_rate_attack_feedback_refresh.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py`
- `.venv/bin/pytest src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py src/tac/tests/test_frontier_rate_attack_feedback.py -q`
  - `59 passed in 27.11s`
- `.venv/bin/python tools/experiment_queue.py --queue .../pair_frame_5d_extended_operator_queue.json validate`
- `.venv/bin/python tools/experiment_queue.py --queue .../autonomous_chain_optimization_queue.json validate`
- `.venv/bin/python tools/experiment_queue.py --queue .../pair_frame_5d_extended_operator_queue.json run-worker --execute --max-steps 8 --max-experiments 8 --max-parallel 1 --output .../pair_frame_5d_extended_operator_queue_worker_result.json`
- `.venv/bin/python tools/audit_5d_canvas_coverage.py --canvas-path experiments/results/pair_frame_5d_canvas_build123_20260527T032000Z/populated_5d_canvas.json --output .../pair_frame_5d_canvas_coverage_audit.json`

## Next Required Automation

The immediate next step is to turn the five coverage work orders into actual
acquisition queues. The most important missing signal is negative-delta,
receiver-feasible cells across both contest axes and at least two receiver
runtime modes, because the currently observed sparse single-anchor canvas cannot
support grouped inverse-steganalysis or waterfill composition yet.
