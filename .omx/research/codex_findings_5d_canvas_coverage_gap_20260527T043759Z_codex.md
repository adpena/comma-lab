# Codex Findings: 5D Canvas Coverage Gap Audit

UTC: 2026-05-27T04:37:59Z

## Verdict

The 5D extended-operator queue was wired, executable, and green, but a sparse
canvas could still collapse into zero emitted candidates with only per-operator
blocker strings. That preserved correctness but lost acquisition signal at the
queue/DAG layer. I canonicalized the missing layer as a first-class coverage
audit that turns sparse/saturated canvases into false-authority, queue-consumable
work orders.

## Landed

- `tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_coverage`
  - strict populated-canvas JSON loader;
  - canvas coverage audit schema;
  - blocker classification for missing CPU/CUDA axes, low pair/frame coverage,
    receiver-runtime diversity gaps, no negative predicted cells, and
    single-anchor degeneracy;
  - machine-readable work orders for paired-axis anchors, pair/frame
    densification, negative-delta acquisition, and receiver-runtime diversity;
  - false-authority enforcement on audit payloads and work orders.
- `tools/audit_5d_canvas_coverage.py`
  - operator-facing JSON audit CLI.
- `pair_frame_5d_extended_operator_queue`
  - every experiment metadata row now carries the coverage audit, so empty
    candidate fanout preserves its next acquisition instructions.

## Live Current-Repo Smoke

Command family:

```bash
.venv/bin/python tools/populate_5d_canvas_cli.py --latest --output-path /tmp/... --json
.venv/bin/python tools/audit_5d_canvas_coverage.py --canvas-path /tmp/... --output /tmp/...
```

Observed current latest canvas:

- archive: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- verdict: `densification_required`
- populated cells: `3`
- feasible cells: `3`
- negative cells: `0`
- unique pairs: `1`
- unique frames: `1`
- blockers:
  - `missing_cpu_cuda_axis:contest_cpu`
  - `receiver_runtime_diversity_missing`
  - `pair_coverage_below_grouped_search_floor:0.001667<0.050000`
  - `frame_coverage_below_grouped_search_floor:0.000833<0.050000`
  - `no_negative_predicted_delta_cells`
  - `only_single_feasible_pair_anchor`
  - `only_single_feasible_frame_anchor`

Work orders emitted:

- `populate_missing_paired_cpu_cuda_axis_anchors`
- `densify_pair_coverage_for_grouped_search`
- `densify_frame_coverage_for_masked_and_feathered_search`
- `acquire_negative_delta_cells_before_operator_fanout`
- `populate_receiver_runtime_mode_diversity`

## Why This Matters

This moves the current failure mode up one layer. The next automated rate attack
should not merely run eight operators and report zero candidates; it should
immediately queue the evidence acquisition needed to make grouped search
meaningful. That is the bridge from leaf operator firing to inverse-steganalysis
acquisition: when a queue cannot find candidates, the system now says exactly
which axes of the scorer surface are under-observed.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_8_extended_operators_5d_canvas.py src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_populator.py src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py -q`
- `.venv/bin/python -m ruff check src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py src/comma_lab/scheduler/pair_frame_5d_extended_operator_queue.py tools/audit_5d_canvas_coverage.py src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py`

## Next Direct Integration

The high-EV follow-up is to let `experiment_queue.v1` consume the coverage
work orders directly as acquisition experiments. The immediate child queues
should target paired CPU/CUDA anchors first, then pair/frame densification, then
receiver-runtime diversity, then rerun the 8-operator batch. Exact eval remains
authority-gated; the coverage audit is planning-only.
