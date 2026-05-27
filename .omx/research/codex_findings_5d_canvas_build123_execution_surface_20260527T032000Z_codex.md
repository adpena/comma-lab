# Codex Findings — 5D Canvas BUILD-1/2/3 Execution Surface

- timestamp_utc: 2026-05-27T03:20:00Z
- agent: codex
- lane_id: lane_pair_frame_scorer_geometry_lattice_5d_canvas_build123_20260527
- authority: predicted/Tier-A planning signal only
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false

## What Changed

The 5D canvas no longer stops at the exact BUILD-1/2/3 scaffolds called out by
the drop-many apparatus audit.

- `PairFrameScorerGeometryLattice.build_lattice(...)` now derives the archive
  SHA-256 and delegates to the canonical empirical populator.
- `PairFrameScorerGeometryLattice.load_empirical_lattice(...)` delegates to the
  canonical sidecar reader.
- `generate_queue_executable_start(...)` now emits a deterministic
  `ExecutableCandidate` with Catalog #323 provenance, Catalog #356
  `AxisDecomposition`, and Tier-A false-authority routing markers.
- `query_receiver_runtime_feasibility(...)` now returns a deterministic
  conservative feasibility map instead of raising.
- `bind_pair_component_xray(...)` and
  `decompose_frame_axis_master_gradient(...)` now expose available empirical
  aggregate scorer geometry through the same lattice path instead of raising.
- `tools/apply_operation_to_5d_canvas_cli.py` now accepts the populated canvas
  schema emitted by `tools/populate_5d_canvas_cli.py`, closing the BUILD-1 ->
  BUILD-2/3 CLI handoff.

## Live Local Proof

Command family:

- `tools/populate_5d_canvas_cli.py --latest --output-path experiments/results/pair_frame_5d_canvas_build123_20260527T032000Z/populated_5d_canvas.json --json`
- `tools/apply_operation_to_5d_canvas_cli.py --canvas-input experiments/results/pair_frame_5d_canvas_build123_20260527T032000Z/populated_5d_canvas.json --operation full-drop --receiver-runtime raw_residual --output-archive experiments/results/pair_frame_5d_canvas_build123_20260527T032000Z/full_drop_raw_residual_candidates.json --json`

Observed result:

- latest archive: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- cells_populated: `3`
- anchors_consumed: `1`
- anchors_skipped_non_authoritative: `2`
- candidate_count: `0`

The zero-candidate result is intentional: the current live populated canvas only
contains aggregate RAW_RESIDUAL cells and does not predict a negative score
delta for the requested full-drop operation. This is a successful no-false-
positive path, not a score result.

## Remaining Blockers

- The 5D canvas is still Tier A observability-only until paired CPU+CUDA
  empirical anchors exist for generated operation candidates.
- The live master-gradient ledger currently yields aggregate cells only; true
  pair/frame/local cells still require per-pair component xray and per-frame
  gradient producers.
- Archive-byte materialization remains downstream. The candidate rows are
  deterministic metadata; they are not byte-closed candidate archives.

## Verification

- `ruff` on the 5D canvas core, populator, CLIs, and tests.
- `pytest src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_populator.py src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_operation_generators.py -q`
- live local CLI population + operation application listed above.
