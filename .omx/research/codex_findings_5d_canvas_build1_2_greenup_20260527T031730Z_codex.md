# Codex Findings - 5D Canvas BUILD-1/2 Greenup

Date UTC: 2026-05-27T03:17:30Z
Agent: Codex
Lane: lane_build_1_populate_5d_canvas_20260526

## Finding

The DROP-MANY apparatus audit at commit `1f62ac788` identified the next
critical gap as moving from isolated drop-many probes into a reusable
pair/frame/scorer/receiver/hardware canvas. The current 5D canvas tranche now
has green focused tests and operator CLI smokes for BUILD-1 population plus
BUILD-2 operation candidate generation.

This is still Tier A observability and predicted-candidate routing. It does not
claim score, rank, promotion, or exact-eval readiness.

## Landed Surface

- `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py`
- `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_populator.py`
- `src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_operation_generators.py`
- `src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_populator.py`
- `tools/apply_operation_to_5d_canvas_cli.py`
- `tools/populate_5d_canvas_cli.py`

## Verification

- `ruff check` over the 5D canvas module, populator, CLIs, and focused tests:
  passed.
- `pytest src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_operation_generators.py src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_populator.py -q`:
  `81 passed`.
- `tools/populate_5d_canvas_cli.py --list-archives --json` returned six
  archive SHA-256 values from the master-gradient ledger.
- `tools/populate_5d_canvas_cli.py --latest --no-sidecar --json` produced a
  Tier A manifest for archive
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` with
  `cells_populated=3`, `anchors_consumed=1`, and
  `anchors_skipped_non_authoritative=2`.

## Interpretation

The apparatus has moved one layer upward:

1. BUILD-1 can populate a sparse 5D canvas from canonical master-gradient
   anchors while preserving contest-axis discipline.
2. BUILD-2 can generate ranked predicted candidates for full-drop, repair,
   masked, and feathered operations without `NotImplementedError` under
   happy-path inputs.
3. BUILD-3-style `AxisDecomposition` and Catalog #323 provenance are carried on
   emitted candidates.
4. The operation CLI gives the queue/autopilot layer a stable predicted-candidate
   manifest surface.
5. The operation CLI now accepts the populated canvas schema emitted by the
   BUILD-1 populator, closing the local BUILD-1 -> BUILD-2/3 CLI handoff.

The live repository still has other unrelated submission linter/compliance and
Modal/state work in progress; those were intentionally left unstaged.

## Next Actions

1. Add the cathedral/autopilot consumer for the 5D canvas candidate manifest so
   this stops being a standalone CLI surface.
2. Feed the candidate manifest into the existing frontier-rate queue as a real
   predicted handoff target, still fail-closed for score authority.
3. Implement the audit's not-built pair/frame operation families on top of the
   canvas: replace-one, replace-many, merge-pair, reorder-pair, drop-frame,
   synthesize-frame, motion-conditional, and temporal-coherence.
4. Promote only after paired exact CPU/CUDA empirical anchors exist for a
   candidate family and the canonical routing markers are updated from
   `[predicted]` to the measured axis.
