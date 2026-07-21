# REUSE MANIFEST — Task #610 wrong-levels describe sweep

UTC: 2026-07-21T22:09:16Z

## Reused exactly

- `tools/measure_per_stratum_recursive_fractal_optimal.py` from merged commit
  `8fa6581f74`, SHA-256
  `0bdde9978a5de7e9e7d9f5b1ec13a83413befaae0d034cc54b0839d94138867b`:
  mandatory fail-closed S2 audit, executed twice unchanged.
- `src/tac/v2_compose/pose_sidecar.py` and `src/tac/scorer_targets.py`:
  canonical PNTG description layout; reproduced read-only in memory.
- `src/tac/boundary_math/ego_xi_trajectory.py::PoseTargetEgoEstimator` and
  `src/tac/boundary_math/warp_real_luma_frame0.py::xi_from_pose_calibration`:
  existing target-to-`xi` mapping surfaces. PNTG targets are not Lie-algebra
  `xi`; a candidate must select exact channels/calibration and preserve the
  fixed-root lift from 600 pair twists to 601 absolute poses.
- `src/tac/optimization/predict_project_schema.py` and
  `predict_project_receiver.py`: PPCS trajectory/seed parse-back and chart
  receiver semantics.
- `src/tac/lie/_se3_numpy.py`,
  `src/tac/boundary_math/xi_spline_residual_coder.py`, and
  `xi_pose_coder.py`: identified as the existing NumPy SE(3) knot/packet
  surfaces for S1; the missing binding is a pure-knot XIP2 coder ID/format and
  shipped parser dispatch, not a new downstream `xi -> H -> RGB` realizer.
- `tools/levelset_byte_close_and_eval.py` and
  `src/tac/boundary_math/warp_real_luma_frame0.py`: existing XIP2
  store-nothing parse, homography derivation, distinct-frame warp, and real
  scorer path; current shipped parsing accepts only `delta_ar|none` layouts.
- `src/tac/boundary_math/analytic_lane_render_band.py`,
  `lane_ground_factorization.py`, `lane_headstart.py`, and
  `lane_track_and_smooth.py`: existing Lane generator, ground abstraction, and
  coherent slot front-end.
- `src/tac/boundary_math/hood_static_component.py`: self-detecting MyCar static
  generator.
- `src/tac/boundary_math/island_protection.py` and
  `movable_site_coder.py`: Movable island/site generator and identity events.
- `src/tac/shared_pmf_model.py`, `src/tac/codec_pipeline.py`, and
  `src/tac/pr91_hpm1_range_contract.py`: existing entropy/custody stack for a
  future S2 packet. The PR91 surface is diagnostic-only until full decode
  closure; it was not misrepresented as a production coder.
- S2/S4 composer and v10 receiver receipts: exact archive and receiver control
  accounting only.

## Inputs consumed read-only

- Frozen n600 cache:
  `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`,
  SHA-256 `cf8d8360...8cd6`.
- PPCS B2 seed: 884,872 B, SHA-256 `a21dde38...56b`.
- R3 component packet: 180,196 B, SHA-256 `32b41a7d...e68d`.
- S4 archive: 451,191 B, SHA-256 `d84f2fe0...1696ed`.
- BEV-v2, G2CS1, full-screw chart, c2, #596, v8, and #503 receipts/memos.
- MAIN inbox supplements at 2026-07-21T21:54:13Z and 21:59:22Z.

## Explicit non-reuse

- No new temporal predictor or delta coder; sibling
  `xi_temporal_delta_coder_574` owns that axis.
- No hand-rolled entropy coder, per-stratum class index, Fourier residual, or
  guessed compression ratio.
- No rebuild of existing Lane/MyCar/Movable carriers.
- No Task #307 contour-string build; MAIN owns the queued design.
- No scorer weights, GT tables, or target cells in a candidate archive.
- No retired archive mutation or promotion. Retired controls were
  harvest-signal-only.
- No paid dispatch, GPU launch, full-video scorer launch, or bulk scratch.

## Fresh outputs

- Full SSD audit:
  `/Volumes/VertigoDataTier/pact/evidence/wrong_levels_610_20260721/per_stratum_recall_full.json`,
  124,731 B, SHA-256 `1d4b4f6f...d4b5c`.
- Compact SSD audit:
  `/Volumes/VertigoDataTier/pact/evidence/wrong_levels_610_20260721/per_stratum_recall_compact.json`,
  23,414 B, SHA-256 `68b20a2a...b46c`.
- Repository receipt, outcome memo, DAG FEED, this manifest, and the
  canonical-equations note. No Python file was added or edited.

## MAIN review boundary

MAIN should independently verify the mandatory tool source hash and double-run
outputs; compare every numerical control to its source receipt; confirm that all
candidate and projected-archive fields remain JSON `null`; and reject any merge
that silently turns a description diagnostic into an archive byte claim.
