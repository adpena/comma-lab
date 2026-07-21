# Build spec — full-screw advection plus chart-level coefficient pricing

Authority: delegated `advected_screw6_chartlevel` arm, 2026-07-21. This is a
build-and-measure spec, not a score or promotion artifact. The contest-CPU
pointer remains `0.19108 [contest-CPU]`.

## Objective

Replace the predecessor's planar PPCS embedding with the complete stored
six-coordinate PoseNet motion, mapped through the existing `tac.lie` SE(3)
exp/log and openpilot ground-plane homography. Price the remaining difference
to the deterministic #549 target as receiver-consumed chart coefficients, not
as a literal RGB pixel exception. Measure n16 first and admit n64 only when the
n16 pose-and-rate gate passes; never run n600 unless the n64 gate passes.

## Execution ownership

This already-delegated Codex arm owns implementation and local CPU-Torch
measurement in its isolated worktree. No nested delegation is needed. Bulk
stages and checkpoints live under
`/Volumes/VertigoDataTier/pact/evidence/advected_screw6_20260721/`.

## Reuse and file plan

- Extend `src/tac/optimization/predict_project_receiver.py` with an additive
  full-screw xi decoder and strict chart-RGB coefficient packet/apply surface.
  Keep the planar predecessor API unchanged.
- Add `tools/measure_advected_screw6_chartlevel.py`, composing the predecessor's
  deterministic #549 target reconstruction, factor-2 realization, CPU-Torch
  scorer custody, chunked stage protocol, and aggregation helpers.
- Add focused tests in `src/tac/tests/test_advected_screw6_chartlevel.py`.
- Land a receipt JSON, DAG FEED, reuse manifest, and dated measurement memo.

## Failed-search justification for new code

The repository has strict localized boundary-coordinate packets and PPCS chart
sections, but no strict packet that binds per-pair RGB coefficients directly to
the decoded five-class scene chart. Literal pixel exceptions therefore cannot
be replaced at the required level by an existing callable. The new packet is
narrow: fixed five-class x RGB int8 coefficients plus one float16 scale per
pair, canonical header, SHA-256, CRC, strict parse-back, and Brotli-11 terminal
accounting. The chart raster and decoder algorithm are already-counted/free;
only coefficients and scales are charged.

## Sealed measurement choices and LawRefs

- Translation scale `0.16`: MEASURED pose-carry calibration anchor W7,
  `ego_motion_cumulative_se3_bspline_v1`.
- Rotation scale `1.0`: DERIVED identity map from stored rotation coordinates
  into `tac.lie` radians under the same equation/convention.
- Ground pitch: hash-pinned G1 measurement LawRef already loaded by
  `predictor_upgrade_xi_chart.load_g1_worldsheet_motion`.
- Rate exchange: canonical `realization_breakeven_bytes_v1`, approximately
  `6.6586e-7` score units/byte.
- Seed 1234; hard CPU Torch; `[macOS-CPU advisory]` only.

## Acceptance and gates

1. Full-screw custody proves all six stored source coordinates feed xi, reports
   per-coordinate nonzero counts and xi-norm quantiles, and charges zero added
   video-derived motion bytes.
2. Packet parse/re-encode and Brotli decompress/parse-back are byte-identical;
   malformed/trailing payloads refuse.
3. n16 measures static, full-screw, solved target, static+chart coefficients,
   and full-screw+chart coefficients through native CPU PoseNet and SegNet.
4. n64 runs only if n16 shows full-screw d_pose below static and the composed
   full-screw chart coefficient stream is no larger than the static chart
   coefficient stream. n600 remains governed by the same two-axis n64 gate and
   the standing n600 OOM law.
5. Every negative is scoped to this stored-pose calibration, chart-offset
   coefficient family, prefix, clip, and advisory hardware axis.

## Do not touch

Main, other worktrees, live run directories, frontier pointers, quarantined
archives, provider/GPU surfaces, and existing planar receipts. MAIN must review
the isolated branch diff before any merge or promotion.
