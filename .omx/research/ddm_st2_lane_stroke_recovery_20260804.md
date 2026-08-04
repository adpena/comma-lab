---
arm: ddm_st2
title: Lane stroke-production recovery with SMEVR race
utc: 2026-08-04
axis: "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
score_claim: false
promotion_eligible: false
pointer_moved: false
receipt: ".omx/research/ddm_st2_lane_stroke_recovery_20260804_receipt.json"
script: "experiments/ddm_st2_lane_stroke_recovery.py"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_st2 - Lane stroke-production recovery with SMEVR race

## Answer First

The cg1r named gap is CLOSED as a bounded real-path measurement for this
stroke-production realizer, not as a production win.

Minimum surviving grid cell:

| width extra scorer px | amplitude blend | target recovered | target recovery | target S recovered |
|---:|---:|---:|---:|---:|
| 0.0 | 0.75 | 55 / 2505 flips | 2.1956% | 0.000874 S |

Interpretation: `width=0.0` means the erased GT Lane support only, with the
camera-res bilinear AA edge. It is not a zero-physical-width stroke. Since
amplitude `0.50` at the same width failed and `0.75` survived, the continuous
minimum is bounded as `0.50 < a_min <= 0.75` for this exact realizer and sample.
No narrower-than-target support was tested.

Production verdict: FOLD the fixed prototype composite stroke. Every measured
cell is net-worse on whole-frame d_seg, and the minimum surviving cell is much
worse after rate and pose: sample seg net `+0.002686 S`, SMEVR rate
`+0.000927 S`, pose `+0.121850 S`, total `+0.125463 S` worse. The target-only
n600 recovered value is only INFERRED: `0.001070 S` from sample recovery fraction
times the measured n600 target pool.

SMEVR verdict: the sparse stroke payload is on SMEVR's side of the race. For the
minimum-survival row, r7-SMEVR is `1392 B`, Brotli-q11 is `2167 B`, and LZMA1 is
`2741 B`; even raw packed nibbles under Brotli-q11 are `2021 B`, still larger
than SMEVR.

The own-vehicle frontier is unchanged: `S = 0.7910689 @ 353,805 B`
`[macOS-CPU advisory]`.

## Source Reads

- `PROGRAM.md`: evidence axes, proxy/advisory boundary, mutation frontier.
- `.omx/research/ddm_cg1_force_class_edge_ledger_20260803.jsonl`: cg1r rows
  `struct.depth_parity_directed`, `tr1.lane.annihilate`, and
  `as1.lane_presence_gap`. The explicit gap is an unmeasured Lane
  presence/existence carrier; Lane has no interior, and the defect is
  volumetric/verb-level rather than cheaper per-flip pricing.
- `.omx/research/ddm_lp1_lane_program_20260803.md`: task #934 Leg B is
  corrected to solve-from-frozen-head and re-render; compositing/painting was
  already measured net-negative.
- `.omx/research/ddm_p4x_lane_existence_birth_matrix_20260803.md` and
  `.omx/research/ddm_hv2_two_week_harvest_20260803.md`: #920 Lane x
  ANNIHILATE existence primitive, seeded random/non-prefix rule, and
  component-level existence force.
- `.omx/research/ddm_rl1_roadlane_interface_price_20260803.md`,
  `.omx/research/ddm_et1_eta_on_the_priced_band_20260803.md`, and
  `.omx/research/ddm_cg3_counted_gt_recovery_20260804.md`: #939 is not a repo
  task id in the current `canonical_task_status.jsonl`; the Lane-crop object is
  rl1's description-price row and carries no realization efficiency.
- `experiments/ddm_qa92_carrier_discriminator.py`: reused the erased
  super-nucleus Lane target harness and constants. Class order is
  `[Road,Lane,Undrivable,Movable,MyCar]`; Lane is index `1`.
- `experiments/ddm_sq1_eta_seg_realization.py`: reused the canonical
  `frame_utils.yuv420_to_rgb` GT decode and frozen CPU scorer path.
- `experiments/ddm_r7_token_coder.py`: reused the SMEVR/Brotli/LZMA payload
  race and round-trip checks.

Structured search of `.omx/state/canonical_task_status.jsonl` returned no exact
`task_id == 934`, `939`, or `920` rows in the current snapshot. That boundary is
consistent with hv2/cg3's m89 two-store warning, so this memo cites content
sources rather than inventing missing task rows.

## Method

Command that produced the survival curve:

```
.venv/bin/python experiments/ddm_st2_lane_stroke_recovery.py \
  --limit 32 \
  --amplitudes 0.125,0.25,0.375,0.5,0.75,1.0 \
  --widths 0,0.5,1.0,1.5,2.0,3.0 \
  --seg-batch 4 \
  --threads 6
```

Additional n32 pose postprocess was computed with the same script helpers for
the minimum-survival row and stored in the same receipt. No full n600 scorer job
was launched.

Definitions:

- Target `T`: QA92 erased super-nucleus Lane components, 8-connected, component
  size `>5 px`, erased iff `<50%` of its GT Lane pixels are classified Lane in
  the base pass.
- Stroke: camera-res pre-R AA alpha from scorer-grid target support, then
  `edited=(1-a*alpha)*decoded + a*alpha*proto`.
- Prototype: frozen-head solved Lane RGB `[77.43, 86.71, 118.53]`.
- Width: extra scorer-pixel soft expansion beyond target support. Width `0`
  paints the target support only.
- GT decode: only through `frame_utils.yuv420_to_rgb`.

Positive controls passed: recomputed decoded argmax matched the cached cx1
argmax on `32/32` selected pairs, and recomputed GT argmax matched the cached GT
argmax on `32/32`.

Denominators:

| denominator | value |
|---|---:|
| measured pairs | 32 |
| selected-pair total-flip ratio vs n600 | 0.997329 |
| sample target flips | 2505 |
| sample target pool | 0.0398159 S |
| n600 cached target flips | 57514 |
| n600 cached target pool | 0.0487552 S |
| sample target-pool ratio vs n600 | 0.816649 |
| S per byte | 6.658589531e-7 |

Selected pairs: `[0,20,32,48,115,154,170,179,180,195,196,211,214,242,261,288,357,365,370,394,400,420,433,439,471,474,485,501,504,514,521,533]`.

## Survival Curve

Entries are target recovery fraction in percent. Negative values mean the stroke
increased target-set errors.

| width \ amp | 0.125 | 0.25 | 0.375 | 0.5 | 0.75 | 1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | -2.36 | -2.00 | -1.68 | -0.84 | +2.20 | +2.95 |
| 0.5 | -3.23 | -4.39 | -4.51 | -4.67 | -4.35 | -2.87 |
| 1.0 | -4.39 | -6.71 | -7.98 | -9.58 | -10.18 | -10.90 |
| 1.5 | -5.75 | -10.58 | -13.73 | -17.92 | -23.03 | -22.91 |
| 2.0 | -6.39 | -13.09 | -17.21 | -21.76 | -26.19 | -25.99 |
| 3.0 | -4.75 | -11.46 | -17.45 | -22.44 | -26.71 | -26.71 |

No cell has negative whole-frame d_seg delta. The least-bad whole-frame row is
not a survival row: width `0.5`, amplitude `0.125`, target recovery `-81`
flips, seg net `+0.000858 S` worse. So the survival threshold and the score
threshold diverge: enough amplitude to recover any target Lane flips already
creates larger collateral.

## Codec Race

Minimum-survival payload (`width=0`, `amp=0.75`, n32 scorer-grid alpha codes):

| coder | bytes | rate S |
|---|---:|---:|
| SMEVR r7 framed | 1392 | 0.0009269 |
| Brotli-q11 r7 framed | 2167 | 0.0014429 |
| LZMA1 r7 framed | 2741 | 0.0018251 |
| raw nibbles + Brotli-q11 | 2021 | control |
| raw nibbles + LZMA1-x9e | 2595 | control |

Best-net payload (`width=0.5`, `amp=0.125`) also goes SMEVR: `2872 B` vs
Brotli `3802 B` vs LZMA1 `4263 B`. This stroke payload is therefore a sparse
phase/stroke stream, not the token-bulk surface where SMEVR lost in cg3.

## S Price

For the minimum-survival row:

| component | S delta, n32 sample |
|---|---:|
| target recovered | -0.000874 |
| off-target collateral | +0.003560 |
| whole-frame seg net | +0.002686 |
| SMEVR rate | +0.000927 |
| pose delta | +0.121850 |
| seg + rate | +0.003613 |
| seg + rate + pose | +0.125463 |

The n600 target-only recovered amount is INFERRED, not measured through n600
scorer: `0.0487552 * 0.0219561 = 0.001070 S`. There is no n600 collateral,
n600 pose, or exact-eval claim here.

## Verdict Scope

MEASURED:

- n32 bounded survival curve through real `R + uint8 + frozen CPU SegNet`.
- n32 matched pose collateral for the minimum-survival row.
- n32 SMEVR/Brotli/LZMA payload race with round-trip checks.
- n600 target pool from cached argmax arrays only.

DERIVED:

- S-per-byte rate arithmetic.
- Sample S decomposition from measured flip counts, byte counts, and pose MSE.

INFERRED:

- n600 target-only recovered S from sample recovery fraction times the cached
  n600 target pool.

NOT CLAIMED:

- No contest-CPU/CUDA exact eval.
- No full n600 scorer job.
- No MPS authority.
- No frontier movement.
- No kill of Lane birth, #920, #934 Leg B, or #939/rl1 description pricing.

## Disposition

FOLD: fixed prototype camera-composite stroke production. It closes the cg1r
minimum-stroke measurement gap but is not a score-improving realizer.

FIRE: #934 Leg B / #920 should stay on the solve-from-frozen-head, re-rendered
existence-hinge path. The next valid fire is a bounded random or stratified
sample with matched pose, not a pixel composite.

QUEUE: rerun the SMEVR race on any future token-space or re-rendered Lane
existence payload that becomes net-positive on a bounded sample. A full n600
scorer row should wait until a bounded sample is net-positive and pose-priced.

STATE-THE-BOUNDARIES: This memo measures a simple stroke realizer, not the Lane
existence family. It uses n32 bounded advisory scoring, not contest authority.
The target pool is measured at n600 from cached argmax arrays, but the realizer
effect is not.

own-vehicle frontier S = 0.7910689 @ 353,805 B [macOS-CPU advisory] - UNMOVED
