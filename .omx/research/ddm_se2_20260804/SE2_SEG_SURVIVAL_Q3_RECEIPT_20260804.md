# SE2 Seg Survival Physics + Q3 Projection Receipt - 2026-08-04

Status: **MEASURED bounded n32, FORMULATION FOLDED for scorer spend**.

Axis: `[macOS-CPU advisory / CPU Torch SegNet+PoseNet bounded n32]`.
`score_claim=false`, `promotion_eligible=false`, `n600_run=false`.

Receipts:

- Summary JSON: `.omx/research/ddm_se2_20260804/se2_survival_q3_summary.json`
- Survival rows: `.omx/research/ddm_se2_20260804/se2_survival_rows.jsonl`
- Q3 rows: `.omx/research/ddm_se2_20260804/se2_q3_rows.jsonl`
- Script: `.omx/research/ddm_se2_20260804/measure_se2_survival_q3.py`

SHA-256:

| file | sha256 |
|---|---|
| `measure_se2_survival_q3.py` | `279179acfd545c64fe2d7b0f4d2c62aa890b0dc8cd133892ef80e2cca96f94ac` |
| `se2_survival_q3_summary.json` | `8deee5d28bc5ec61074653f9ebf0b60e62aa98097481bc19560b5313341a268b` |
| `se2_survival_rows.jsonl` | `a50f5e90642ab6ecb426284dc8006248318805d4be48334129af80be0abab9cd` |
| `se2_q3_rows.jsonl` | `6fd42764dd39971ddd952d99297095be9496a1c737abd90a50dfc705439a562d` |

## Inputs And Controls

Matched base: qo1 pair-bitpack row
`/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip`
sha256 `d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a`,
357,836 B.

Selection: n32 stratified random non-prefix by Road/Lane target count, seed
`20260804`. Pairs:

`31, 43, 62, 82, 94, 118, 147, 165, 167, 182, 185, 200, 237, 241, 247, 259, 272, 286, 288, 292, 296, 306, 327, 382, 390, 419, 473, 488, 525, 555, 560, 581`.

Denominators:

| denominator | value |
|---|---:|
| selected pairs | 32 |
| selected Road/Lane target cells | 12,407 |
| n600 Road/Lane target cells | 235,148 |
| subset scorer cells | 6,291,456 |

Controls passed:

- qo1 decoded raw matched fz4/fz1 `sub_final` raw on a two-pair, four-frame sha256 spot check.
- qo1 n32 frozen SegNet argmax matched the pu2 `cx1_argmax_n600.npy` cache exactly.
- MPS was not used.
- GT frame decode for PoseNet used the existing `frame_utils.yuv420_to_rgb` path through `decode_gt_frames`.

## Leg 1 - Survival Condition

DERIVED from `upstream/modules.py` and the measured D-structure:

SegNet reads frame 1 only, after bilinear resize from camera `874x1164` to scorer
`384x512`. The scale is greater than 2 on both axes, so each scorer-grid pixel
has a disjoint private 2x2 camera support. A private-support paint therefore
controls the local resized RGB input at the target scorer pixel after uint8
snap. That is necessary but not sufficient for a class flip.

The sufficient condition is regional: for target scorer cell `u` and target
class `c`, the painted private support and any coherent neighborhood must induce
a post-stem SegNet feature response whose target logit margin is positive at
`u`. The private support solves local input delivery; it does not isolate the
stride-2/regional receptive-field veto.

Prediction tested by Leg 2:

| mechanism | prediction |
|---|---|
| uint8 floor | Should be absent once emitted deltas change bytes. |
| R attenuation | Should be absent for private supports because D reads private 2x2 supports exactly. |
| amplitude timidity | Low deltas should fail cells that full-color same-radius paint corrects. |
| receptive-field veto | Full-color private-support failures should dominate if the SegNet regional context is the wall. |

## Leg 2 - Survival Surface

MEASURED on qo1 matched n32. Rows are formulation-scoped to private-support
prototype Road/Lane paints.

| variant | radius | max_delta | target survival | global flip delta | collateral new wrong |
|---|---:|---:|---:|---:|---:|
| `r0_delta_1` | 0 | 1 | 0.013621 | -114 | 96 |
| `r0_delta_4` | 0 | 4 | 0.046184 | -390 | 345 |
| `r0_delta_16` | 0 | 16 | 0.169340 | -1,387 | 1,474 |
| `r0_delta_32` | 0 | 32 | 0.263238 | -1,563 | 3,081 |
| `r0_delta_64` | 0 | 64 | 0.310470 | -733 | 5,075 |
| `r0_full_color` | 0 | full | 0.364310 | -22 | 6,790 |
| `r1_delta_16` | 1 | 16 | 0.369630 | +2,431 | 10,076 |
| `r1_delta_32` | 1 | 32 | 0.407028 | +9,975 | 18,845 |
| `r1_delta_64` | 1 | 64 | 0.386556 | +19,943 | 28,827 |
| `r1_full_color` | 1 | full | 0.369952 | +27,150 | 35,925 |
| `r2_delta_16` | 2 | 16 | 0.263722 | +17,113 | 24,516 |
| `r2_delta_32` | 2 | 32 | 0.261304 | +42,214 | 49,951 |
| `r2_delta_64` | 2 | 64 | 0.235915 | +69,777 | 77,378 |
| `r2_full_color` | 2 | full | 0.228258 | +86,040 | 93,623 |

Winner by whole-frame SegNet accounting: `r0_delta_32`, with target survival
0.263238 and 1,563 fewer subset flips.

Best raw target-survival row: `r1_delta_32`, survival 0.407028, but it is
collateral-dominated and worsens subset flips by 9,975.

No tested amplitude/radius cleared 0.9 survival. No tested amplitude/radius
cleared ED1's byte-closed break-even survival `0.6964303814`.

Failure taxonomy for the `r0_delta_32` winner:

| mechanism | cells |
|---|---:|
| failed target cells | 9,141 |
| uint8 floor | 0 |
| R attenuation | 0 |
| amplitude timidity | 1,587 |
| receptive-field veto | 7,554 |

Failure taxonomy for `r0_full_color`:

| mechanism | cells |
|---|---:|
| failed target cells | 7,887 |
| uint8 floor | 0 |
| R attenuation | 0 |
| receptive-field veto | 7,887 |

Verdict: cg3/se1's private-support survival wall is not explained by byte
emission, R attenuation, or low amplitude alone. In this formulation, the
binding failure is SegNet regional/receptive-field veto plus collateral from
larger coherence radii.

## Leg 3 - Q3 Pose-Null Projection

MEASURED by projecting the Leg-2 winner (`r0_delta_32`) through the exact
frame-1 yuv6-null projector with 2x2 block snapping before realizing back to
camera uint8.

| metric | value |
|---|---:|
| unprojected target survival | 0.263238 |
| Q3 projected target survival | 0.017007 |
| retained target-survival fraction | 0.064605 |
| unprojected global net flip reduction | 1,563 |
| Q3 global net flip reduction | 60 |
| retained global seg reach | 0.038388 |
| d_pose before mean | 0.000648286 |
| d_pose unprojected mean | 0.005144370 |
| d_pose Q3 projected mean | 0.000650002 |
| d_pose Q3 delta mean | +0.000001716 |
| d_pose Q3 ratio vs before | 1.002647 |

Q3 does what it should on pose for this prototype paint: it keeps d_pose almost
flat. It does not retain enough seg reach. It clears neither ED1's break-even by
target survival nor by retained global reach.

## Disposition

FOLDED as a formulation: private-support prototype Road/Lane paints on qo1 do
not reach ED1 break-even, and Q3 projection of the best net variant retains
only 3.84% of its global seg reach.

No full-n600 scorer spec was appended to `.omx/research/scorer_batch_20260804.md`
because the charter's fire condition did not clear.

No byte-closed archive was built. No upstream files were edited. No protected
files from the common contract were edited. No `/tmp` evidence path is cited.

## NEXT-IF-RESUMED

Do not relaunch this exact grid. The next useful unit is a different realizer:
a scorer-trained regional paint or solved-paint field constrained in Q3 from the
start, with the same n32 stratified qo1 base and the same ED1 break-even
arithmetic. Preserve this receipt as the negative control: local private
supports deliver input exactly, but the regional SegNet veto and Q3 seg-collapse
kill this formulation.

Own-vehicle frontier: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`
from qo1; contest pointer unmoved.
