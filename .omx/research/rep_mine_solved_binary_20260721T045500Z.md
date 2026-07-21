# Solved-binary representation mine: constraints, cells, and causal dynamics

**UTC:** 2026-07-21T04:55:00Z
**Lane:** `lane_rep_mine_solved_binary_20260721` · task #596
**Authority:** delegated prompt SHA-256 `061e6b8be93c184fcad844dccb54c1abe754f16538214cc50801400f62f59466`
**Verdict axis:** **[macOS-CPU advisory]**, representation-only, non-promotable
**Pointer:** `0.1910828242 [contest-CPU]` **UNMOVED**
**Score/candidate claim:** none

## Outcome

The solved object does not natively want dense camera values. It wants a **causal constraint and
cell-complex description**: exact evaluator inequalities, a discrete Morse-Smale-style partition
skeleton and event vineyard, causal position/adjacency/xi context, then only the RGB/Pose realization
residual that those constraints fail to regenerate.

The first exact reconstructive **payload estimate** I can defend today is still large:
**1,571,792,105 B** = 1,571,792,092 B measured owned-camera seed + 13 B for 50 two-bit fill selectors.
That is 145,380,636 B (8.466279%) below the exact M2 archive before a new manifest/container is
counted, and it remains 7,269x the 216,222 B box. It is not an emitted archive size; the existing
1,717,172,741 B M2 archive remains the valid full-archive upper bound.
At the other end, the strongest measured context model has an **optimistic ideal length of
222,447.03 B**, already 6,225.03 B above the box before model/header, Pose side information, RGB
realization, receiver, or archive framing. That figure is not an information-theoretic lower bound;
a better model may be smaller. The requested “true information bytes” is therefore **not identified**.
The evaluated-family envelope runs from that incomplete 222,447 B model to the 1,571,792,105 B exact
reconstructive payload estimate, but it is not a mathematical interval on Kolmogorov complexity and
the latter still owes new archive framing.

Subtracting the 80.67% resize-kernel dimension, 31.11% logit-gauge energy, and label entropy from one
another would be fake: they are overlapping invariances on different objects, and compressor
correlations are not additive.

## Custody and execution

| object | bytes / geometry | SHA-256 | authority |
|---|---:|---|---|
| M2 exact archive | 1,717,172,741 B; d_seg=0, d_pose=0 | `0fee1b74...cfb6b` | existing `[contest-CPU]` receipt |
| inflated camera raw | 3,662,409,600 B; `(1200,874,1164,3)` uint8 | `a7192f93...dd5` | source-derived exact M2 bytes |
| `gt_n600.npz` | 5,078,017,610 B | `cf8d8360...8cd6` | ZIP_STORED mmap planes |
| teacher logits | 1,179,648,000 B; `(600,5,384,512)` fp16 | `41d3ef53...3b52` | source manifest has 24 ppm tie caveat |

The full pass ran in **1,482.818 s**, with atomic 12-pair checkpoints and hash-bound `--resume`; the
final label-schema/custody rerun took 36.981 s including full input re-hashing. Bulk evidence is preserved at
`/Volumes/VertigoDataTier/pact/evidence/rep_mine_20260721/full_n600/receipt.json`. The durable,
reviewable implementation is `tools/measure_rep_mine_solved_binary.py`.

## M1 — byte-mass algebra

### Algebraic decomposition

| component | full n600 measurement | byte meaning |
|---|---:|---|
| full resize-kernel dimension | 80.6742315% **DERIVED** | degrees of freedom, not savings |
| implemented exact blind-coordinate fraction | 22.6969261% **MEASURED** | coordinates the current integer fill does not ship |
| actual M2 camera energy in ker(A) | **45.1667835% MEASURED** | signal energy removable without changing resize numerators; not a code length |
| actual M2 camera energy in range(A) | 54.8332165% **MEASURED** | orthogonal complement; closure error `5.68e-9` |
| teacher-logit per-pixel gauge energy | **31.1071138% MEASURED** | different object; cannot be subtracted from camera bytes |
| teacher-logit quotient energy | 68.8928862% **MEASURED** | empirical centered rank is exactly 4 at relative `1e-10` |
| fifth centered logit direction | **0 energy MEASURED** | rank-4 law confirmed on this tensor |

The old #519 ~52% gauge result does not transfer: on these source-derived teacher logits the measured
gauge energy is 31.11%.

### Direct code-length experiments

All rows sum independently compressed 12-pair chunks; headers/framing are excluded.

| representation | measured bytes | exactness / interpretation |
|---|---:|---|
| raw camera, zlib-6 | 1,801,803,583 | exact raw |
| owned camera seed, zlib-6 | 1,667,314,419 | exact with generic fill + selector |
| owned camera seed, Brotli-5 | **1,571,792,092** | best exact reconstructive payload seed; new archive framing owed |
| exact resize numerators u32, zlib-6 | 2,161,746,454 | exact scorer-plane constraints, worse |
| rounded RGB scorer plane u8, zlib-6 | 455,421,684 | loses fractional numerator information |
| fractional numerator residual i32, zlib-6 | 1,722,576,181 | exact residual beyond nearest rounded plane |
| rounded + fractional, separate streams | 2,177,997,865 | separation destroys cross-stream correlation |
| camera delta from rounded canonical fill i16 | 1,747,002,483 | exact, still huge |

The 50 fill choices cost 100 bits if reproducing the same M2 raw. For evaluator equivalence, a fixed
generic canonical fill could eliminate those selector bits, but only after exact RGB/Pose replay shows
that the alternative fill remains in the frozen cells.

### Where the exact f1 fractional residual lives

These are 600 f1 planes, one independently compressed stream per row.

| class | cells | values | zlib-6 bytes | bytes/pair |
|---|---:|---:|---:|---:|
| Road | 27,407,046 | 82,221,138 | 178,307,718 | 297,179.5 |
| Lane | 690,639 | 2,071,917 | 5,617,887 | 9,363.1 |
| Undrivable | 58,413,281 | 175,239,843 | 441,268,981 | 735,448.3 |
| Movable | 1,460,325 | 4,380,975 | 11,644,451 | 19,407.4 |
| MyCar | 29,993,509 | 89,980,527 | 224,671,774 | 374,453.0 |

Boundary cells cost 27,567,290 B versus 833,879,076 B for interior cells. This does **not** mean the
boundary is unimportant; it means a boundary-only exact-value stream cannot replace the bulk RGB/Pose
information. Almost all cells are high-margin (`>=1e-2`), and their separately coded residual costs
860,944,504 B.

## M2 — value domain and spatial structure

The dense logit stream is not low-cardinality: the five planes contain 31,505 / 31,795 / 28,933 /
27,203 / 31,333 distinct fp16 values. Raw logits compress to 841,764,897 B. Four class-difference
coordinates compress to 693,329,263 B but produce **4,367 argmax mismatches out of 117,964,800** after
fp16 reconstruction, so this is not an exact partition code. Separately adding the 260,364,455 B f32
gauge mean is worse than raw because it destroys correlation.

On the digital argmax complex:

- 21,304 4-neighbor connected cells exist across the 600 rasters; 1,762 are singletons.
- A constant quotient vector per cell explains **83.1564%** of centered logit energy, but the within-cell
  remainder is still **16.8436%**, so cell constants are a model, not an exact code.
- Horizontal and vertical equal-label adjacency are **99.7115%** and **98.9119%**.
- Mean horizontal run lengths by Road/Lane/Undrivable/Movable/MyCar are
  139.15 / 6.18 / 369.26 / 37.27 / 468.41 cells. Lane is the short-run topology outlier.

The operator’s discrete Morse-Smale interpretation is useful, but the measured object here is a
digital connected-cell complex. A classical certificate still needs critical points, separatrix arcs,
incidence, and persistence order; I do not rename connected components into that missing proof.

## M3 — temporal dynamics, jitter, and the vineyard

### Label dynamics

| model | ideal bits/cell | ideal bytes | finite coder/model/header? |
|---|---:|---:|---|
| unconditional labels | 1.615508 | 23,821,631.80 | no |
| previous pair's f1 conditioned | 0.083465 | 1,228,696.81 | no |
| nearest-target pose-proxy warp alone | 0.096174 | 1,418,143.34 | no |
| 12 row bins + left/up + pose proxy | **0.015086** | **222,447.03** | no |

This is a **600-f1 pair-index proxy**, not exact sequential xi transport: `f1[p-1]` is two video
frames earlier, while `gt_poses[p]` is exact for `f0[p] -> f1[p]` and only a nearest-target proxy for
`f1[p-1] -> f1[p]`. The pose proxy is 15.23% worse than direct previous-pair-f1 prediction. The
direct change mask covers 1.24564% of cells and zlib-compresses to 884,460 B; the pose proxy changes
2.89547% and costs 1,128,863 B. Model tables and Pose side information are excluded.

For the actual exact fractional-numerator remainder, the measured previous-pair f1 delta stream is
964,201,392 B. An xi-advected exact residual stream was **not measured**; the label proxy is not a
substitute. G1 remains the authority for the real 1,199-transition boundary dynamics.

G1 remains smooth across the pair boundary: within/cross medians are 0.27921/0.27985 px and `>4 px`
events are 8.3977%/8.2497%. The tail is therefore a stratum-dependent event process, not a reset and
not permission to call it noise.

### Jitter ladder

| rung | representation | measured evidence | exact status |
|---|---|---|---|
| R0 | signed normal offsets + birth/death | G1 exact census exists | byte length **UNMEASURED** because the durable histogram bins magnitude and drops sign/exact offset |
| R1 | phase-conditioned residual | #425: 13,222 residuals, 10,682 B, 0.06568 px RMSE | bytes measured on another cached vehicle; through-R recovered d_seg owed |
| R2 | causal ground appearance chart + xi response + true exceptions | design grounded by R0/R1 | **FORMALIZATION_PENDING**; must count chart parameters and exact exceptions |

This is the required framing: jitter is solvable response. But no 66% explanatory statistic or
cross-vehicle 11.3x dash-code gain is promoted into an M2 byte estimate without an equal-fidelity,
receiver-closed replay.

### Morse-Smale / vineyard stream

For the 17,926 G3 flip identities, the spatial-temporal target-symbol ideal is **2,724.873 B**.
Enumerating the site set by colex costs **31,653.132 B ideal**, for **34,378.005 B** total before
headers/model/finite coder. This is an event-sidecar floor, not a full witness description.

A real spacetime vineyard would ship an initial critical-point/separatrix graph, persistence order,
and birth/death/split/merge lifecycle events. Those objects were not measured here. DMTz-style
posthoc topology edits remain a negative formulation for this lane: they describe corrections after a
lossy field, not the source-native constraint/cell representation requested by the operator.

## M4 — ranked native reformulations

“Free” below passes the generalizability test: **given a different dashcam seed, does the same code
remain correct?** If yes, it is interpreter/procedure. If not, its tables/coefficients are counted.
Every decoder wall clock remains **UNMEASURED** until an emitted stream exists.

| rank | structure · ops · transform · primitive · abstraction | measured byte anchor | counted seed | free generic interpreter | consumer / verdict |
|---:|---|---:|---|---|---|
| 1 | causal table · constrained inverse/project · values→constraints · inequality residual · predict-project witness | **222,447.03 optimistic ideal** | label stream, fixed video tables, Pose constraints, true exceptions | adaptive causal model from decoded history; deterministic projection/tie break | S2 #595/v10 · best measured context model, receiver absent |
| 2 | critical points+separatrices+winged adjacency · cancel/advect/event · raster→vineyard · cell/arc/event · spacetime MS complex | **34,378.00 ideal sidecar** | initial graph, persistence/order if video-derived, charts, lifecycle exceptions | generic discrete-Morse cancellation and vineyard update | G3/#595 · promising event grammar, not full description |
| 3 | causal context automaton · arithmetic code · labels→symbols · class · probability/channel split | **255,288 exact partition** | partition and any fixed learned tables | arithmetic coder and decoded-history adaptation | #557 · RGB/Pose receiver absent |
| 4 | colex subset rank · rank/unrank · sites→integer · subset rank · enumerative set | **31,653.13 ideal** | rank, k, universe binding, target symbols | generic combinatorial ranker | G3/r1b5 · headers excluded |
| 5 | Fisher prefix+exceptions · regenerate/patch · sites→ranking · prefix · sensitivity set | 31,484.71 ideal | exceptions; current 2,225,887 B 0.mkv-derived ranking | ranking only if recomputed from decoded seed | r1b5 · 168 B ideal gain erased by custody |
| 6 | appearance chart · predict/render/patch · jitter→phase · coefficient/exception · causal response | **UNMEASURED** | video chart params and exact residual | generic chart; params adaptive from decoded seed | #425/flicker · formalization pending |
| 7 | quotient/range basis · project/fill · remove gauge/null · coefficient · equivalence-class format | **1,571,792,105 exact payload estimate** | owned camera coordinates | #580 projector, canonical fill, gauge normalization | M2 successor · new archive framing owed and far over box |
| 8 | rank-4 tropical charts · max-plus/tie · logits→decision complex · affine inequality · tropical partition | rank-4 and 21,304-cell anchors; bytes unmeasured | chart coefficients, incidence, exceptions | generic max-plus evaluator | v10 · formalization pending |
| 9 | reduced integer basis · LLL/reconstruct · affine fiber→short coordinates · integer coefficient · lattice point | **UNMEASURED** | basis/generator if video-derived and coefficients | generic LLL with fixed convention | #586 successor · structural only |
| 10 | Freeman turns+winged edges · trace/rasterize · raster→contour graph · direction/edge/face · topology+geometry | **228,764 optimistic** | starts, turns, junctions, cell values, residual | generic tracer/rasterizer | MS codec · estimate, not emitted |

The research basis is primary: tropical ReLU maps and decision complexes
([Zhang et al.](https://arxiv.org/abs/1805.07091)); Morse-Smale complexes
([Edelsbrunner et al.](https://doi.org/10.1145/777792.777846)); persistence vineyards
([Cohen-Steiner et al.](https://doi.org/10.1145/1137856.1137877)); enumerative source coding
([Cover](https://doi.org/10.1109/TIT.1973.1054929)); arithmetic coding
([Witten, Neal, Cleary](https://doi.org/10.1145/214762.214771)); chain codes
([Freeman](https://doi.org/10.1109/TEC.1961.5219197)); winged edges
([Baumgart](https://www.cs.jhu.edu/~misha/Spring25/Readings/Baumgart75.pdf)); MDL
([Rissanen](https://doi.org/10.1016/0005-1098(78)90005-5)); convex projection
([Youla and Webb](https://doi.org/10.1109/TMI.1982.4307555)); LLL
([Lenstra, Lenstra, Lovasz](https://doi.org/10.1007/BF01457454)); and DMSC preservation
([Li et al.](https://arxiv.org/abs/2409.17346)). These sources motivate algorithms; none supplies an
M2 byte claim.

## Global waterfill and exact interaction blocker

At `lambda*=25/37,545,489 = 6.6585895e-7 score/B`, the only measured isolated sparse-flip point is
the r2b stream: 27,213 B recovers 0.0012332317 score, or `4.531774e-8 score/B` = **6.806% of lambda**.
Therefore the isolated marginal is eaten: 1,585 fixed flips out of 16,751 evaluated decisions are not
worth their bytes at the global price.

No global allocation is claimed. Partition, MS events, xi, jitter, exact values, and sparse flips do
not yet have equal-fidelity rate-distortion curves on one receiver. Every off-diagonal interaction is:

`NOT_MEASURED_RECEIVER_COMPOSITION_ABSENT`.

The settling protocol is the #30/#535 commutator-aware joint rollout: both application orders for
each stream pair, exact bytes/d_seg/d_pose, and the interaction residual. This lane does not fork #535
or fabricate a matrix from isolated points.

## M5 — box verdict

| surface | bytes | status |
|---|---:|---|
| target box | 216,222 | governing target |
| position+adjacency+pose-proxy label entropy | **222,447.03** | optimistic ideal; model/header/Pose side information absent; +6,225.03 B over box |
| exact CPC1 partition | 255,288 | exact partition, no realization |
| optimistic contour + xi | 235,974 | not emitted/equivalent-rate/receiver-closed |
| G3 flip symbols | 2,724.873 | ideal symbols, sites excluded |
| G3 sites + symbols | 34,378.005 | event sidecar only |
| exact reconstructive M2 payload estimate | 1,571,792,105 | exact raw reconstruction; new archive framing owed |
| existing exact M2 archive | 1,717,172,741 | valid full-archive upper bound |

**Narrow verdict:** the representation family remains open, but a full n600 sub-216,222 B description
is **NOT PROVEN**, and the 222,447 B model does not prove impossibility. The next build should target the rank-1 constraint seed composed with rank-2
cell/vineyard dynamics, because that is the only route suggested by the measurements that can erase
both dense label cost and dense value cost. Its admission gate is an emitted counted seed whose generic
decoder reconstructs RGB/Pose through R, stays under 30 minutes, and beats the box on exact archive
bytes.

## Canonicalization and triality

Four empirical anchors are `FORMALIZATION_PENDING`: M2 kernel-energy fraction, teacher-logit gauge
fraction, context entropy, and colex ideal length. They are single-object/ideal-code measurements, not
general laws, so registering them as universal evaluators would be false authority.

- **DSL leg:** no new launch lever; the eventual receiver needs typed modes for constraint seed, MS
  vineyard, causal chart, and exact exception stream.
- **DAG leg:** emitted in `rep_mine_solved_binary_DAG_FEED_20260721T045500Z.md`; S2 #595 is the
  composition consumer and #535 owns interaction rollouts.
- **Equation leg:** exact measured anchors live in the JSON receipt; promotion to callable equations
  waits for receiver-closed evaluator semantics.
- **Pointer delta:** none.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; operating manual; v7.5/v8 specs; M2, G1/G3, r1b3,
#425/flicker, and v10 blocker receipts; lane registry; subagent ledger; live delegation inbox through
`2026-07-21T04:31:32Z`; and the primary sources linked above.

**Tests:** 5 focused tests passed; Ruff passed; `py_compile` passed. **MAIN landing review is
required** for the Python measurement implementation, empirical interpretation, and lane/state diff.
