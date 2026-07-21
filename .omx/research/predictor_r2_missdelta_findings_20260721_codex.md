# Task #578 Round 2 findings — predictor miss delta

`lane_id=predictor_r2_missdelta` · `research_only=true` ·
`[macOS-CPU advisory]` · `score_claim=false` · `promotion_eligible=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `receiver_closed=false` ·
`MAIN_REVIEW_REQUIRED=true`

## Verdict

`BOUNDARY_DELTA_BAR_MISSED`.

The strict PBD1 formulation costs 653,565 bytes for 1,772,327 exactly decoded
class-(a) misses, or 2.950087653 bits/miss against the required 0.365
bits/miss. This is an 8.0824x miss. The negative is scoped to this
predictor-known compatible-contour grammar and context model; it is not a
negative on sparse contour/run grammars or the boundary-residual family.

Both PBD1 and PBS1 sidecars were decoded against the predictor, reproduced the
selected target sites exactly, and re-encoded byte-identically. No camera-RGB
realization, R operator, scorer, archive, or receiver was executed.

## D1 — real round-1 miss structure

The n600 total re-derived exactly to 3,122,086 misses. The partition is mutually
exclusive and exhaustive.

| target class | misses | (a) 1-2 px compatible boundary | (b) coherent blob | (c) scattered |
|---|---:|---:|---:|---:|
| Road | 2,074,384 | 1,129,125 | 940,627 | 4,632 |
| Lane | 201,896 | 120,656 | 78,767 | 2,473 |
| Undrivable | 805,738 | 489,169 | 313,861 | 2,708 |
| Movable | 5,006 | 4,939 | 35 | 32 |
| MyCar | 35,062 | 28,438 | 6,617 | 7 |
| **all** | **3,122,086** | **1,772,327 (56.77%)** | **1,339,907 (42.92%)** | **9,852 (0.32%)** |

The receipt carries per-class/per-stratum horizontal-run, compatible-contour
adjacency, component-size, and Chebyshev distance-to-predicted-boundary
histograms. The independent n64 prefix contains 177,391 misses: 122,131 class
(a), 54,742 class (b), and 518 class (c).

## D2 — strict signed boundary delta coder

PBD1 traverses #307 straightest-first predictor contours and uses #557 adaptive
arithmetic contexts over arc phase, ordered class pair, local curvature, and
prior activity/offset. The n600 aggregate has 3,269,587 anchor symbols,
338,386 activity bytes, 315,146 offset bytes, and a 33-byte header. The payload
alone is 2.949938696 bits/miss; container accounting is 2.950087653.

No nonempty n600 class/stratum row meets 0.365 bits/miss. The densest large rows
are Road interior 2.840794196, Undrivable interior 2.599749208, and MyCar
boundary 2.495626822 bits/miss. Lane is especially expensive: 9.205316544
interior and 6.482329244 boundary bits/miss. Small critical-event rows are
header/context dominated.

DERIVED: the 0.365 bar allows about 80.9 KB for these 1.77M events. Either the
activity stream (338.4 KB) or offset stream (315.1 KB) alone exceeds that
budget, so tuning probabilities inside the current dense-anchor syntax cannot
close the gap. A future formulation must remove dense per-anchor activity and
encode sparse contour runs/gaps or a stronger shared shape primitive.

## D3 — bounded n64-fit refinements

The Lane arc-phase table is rejected. Its 151 counted bytes save only 94 net
n64 misses (385.22 measured coder bits, less than its 1,208 policy bits). On
n600 it adds 669 correct Lane cells but spills 521 other cells, for only 148 net
misses saved.

The Road contour-context table is admitted by the stipulated global d_seg byte
gate: 506 bytes save 7,965 net n64 misses (32,641.30 measured coder bits versus
4,048 policy bits) and 87,134 net n600 misses. Its full-facet result must remain
visible: Road gains 165,300 correct n600 cells while Lane loses 56,905,
Undrivable 9,028, Movable 10,807, and MyCar 1,426. Thus the aggregate
satisfaction rises from 0.973533749051 to 0.974272393121, but Lane satisfaction
falls from 0.707667826462 to 0.625273116635. This policy is description-space
KKT-admitted, not a promotion recommendation.

Road residuals are entirely in the lower two image thirds under the measured
top-third horizon diagnostic. Before refinement, Road misses are predicted as
Undrivable 917,339, Lane 533,107, Movable 367,161, and MyCar 256,777; 1,129,125
are class-(a) boundary events. Lane has 201,896 false negatives, of which
120,656 are class-(a), while only 5,521 lie in the rendered Lane chart mask;
196,375 are chart-not-visible. The chart visibility failure, not only dash
phase, dominates remaining Lane headroom.

## D4 — composed description-space curve

`realization_breakeven_bytes_v1` resolves to 6.658589531e-7 S/byte. One
description flip is 8.477105035e-7 S, hence 1.273108215 bytes/flip. With the
delegated 0.365-bit bar, the derived exception budget is 142,445 bytes and the
implied base is 73,777 bytes.

The indivisible-chunk reverse-waterfill admits the 506-byte Road policy and
five refined PBD1 rows: MyCar boundary, Undrivable boundary, Movable track,
MyCar interior, and Lane interior. At the measured knee:

- variable bytes: 96,078;
- corrected misses from round 1: 237,414;
- remaining misses: 2,884,672;
- description d_seg: 0.024453667535;
- implied-base total: 169,855 bytes.

This box projection is not an archive claim. The round-1 declared raw PXCH plus
Lane base is already 262,498 bytes; the same knee is therefore 358,576 bytes
under declared accounting and cannot fit the 216,222-byte box.

The best coherent-shape bundle corrects 1,236,868 pixels for 233,098 bytes
(1.507666137 bits/miss), but it is too large as one indivisible chunk. The full
PBD1 + >=4-pixel PBS1 composition uses 922,738 variable bytes and leaves 11,454
description misses; its implied-base total is 996,515 bytes. This proves that
most misses are structured, while also proving that the current containers do
not make them affordable. Surgical per-class/per-stratum/component shape
partitioning remains an open formulation.

## Custody and pointer delta

Fresh durable root:
`/Volumes/VertigoDataTier/pact/evidence/predictor_r2_20260721/canonical_r2_20260721/`.
The n64/n600 config hashes are respectively
`8dc985e9fe8408d49f763837ea5a54709fe49e0039e687fb16b2f55341a5d8e0`
and `9946ce10d0c19910c75d8b12cc74cb7643c026046d55d8b78cd28780b75f374f`.
The machine receipt SHA-256 is
`681c4ce7b29844553d264017152785802bf17d59cc31454954ccf9f13e051e27`.
The contest-CPU pointer remains exactly 0.1910828242. MAIN review is required.

## STORES CONSULTED

Task #578 delegated authority; CLAUDE.md and AGENTS.md; v7.5/v8 operating
specifications; round-1 build spec, findings, DAG, reuse manifest, measurement
receipt, static chart and predecessor seeds; Task #595 Lane packet; frozen n600
cache; #557 in-tree range coder; #307 contour prior; #595 stratum and s2 seed
surfaces; canonical breakeven equation; lane registry; latest arm and broadcast
inboxes.
