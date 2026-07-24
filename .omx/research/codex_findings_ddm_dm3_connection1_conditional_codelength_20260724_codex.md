# Codex findings — DDM DM3 CONNECTION-1 conditional codelength

`lane_id=lane_ddm_dm3_connection1_conditional_codelength_20260724`
`research_only=true` · `score_claim=false` · `promotion_eligible=false` ·
`main_review_required=true`
`evidence_axis=[macOS-CPU frozen-scorer advisory]`
`0.1910828242 [contest-CPU] UNMOVED`

## One-line verdict

**MEASURED:** one deterministic leave-one-pair-out fold in each of 36 eligible
same-bucket families gives
`delta_B_connection = 7,049 - (624 + 5,237) = +1,188 B`; 32 buckets are
positive and four nonpositive. Identity persistence wins 34/36 selections and
all 1,557 positive selected-family bytes; ξ-advection wins 0/36, while the two
affine selections are both negative. This is positive CONNECTION evidence for
this semantic-record formulation, concentrated outside static boundary rows,
not receiver/archive or score evidence.

Receipt:
`.omx/research/ddm_dm3_connection1_conditional_codelength_20260724T135912Z/ddm_dm3_connection1_conditional_codelength_receipt.json`
(`sha256=2c175366da196b8f79e2d3de1ad0a8c1844e78a1d2cab25921eaa15bec46346b`).

## Population and held-out protocol

The SHA-bound PF2 event index represents 37 bucket families across the full
n600 solved object. Thirty-six have at least one same-bucket consecutive
transition, totaling 8,602 eligible transitions. The one honest `NULL` is
`lane_mycar__cell__transient`: 16 represented pairs but no consecutive pair.

For each eligible bucket, the lower-median consecutive transition is the priced
row. Every other consecutive transition in that bucket is available to fit the
ξ-translation or affine state; the priced pair is excluded. Identity is a
fit-free generic program. This is a deterministic bucket-level
leave-one-pair-out probe, not an all-8,602-fold estimate. Each row's next
measurement is the full held-out-fold sweep.

The later exact DM1 semantic record is priced two ways:

1. static zlib-9, LZMA preset 9, or adaptive order-1 context arithmetic;
2. identity, ξ-advected, or affine-tracked history plus an exact XOR residual,
   with the shortest exact coder selected deterministically.

The history price charges a 15-byte fixed packet frame, one-byte program
selector, 0/4/24 bytes of program state, and the entire exact DM1 residual
container. Every static container and history candidate parses back exactly to
the later canonical semantic record.

## Aggregate byte accounting

| quantity | bytes / count |
|---|---:|
| `B_static` | 7,049 B |
| `B_history_program` | 624 B |
| `B_residual` | 5,237 B |
| `B_history_total` | 5,861 B |
| `delta_B_connection` | **+1,188 B** |
| positive / nonpositive buckets | 32 / 4 |
| identity selections | 34 / 36 |
| ξ-advected selections | 0 / 36 |
| affine-tracked selections | 2 / 36, both negative |

Identity accounts for 94.444% of selected bucket families and 100% of positive
selected-family savings. The positive result therefore supports persistence
as generic program structure. It does not support a positive motion-conditioned
ξ or affine claim at this formulation.

## Required type × stratum × exact-support-size decomposition

The machine-readable receipt gives one row for every exact later-support size.
The grouped headline is:

| bucket type | stratum | rows | `B_static` | program | residual | history total | `delta_B_connection` | positive / nonpositive |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| static in image | boundary | 9 | 2,740 | 192 | 2,790 | 2,982 | **-242** | 5 / 4 |
| static in image | cell | 9 | 1,531 | 144 | 730 | 874 | **+657** | 9 / 0 |
| static in ξ proxy | boundary | 1 | 162 | 16 | 107 | 123 | **+39** | 1 / 0 |
| transient | boundary | 9 | 1,441 | 144 | 1,055 | 1,199 | **+242** | 9 / 0 |
| transient | cell | 8 | 1,175 | 128 | 555 | 683 | **+492** | 8 / 0 |

Static boundary is the only aggregate-negative slice. The four nonpositive
bucket rows are `lane_undrivable__boundary__static_in_image` (-34 B),
`road_lane__boundary__static_in_image` (-116 B, affine),
`road_mycar__boundary__static_in_image` (-40 B), and
`road_undrivable__boundary__static_in_image` (-179 B, affine). This is
instance/formulation-scoped surprisal, not a CONNECTION family kill.

### Exact per-bucket ledger

`n` is the exact later PF2 occupied-support count.

| row | bucket | type | stratum | n | winner | static | program | residual | history | delta |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|
| 0 | `lane_movable__boundary__static_in_image` | static | boundary | 31 | identity | 184 | 16 | 139 | 155 | +29 |
| 1 | `lane_movable__boundary__transient` | transient | boundary | 6 | identity | 160 | 16 | 119 | 135 | +25 |
| 2 | `lane_movable__cell__static_in_image` | static | cell | 79 | identity | 158 | 16 | 69 | 85 | +73 |
| 3 | `lane_movable__cell__transient` | transient | cell | 2 | identity | 142 | 16 | 69 | 85 | +57 |
| 4 | `lane_mycar__boundary__static_in_image` | static | boundary | 11 | identity | 163 | 16 | 113 | 129 | +34 |
| 5 | `lane_mycar__boundary__transient` | transient | boundary | 3 | identity | 151 | 16 | 115 | 131 | +20 |
| 6 | `lane_mycar__cell__static_in_image` | static | cell | 5 | identity | 151 | 16 | 69 | 85 | +66 |
| 7 | `lane_undrivable__boundary__static_in_image` | static | boundary | 291 | identity | 273 | 16 | 291 | 307 | -34 |
| 8 | `lane_undrivable__boundary__transient` | transient | boundary | 2 | identity | 153 | 16 | 108 | 124 | +29 |
| 9 | `lane_undrivable__cell__static_in_image` | static | cell | 19 | identity | 165 | 16 | 73 | 89 | +76 |
| 10 | `lane_undrivable__cell__transient` | transient | cell | 3 | identity | 147 | 16 | 69 | 85 | +62 |
| 11 | `movable_mycar__boundary__static_in_image` | static | boundary | 19 | identity | 177 | 16 | 125 | 141 | +36 |
| 12 | `movable_mycar__boundary__transient` | transient | boundary | 9 | identity | 162 | 16 | 115 | 131 | +31 |
| 13 | `movable_mycar__cell__static_in_image` | static | cell | 9 | identity | 157 | 16 | 69 | 85 | +72 |
| 14 | `movable_mycar__cell__transient` | transient | cell | 7 | identity | 150 | 16 | 69 | 85 | +65 |
| 15 | `road_lane__boundary__static_in_image` | static | boundary | 591 | affine | 754 | 40 | 830 | 870 | -116 |
| 16 | `road_lane__boundary__transient` | transient | boundary | 10 | identity | 175 | 16 | 135 | 151 | +24 |
| 17 | `road_lane__cell__static_in_image` | static | cell | 1,206 | identity | 197 | 16 | 128 | 144 | +53 |
| 18 | `road_lane__cell__transient` | transient | cell | 6 | identity | 144 | 16 | 69 | 85 | +59 |
| 19 | `road_movable__boundary__static_in_image` | static | boundary | 119 | identity | 261 | 16 | 244 | 260 | +1 |
| 20 | `road_movable__boundary__transient` | transient | boundary | 3 | identity | 152 | 16 | 111 | 127 | +25 |
| 21 | `road_movable__cell__static_in_image` | static | cell | 913 | identity | 168 | 16 | 80 | 96 | +72 |
| 22 | `road_movable__cell__transient` | transient | cell | 1 | identity | 141 | 16 | 69 | 85 | +56 |
| 23 | `road_mycar__boundary__static_in_image` | static | boundary | 250 | identity | 269 | 16 | 293 | 309 | -40 |
| 24 | `road_mycar__boundary__transient` | transient | boundary | 6 | identity | 158 | 16 | 115 | 131 | +27 |
| 25 | `road_mycar__cell__static_in_image` | static | cell | 61 | identity | 156 | 16 | 69 | 85 | +71 |
| 26 | `road_mycar__cell__transient` | transient | cell | 38 | identity | 150 | 16 | 69 | 85 | +65 |
| 27 | `road_undrivable__boundary__static_in_image` | static | boundary | 571 | affine | 425 | 40 | 564 | 604 | -179 |
| 28 | `road_undrivable__boundary__transient` | transient | boundary | 10 | identity | 174 | 16 | 124 | 140 | +34 |
| 29 | `road_undrivable__cell__static_in_image` | static | cell | 1,290 | identity | 217 | 16 | 101 | 117 | +100 |
| 30 | `road_undrivable__cell__transient` | transient | cell | 1 | identity | 144 | 16 | 69 | 85 | +59 |
| 31 | `undrivable_movable__boundary__static_in_image` | static | boundary | 79 | identity | 234 | 16 | 191 | 207 | +27 |
| 32 | `undrivable_movable__boundary__static_in_xi_proxy` | ξ proxy | boundary | 1 | identity | 162 | 16 | 107 | 123 | +39 |
| 33 | `undrivable_movable__boundary__transient` | transient | boundary | 2 | identity | 156 | 16 | 113 | 129 | +27 |
| 34 | `undrivable_movable__cell__static_in_image` | static | cell | 324 | identity | 162 | 16 | 72 | 88 | +74 |
| 35 | `undrivable_movable__cell__transient` | transient | cell | 317 | identity | 157 | 16 | 72 | 88 | +69 |

## DM1 CONNECTION-NULL crosswalk

All 25 DM1 demand rows now point to a measured DM3 bucket-family cell. The
scope is deliberately narrower than an original-row fill: the measured pair is
the deterministic consecutive transition for that bucket family, not the
original nonconsecutive DM1 pair. The receipt preserves that distinction in
every crosswalk row. No value is imputed for the one ineligible bucket.

## Typed adjudication

No new typology was introduced. Static records retain DM1's
`SKELETON/L4_scorer_feature` boundary or `FIBER/L4_scorer_feature` cell home.
The generic history program is `CONNECTION/L4_scorer_feature`; the exact XOR
correction is `RESIDUAL/L4_scorer_feature`. A positive conditional byte delta
means the generic CONNECTION program absorbed part of this solved semantic
record. It does not make the free program's video-derived selector or state
free: both are charged in `B_history_program`.

## Directive disposition

| Binding directive | Disposition |
|---|---|
| Full n600 solved object, every represented eligible bucket | PASS — 37 represented, 36 eligible, 8,602 transitions |
| Same-bucket consecutive supports only | PASS — every priced pair has gap one |
| No RG4 or reachability vocabulary | PASS — only sealed PF2 support and solved DM1 semantic records |
| Reuse DM1 real coder stack | PASS — exact zlib-9, LZMA-9, and order-1 context arithmetic |
| Identity / ξ-advected / affine-tracked | PASS — all three candidates priced where fit is available |
| Charge selector, state, residual, and framing | PASS — 16/20/40 B program packets plus full residual container |
| Leave one pair out | PASS — priced transition excluded from same-bucket fit |
| Exact parseback | PASS — every static and history candidate exact |
| Decompose headline | PASS — exact type × stratum × support-size rows plus aggregate |
| Fill DM1 CONNECTION NULL honestly | PASS — 25/25 bucket-family crosswalk; original-pair distinction retained |
| $0, no training/dispatch/eval/archive/pointer mutation | PASS |
| Per-arm inbox | EMPTY through the final measurement milestone |
| Fleet broadcast | CONSULTED; prior Fisher/waterfill guidance not applicable to this byte-only probe |
| Triality and MAIN review | PASS / REQUIRED |

## STORES CONSULTED

- delegated authority prompt, 6,365 bytes, SHA
  `3285c2f23b3eb515cf5bbc86288d1f689c35981af78418ea8eb7de543f3fc834`
- `CLAUDE.md`, `AGENTS.md`, and `docs/operating_manual_craft_handoff.md`
- top ten Claude memory entries and current Codex/Claude sister memos
- `reports/latest.md`, lane registry, subagent progress, canonical-equation
  registry, probe ledgers, and both live directive inboxes
- OP-GC1-4 council memo and the DM1 findings/DAG/equation/receipt
- SHA-bound solved-plane receipt/chunks and production input loader
- PF2 event-index receipt and NPZ on the SSD tier
- frozen `segnet.safetensors` and `upstream/modules.py`

No web source, RG4 vocabulary, old vehicle lineage, training surface, paid
provider, archive evaluator, or live frontier artifact was used.

## Triality, first-rung use, and MAIN review

- **DSL/code:** `ddm_dm3_connection_conditional_codelength.py`, its CLI,
  config, sealed DM1 coder/type reuse, and tests.
- **DAG:** `ddm_dm3_connection1_conditional_codelength_DAG_FEED_20260724.md`.
- **Equations:** callable
  `ddm_dm3_heldout_connection_conditional_codelength_v1`, registry appended.
- **Evidence:** SHA-bound receipt and manifest above.

This is the first program-structure rung. The next rung is the complete
held-out-fold distribution per bucket; only after a MAIN-reviewed positive
sign survives should a receiver-closed RGB/Pose acquisition test be considered.

MAIN must review: the one deterministic fold per bucket versus the owed
all-fold estimate; external PF2 support as context for cell records; fixed
packet overhead and the Wyner/two-part charge boundary; ordinal
correspondence-derived ξ/affine fits; the conclusion that the measured positive
signal is identity persistence rather than motion; the registry append; and
the firewall against treating +1,188 semantic bytes as archive or score bytes.

## Verdict scope

`MEASURED` means exact held-out conditional bytes for 36 deterministic
bucket-level semantic-record rows on the local frozen-SegNet axis. It excludes
an all-fold estimate, legal RGB receiver, Pose survival, full-video replay,
`upstream/evaluate.py`, contest CPU/CUDA, archive bytes, submission score,
promotion, and frontier movement.
