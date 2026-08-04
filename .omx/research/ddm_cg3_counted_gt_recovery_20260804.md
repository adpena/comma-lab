# ddm_cg3 counted-GT recovery - surgical islands and interfaces - 2026-08-04

- arm: `ddm_cg3`
- axis: `[macOS-CPU advisory]` / bounded realization sample; NON-PROMOTABLE
- score_claim: false
- promotion_eligible: false
- pointer_moved: false
- receipt: `.omx/research/ddm_cg3_20260804/ddm_cg3_counted_gt_recovery_receipt.json`
- receipt sha256: `5f633f04c3b320f784fcde51ffb2e54265e2b7725ea6518eb84754650b800f50`
- command: `.venv/bin/python experiments/ddm_cg3_counted_gt_recovery.py --realization-pairs 32 --realization-batch 4`
- wall_seconds: 427.5

## Source Reads

- `PROGRAM.md`: exact-score evidence axes and proxy/advisory boundary.
- `CLAUDE.md`: SegNet class order is the measured comma10k canonical order
  `[Road,Lane,Undrivable,Movable,MyCar]`; luma-sort derivation is forbidden.
- `.omx/state/canonical_task_status.jsonl`: no literal `task_id == 939`, `#939`,
  or `task #939` row exists in the current snapshot. This memo resolves the
  operator-specified #939 recovery request, but does not invent a missing ledger row.
- `.omx/research/ddm_rl1_roadlane_interface_price_20260803.md`: survivor row was
  n32 independent-pair per-class Lane crop, Brotli-q11 projected to 272,869 B,
  0.911x W, +26,500 B under the Road<->Lane budget; description only.
- `.omx/research/ddm_pc2_perclass_road_edges_20260802.md`: SEG is one graph;
  decompose per edge, never per class; Road hub and Road<->Lane dominance.
- `.omx/research/ddm_cg1_force_class_edge_ledger_20260803.jsonl`: cg1r edge and
  verb ledger; Road<->Lane 235,148 flips and Road hub 87.48%.
- `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_directed_flip_receipt.json`:
  live cx1 n600 directed/undirected flip receipt, total flips 508,640.

## Method

All description prices are MEASURED at n600 from cached GT and cx1 argmax arrays:

- GT argmax: `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy`
- cx1 argmax: `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy`
- positive control: total flips `508,640`, d_seg `0.004311794704861111`, matching
  the cx1 receipt.

Class order was self-detected by spatial/static signature and then checked
against the canonical source order:

`Road=0, Lane=1, Undrivable=2, Movable=3, MyCar=4`.

S arithmetic:

- `gross described correction S if perfect = 100 * fixable_flips / (600 * 196608)`
- `rate S = 25 * bytes / 37,545,489`
- `net S if perfect and pose unchanged = rate S - gross described correction S`

The realization half is a bounded sample, not authority: 32 evenly strided
pairs, CPU frozen SegNet, no MPS, no full n600 scorer job. The script reuses
the actual inflated cx1 RGB and paints described score-grid pixels to target
class prototype colors, then runs SegNet. This is a real R+uint8+SegNet
survival measurement of a simple realization mechanism, not a receiver-closed
submission and not a kill of the family.

GT video was not decoded in this run; the run read cached GT argmax only. The
GT decode safety boundary therefore was not engaged. Any future GT-video decode
must use `frame_utils.yuv420_to_rgb`.

SSD cold-store note: the sandbox refused creating
`/Volumes/VertigoDataTier/pact/ddm_cg3_20260804` with `PermissionError`. The
evidence is small (`912K` total, `880K` artifacts), so the small deterministic
coder artifacts were written durably under `.omx/research/ddm_cg3_20260804/`
instead of `/tmp`.

## n600 Description Prices

Verdict scope for every price row below: INSTANCE. The byte counts are MEASURED
with real coders; the perfect-realization S arithmetic is DERIVED from those
bytes and the measured flip counts.

| surface | fixable flips | best coder bytes | rate S | gross S if perfect | net S if perfect | coder race bytes |
|---|---:|---:|---:|---:|---:|---|
| `rl1_lane_crop` | 185,801 | Brotli-q11 206,688 | 0.137625 | 0.157505 | -0.019880 | Brotli-q11 206,688; LZMA1-raw 221,137; SMEVR-r7-nibble 252,434 |
| `island_lane_components` | 185,801 | LZMA1-raw 174,338 | 0.116085 | 0.157505 | -0.041421 | LZMA1-raw 174,338; Brotli-q11 174,639; SMEVR-r7-nibble 298,685 |
| `island_movable_components` | 78,833 | LZMA1-raw 60,585 | 0.040341 | 0.066828 | -0.026486 | LZMA1-raw 60,585; Brotli-q11 61,166; SMEVR-r7-nibble 91,364 |
| `interface_Road<->Lane` | 235,148 | Brotli-q11 232,155 | 0.154582 | 0.199337 | -0.044755 | Brotli-q11 232,155; SMEVR-r7-nibble 236,985; LZMA1-raw 248,469 |
| `interface_Road<->Undrivable` | 89,545 | Brotli-q11 68,919 | 0.045890 | 0.075908 | -0.030018 | Brotli-q11 68,919; SMEVR-r7-nibble 74,872; LZMA1-raw 77,141 |
| `interface_Road<->MyCar` | 63,027 | Brotli-q11 34,697 | 0.023103 | 0.053429 | -0.030325 | Brotli-q11 34,697; LZMA1-raw 36,154; SMEVR-r7-nibble 39,100 |
| `interface_Undrivable<->Movable` | 61,892 | LZMA1-raw 46,862 | 0.031203 | 0.052466 | -0.021263 | LZMA1-raw 46,862; Brotli-q11 46,971; SMEVR-r7-nibble 50,456 |
| `interface_Road<->Movable` | 57,225 | Brotli-q11 45,888 | 0.030555 | 0.048510 | -0.017955 | Brotli-q11 45,888; LZMA1-raw 45,903; SMEVR-r7-nibble 50,427 |
| `interface_Lane<->MyCar` | 903 | LZMA1-raw 1,712 | 0.001140 | 0.000765 | +0.000374 | LZMA1-raw 1,712; Brotli-q11 1,746; SMEVR-r7-nibble 3,102 |
| `interface_Lane<->Movable` | 681 | LZMA1-raw 1,557 | 0.001037 | 0.000577 | +0.000459 | LZMA1-raw 1,557; Brotli-q11 1,623; SMEVR-r7-nibble 2,465 |
| `interface_Movable<->MyCar` | 135 | LZMA1-raw 196 | 0.000131 | 0.000114 | +0.000016 | LZMA1-raw 196; Brotli-q11 200; SMEVR-r7-nibble 251 |
| `interface_Lane<->Undrivable` | 84 | LZMA1-raw 306 | 0.000204 | 0.000071 | +0.000133 | LZMA1-raw 306; Brotli-q11 333; SMEVR-r7-nibble 470 |

## rl1 Lane-Crop n600 Re-Price

MEASURED, INSTANCE, n600.

The rl1-compatible apples-to-apples independent-pair Brotli-q11 row is
`277,034 B`. Against Road<->Lane's `235,148` flips and W
`1.2731082153320312 B/flip`, that is:

- `1.178126 B/Road-Lane flip`
- `0.925394x W`
- `+22,334.85 B` under the `299,368.85 B` Road<->Lane budget

So the n32 projection is directionally confirmed but not exact: `272,869 B`
projected became `277,034 B` measured, reducing the projected margin from
`+26,499.85 B` to `+22,334.85 B`.

The all-pair framed cg3 Lane-crop object is a different serialization and races
all 600 pairs jointly: Brotli-q11 `206,688 B`, `0.878970 B/Road-Lane flip`,
`0.690413x W`, `+92,680.85 B` under the Road<->Lane budget. That is not the
same object as rl1's independent-pair row, but it is a measured stronger price
for the same counted GT Lane-mask description.

## Description to Realization

MEASURED, INSTANCE sample, denominator `n=32` evenly strided pairs. Base sample
d_seg was `0.004253705342610677` and the live raw recomputed argmax matched the
cache exactly: `base_argmax_match_rate_vs_cache = 1.0`.

`survival = fixed described flips / described fixable flips`. `local net ratio`
subtracts collateral new flips inside the local accounting. `sample seg+rate`
adds the n600 measured rate S to the sample whole-frame seg S delta; it is
included only as a realization-screen arithmetic, not as n600 authority.

| surface | described flips in sample | fixed | collateral | survival | local net ratio | sample seg delta S | sample seg+rate delta S |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rl1_lane_crop` | 9,591 | 4,768 | 15,994 | 0.497 | -1.170 | +0.116491 | +0.254116 |
| `island_lane_components` | 9,591 | 4,767 | 15,996 | 0.497 | -1.171 | +0.116587 | +0.232671 |
| `island_movable_components` | 3,791 | 2,917 | 17,209 | 0.769 | -3.770 | +0.179736 | +0.220077 |
| `interface_Road<->Lane` | 12,065 | 6,697 | 7,605 | 0.555 | -0.075 | -0.024509 | +0.130073 |
| `interface_Road<->Undrivable` | 4,730 | 1,550 | 2,592 | 0.328 | -0.220 | -0.001065 | +0.044825 |
| `interface_Road<->MyCar` | 3,548 | 831 | 1,452 | 0.234 | -0.175 | +0.003322 | +0.026425 |
| `interface_Undrivable<->Movable` | 3,011 | 1,279 | 2,233 | 0.425 | -0.317 | -0.004530 | +0.026674 |
| `interface_Road<->Movable` | 3,254 | 1,441 | 3,323 | 0.443 | -0.578 | +0.003799 | +0.034354 |
| `interface_Lane<->MyCar` | 73 | 47 | 419 | 0.644 | -5.096 | +0.001812 | +0.002952 |
| `interface_Lane<->Movable` | 81 | 3 | 83 | 0.037 | -0.988 | -0.000795 | +0.000242 |
| `interface_Movable<->MyCar` | 0 | 0 | 0 | n/a | n/a | +0.000000 | +0.000131 |
| `interface_Lane<->Undrivable` | 0 | 0 | 0 | n/a | n/a | +0.000000 | +0.000204 |

## Verdicts

- MEASURED INSTANCE: counted GT descriptions are not intrinsically too large.
  Four high-mass interfaces have negative perfect-realization S even after rate:
  Road<->Lane `-0.044755`, Road<->Undrivable `-0.030018`,
  Road<->MyCar `-0.030325`, Undrivable<->Movable `-0.021263`, plus
  Road<->Movable `-0.017955`. Lane and Movable island component descriptions
  are also negative under perfect realization.
- MEASURED INSTANCE sample: the simple camera-paint realization is not
  shippable. No surface beats rate on the bounded sample. Class/island crops
  are actively bad due to collateral. Road<->Lane is the only material edge
  with a meaningful whole-frame sample seg improvement (`-0.024509 S` before
  rate), but its measured rate is `+0.154582 S`.
- DERIVED: counted GT description price alone is not the blocker; receiver-local
  realization/collateral control is the blocker.
- FORMULATION-scope fold: the prototype-color camera-paint realization tested
  here folds. This is not a FAMILY kill for counted interface descriptions.

## Boundaries

- No byte-closed archive was built.
- No pose was measured, changed, or assumed improved.
- No full n600 scorer job was run; the only scorer forwards were the bounded
  n32 realization sample.
- MPS was not used.
- The receiver, `tac.submission_chain`,
  `src/tac/optimization/direct_description_carrier_compose.py`,
  `.omx/research/ddm_cr1_composition_row_827_20260801.md`, and
  `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md` were not edited.
- The staged git index was not touched.
- The realization method is a screen for survival, not a production receiver.
  It paints target-class prototype colors into camera rectangles and therefore
  measures one concrete mechanism's survival/collateral, not the best possible
  realization for the counted descriptions.

## Fire / Fold / Queue

1. QUEUE behind `ddm_bz1`: Road<->Lane receiver-local realization, bounded n<=120
   first, then n600 only when the scorer slot is free. It has the largest
   measured description headroom and the only material sample seg improvement,
   but must solve collateral and rate.
2. QUEUE after Road<->Lane: Road<->Undrivable and Undrivable<->Movable
   receiver-local edge realizations. Their descriptions are priced favorably,
   but the tested realization survival is weak.
3. FOLD as tested: class/island crop camera-paint realization for Lane and
   Movable. The descriptions are cheap enough under perfect realization, but
   this mechanism pays unacceptable collateral on n32.
4. FOLD standalone minor interfaces: Lane<->MyCar, Lane<->Movable,
   Movable<->MyCar, Lane<->Undrivable. Their perfect-realization S is already
   rate-negative or negligible as standalone counted surfaces; they can only
   ride as piggyback context inside a larger edge receiver.

own-vehicle frontier S = 0.7910689 @ 353,805 B [macOS-CPU advisory] — UNMOVED (this arm byte-closes nothing).
