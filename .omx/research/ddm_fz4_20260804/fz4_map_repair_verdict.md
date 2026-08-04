---
title: "ddm_fz4 map-repair verdict: the tz1 adaptive map loses despite the byte win"
unit: ddm_fz4
date_utc: 2026-08-04
axis: "[macOS-CPU advisory]"
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
own_vehicle_frontier: "S = 0.7541459 @ 358,084 B [macOS-CPU advisory]"
---

# ddm_fz4 - stranded map row closeout

## Answer

The with-map repair row does **not** pay. It is smaller by 117,092 bytes, but
the map damages both scorer legs:

| row | S | bytes | d_seg | d_pose | axis |
|---|---:|---:|---:|---:|---|
| without-map `sub_final` | 0.7541459 | 358,084 | 0.00431179 | 0.00071459 | `[macOS-CPU advisory]` |
| with-map `sub_map_repair` | 1.9690434 | 240,992 | 0.00515854 | 0.16711321 | `[macOS-CPU advisory]` |
| delta, with-map minus without | +1.2148976 | -117,092 | +0.00084675 | +0.16639862 | same |

Breakdown of the +1.2148976 score regression: rate term -0.0779668,
seg term +0.0846750, pose term +1.2081893. The pre-registered map
break-even was `delta d_seg < 7.56e-4`; observed `delta d_seg` is
`8.4675e-4`, so the seg leg alone fails before pose is counted.

Verdict scope: **FORMULATION**. This closes tz1's `[16,12,8,4]`
margin-coupled map plus the current selective F0PR1 repair on the pu2 vehicle.
It does not kill gentler maps or pose-aware rung maps.

## Custody

`sub_map_repair` now has a clean canonical inflate receipt:
`/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_map_repair/fz4_byteclose_inflate_receipt.json`.
That wrapper ran `tac.submission_chain.run_inflate` in the foreground, rc=0,
one raw file, `3,662,409,600` bytes, `204.403s`.

The n600 evaluator report is:
`/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_map_repair/report.txt`.
It was the active fz4 scorer-slot report; this closeout did not run a second
n600 evaluator call. The clean inflate was rerun after the report to remove the
stranded custody ambiguity, and the raw SHA remained
`3e7fb620e961aa73d5c445872deb8d56213863ca7b422bb219fdd64d3cb868cb`.

The no-map row's committed fz3 receipt carries a historical symlink note, but
the SSD-side `fz2` receipt already supersedes that custody caveat for the same
archive: `/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/fz2_byteclose_receipt.json`
has clean `run_inflate`, byte ledger closes true, residual 0, and the same
recomputed S `0.7541458627114951`.

Machine-readable fz4 receipt:
`.omx/research/ddm_fz4_20260804/fz4_map_repair_eval_receipt.json`.

## Census And Ranking

Per-pair table: `.omx/research/ddm_fz4_20260804/fz4_per_pair_damage_table.csv`
has 600 data rows. Summary:
`.omx/research/ddm_fz4_20260804/fz4_per_pair_damage_summary.json`.

Key census facts:

| fact | value |
|---|---:|
| ep854 damage census n | 600 |
| ep854 mean d_pose | 37.8770625 |
| ep854 evaluator control | 37.87713242 |
| pu2 mean d_pose | 0.0015451774 |
| map-only mean d_pose | 0.1688459936 |
| map pairs worse than pu2 by >1e-3 | 523 |
| map pairs better than pu2 | 12 |
| no-map repaired pairs | 21 |
| with-map repaired pairs | 4 |

Measured k4-envelope rule: the threshold was `<=123.85x`. Pairs
`0, 1, 3, 511` exceeded it, and all four have k>=6/k8 escalation evidence.
No measured beyond-envelope pair is left without k>=5 treatment or exclusion.

Three-row disposition, ordered by current row authority rather than optimistic
estimate:

| order | row | evidence | S |
|---:|---|---|---:|
| 1 | standalone-live / pu2 + selective F0PR1 tail repair | measured n600 exact archive bytes | 0.7541459 |
| 2 | #827 cr2_ep854 + mixed-k F0PR1 stream | estimate from n600 census + n=29 k-ladder/stratified repairs, not byteclosed | 0.757-0.783 est. |
| 3 | ep854 standalone (#881) + F0PR1-compatible repair | stranded estimate, no separate pose-carrying byteclosed row here | 0.74279 floor estimate from hot-state arithmetic |

The estimated #881 row may be better by score if it composes, but it is not a
row until receiver custody and n600 eval exist. The measured own-vehicle
frontier remains `sub_final`.

## Follow-ons

FIRED: map-vs-no-map n600 verdict, clean inflate custody for `sub_map_repair`,
per-pair damage table, k4-envelope disposition, and the three-row ranking.

FOLDED: the current `[16,12,8,4]` margin-coupled map on this vehicle.

QUEUED-WITH-FIRE-ORDER: gentler or pose-aware maps must pass the same census
apparatus before candidacy; #827/#881 repair requires the all-600 mixed-k
solve plus receiver wiring before any score-rank claim.
