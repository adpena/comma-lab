# ddm_bs3 route-2 adjudication: born-small resolved carrier (B+C) — REFUSED at measured scope

Date: 2026-08-27 · Author: MAIN · Status: ADJUDICATED (measurement pre-existing, adjudication new)
verdict_scope: FORMULATION — born-small resolved-carrier body (bs3 101,150 B object) as a sub-0.12
route, measured at n=32 seeded-uniform-random advisory scope. Not a family-wide kill of every
possible born-small object; reactivation criterion below.

## What this closes

fb2's route table (`.omx/research/ddm_fb2_route_table_gb1_20260826.md`) names exactly 3 sub-0.12
routes. Route 2 = "born-small resolved carrier B+C". This memo adjudicates it DEAD on distortion
using the measurement that already existed on disk at harvest time:

`/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/checkpoints/stage_40_three_way_measurement.json`
(n=32 seeded-uniform-random, [macOS-CPU advisory, NON-PROMOTABLE], payloads retained per P0).

## The numbers (MEASURED, n=32 advisory)

- born_small_fresh_solve: d_seg 0.01188074757810682 (seg_s 1.188) · d_pose 0.5488710997248754
  (pose_s 2.343) · distortion_s 3.5308745779508395
- base (gb1-lineage) distortion_s on the same 32 pairs: 0.07624409308700053
- Δdistortion = **+3.455 S** — vs the route's entire rate credit of 79,065 B ≈ 0.0526 S,
  the distortion penalty is ~66× the credit.
- perfect_pose_floor_distortion_s 1.2316650424494517 (**+1.155 S vs base even granting pose = 0**)
- pose_gap_recovered_fraction 0.13324699682520516 — the fresh QS5-style solve recovered only
  13.3% of the pose gap.
- Composed with the bs3 body's rate 0.0673516331 (101,150 B): best-case S ≈ 1.26 ≈ 10.5× the
  0.12 bar. Even the seg-only perfect-pose floor composes to ≈ 1.30.

Concordant with bo2's independent REFUSAL at 209× (#1262) — two instruments, same direction,
different magnitudes because different objects; both far past any admission bar.

## Consequences (routed)

1. **fb2 route table now has 2 live routes**: route 1 (aligned W96 R+M — the r8 burn LIVE on
   Metal, past stage-04) and route 3 (rb1 four sealed configs, queued behind Metal). Route 2
   removed from the fire queue.
2. **bs4x storage unlock DE-PRIORITIZED**: its +12.3 GB reclaim only enables stages 1–4 of a
   family whose stage-40 adjudication is measured dead. Do not spend the storage boundary on it.
3. Task rows: #1258 (bo2 fired — distortion now MEASURED, this memo is the durable record),
   #1261 (address-free class no longer rests on n≤1 — bo2 + bs3 stage-40 concordant), #1262
   (verdict stands, reinforced). #1247's "byte-feasible by 36,858 B" clause survives as a RATE
   fact; the route dies on DISTORTION, which is exactly the two-way demand law (m124): byte
   feasibility alone is half the ledger.

## Reactivation criterion

A born-small object re-enters ONLY with a measured distortion_s within +0.028 of base at
n≥32 random scope (i.e., inside the campaign gap, not 123× it) — which per the sy2 law requires
another leg to CHANGE ITS OBJECT first (e.g., a trained renderer that carries pose, the r8/rb1
line). No re-fire on rate arithmetic alone.
