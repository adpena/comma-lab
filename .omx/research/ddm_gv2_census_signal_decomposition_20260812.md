# gv2 design supplement — the vd1 census joint distribution (ALL 200 per-event exact rows mined)

**Date:** 2026-08-12 · **Producer:** MAIN · **Consumer:** ddm_gv2 (gen-2 alphabet design) + any future prescreen calibration.
**Source:** `/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/main_harvest/results/EVENT_RESULTS.jsonl`
(sha `a97400d32878318d8eb657a36e62f523e4db48e402b292c09e611d2104b500b3`, 200 rows, [contest-CUDA T4 exact-upstream affected-pair n600 delta]).
Class indices (canonical, CLAUDE.md): 0=Road · 1=Lane · 2=Undrivable · 3=Movable · 4=MyCar.

## 1. The rescuable mass — pose-null construction is the measured single unlock

- Eligible (flip+ ∩ pose-pass): 5 events, **+6 flips**. Rescuable (flip+ but pose-FAIL): **21 events, +32 flips = 6.3× the eligible mass** (optimistic 2.713e-05 S).
- The failures are NEAR-misses: rescuable pose costs are **2–42× over** the 2.955e-09 per-event budget; median pose cost across all 26 flip-positive events is 3.06e-08 ≈ **10× budget**. A projection suppressing ≥~97% of the pose leak rescues essentially the whole set.
- Consequence for gv2: pose-null-BY-CONSTRUCTION (js4 projector / Q3 placement) is not a refinement — it is the difference between 5 and ~26 live events per 200, BEFORE any reach scaling.

## 2. The site-count law — bigger events are pose-fair

| site_count | n | med pose (global) | med net flip |
|---|---|---|---|
| 1 | 179 | 3.50e-08 | −1.0 |
| 2 | 14 | 2.93e-08 | −2.0 |
| 3 | 3 | 4.68e-08 | −3.0 |
| 4 | 3 | 3.03e-07 | −3.0 |
| 6 | 1 | 2.61e-07 | −5.0 |

Pose cost grows roughly with sites while flip magnitude does too → **per-flip pose cost ≈ constant with event size**. gv2's 10–100×-reach events (boundary-segment scale) do not pay a superlinear pose penalty. (Caveat: n is tiny above 2 sites; treat as a prior, not a law.)

## 3. Advisory→exact precision = 13% — calibrate every prescreen

All 200 events had passed js5's LOCAL realized-acceptance. At n600 exact only **26/200 gain net flips** (and median net flip across the store is −1: most "accepted" events HURT). Binding rule for gv2: divide any local-advisory optimistic projection by ~7.7× before comparing to the 0.000216 fire bar, or (better) prescreen with the affected-pair exact-affinity trick vd1 itself uses.

## 4. Edge structure of what actually works

Flip-positive edges: **Undrivable→Road 7 · MyCar→Road 6 · Road→Lane 5 · Lane→Road 3 · Road→Undrivable 2** — Road↔Lane = 8/26 (the m91 hub confirmed at event granularity), but the horizon edge (Undrivable↔Road) and hood edge (MyCar→Road) are co-equal veins gen-1 never designed for. Top single rescuable event is Undrivable→Movable (+4 flips at only 10× pose budget). gv2 should carry ALL THREE edge families, not Road↔Lane alone — the charter's hub-edge focus is hereby WIDENED (this supplement supersedes the charter's target-1 narrowing; the pose-null and store-schema contracts are unchanged).

## 5. Pair concentration

Flip-positive events concentrate on pairs {96: 9, 18: 7, 73: 4, 76: 3, 7: 2, 53: 1} — the same 6 affected pairs. Gen-1's generator only ever touched 6 pairs of 600. gv2's biggest structural headroom is COVERAGE: propose events across the full n600 flip-mass distribution (g3 score-atlas heavy-tail), not the 6 pairs js5 happened to reach.

verdict_scope: instance — distributional facts of the gen-1 ec1 store on the cp135 base; §1–§5 are design priors for gv2, not family verdicts.
