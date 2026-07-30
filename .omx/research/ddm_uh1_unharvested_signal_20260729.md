---
schema: ddm_uh1_unharvested_signal.v1
date_utc: 2026-07-29
arm: MAIN (inline $0 cached-data harvest — no convocation arm; sb1 live on the scorer slot)
lane_id: "lane_ddm_uh1_unharvested_signal_20260729"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[$0 cached-data analysis; per-pair numbers are realized through the real receiver+PoseNet; composed byte-close gates owed]"
operator_verbatim: "What signal would they find and what would they interpret and be curious about and be confused about and be excited about that we have or haven't harvested and interpreted and acted upon ourselves yet?"
operating_row: "pfs1 D1 MEASURED S=2.256641 (seg 0.389011 + pose 1.488093 + rate 0.379537; archive 624ffe57…, 569,996 B) [macOS-CPU advisory]"
inputs: "d2_ep_solve.partial.jsonl (600/600) · d2_price_realize.partial.jsonl (600/600) · qa03_instances.jsonl · qa11/receipt.json · xi1 race receipt · wr1 descent receipt; scratchpad uh1_unharvested.py"
---

# ddm_uh1 — The unharvested-signal pass (6th standing question)

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Bars 0.172141 / 0.15. Everything
below is advisory means. Method: re-read today's receipts for signal we RECORDED but did not ACT
on. Two of the findings below are corrections to my own same-day framing (ng1) — logged per the
NO-FAKE honesty rule, not smoothed over.

## §1 RUNG P0 — a measured net −0.2206 S was filed under "falsifier fired" (ACT NOW)

The pfs1 D2 e_p ladder's 6dof_f16 point (all 600 pairs, per-pair warp-chart solve, f16) is in the
receipts as a FAILURE because it missed the pre-registered ≥0.5 falsifier bar (reached 1.2630).
But read as a RUNG, every number is already realized per-pair through the real receiver+PoseNet
(`d2_price_realize.partial.jsonl`):

- pose contribution 1.4881 → **1.2630** (mean d_pose 0.2214 → 0.1595) = **ΔS_pose −0.2251**
- price: 600×6×f16 = 7,200 B raw (codable lower; params near-constant on tail per §3) = **ΔS_rate +0.0048**
- **net −0.2206 S = 10.6% of the 2.08 gap, ~free, composable** — larger than the QA03+QA04 seg
  corrections combined (−0.0022) by 100×, and it was sitting in a "negative" receipt.

Split check (tail-only application would be dominated): tail-only 6dof → 1.4143 (net −0.073);
full 600-pair application → 1.2630 (net −0.2206). Full application wins because the chart works
WELL off-tail (§2). **Verdict-framing bug class**: a binary falsifier verdict (bar 0.5) buried a
measured 10.6%-of-gap move. Sister of gc8 §1's form-artifact sweep — this is the VERDICT-form
artifact. Routing: RUNG P0 joins the composed candidate (Knee base + P0 + QA43 stages + base
seg); the pose-constrained Knee-B gate must run WITH P0 active (shipping config), because Knee-B's
~100 road-midband drops attack the same warp cue P0 corrects — knee law (8.8% additivity) applies.

## §2 CHART SATURATION — honest correction of ng1 §1 (my own table, same day)

New measurement from the same cache: the 6-DOF pose-warp chart removes **73.8%** of non-tail
residual (0.0675→0.0177) but only **12.8%** of tail residual (0.8922→0.7775). The tail is NOT a
pose-parameter error — it is a warp-MODEL (content) error: no setting of the 6 ground-homography
params manufactures the turn-cluster image content PoseNet reads. Consequences:

1. ng1's counterfactual table (top-112 → 0.382) is CONDITIONAL on the **free-frame_0 class**
   solve (p3v2 bound, n24 ≤1e-3) — the only class that reaches 1e-3 on the tail. The "~120 B/pair"
   price shape (6 params) does **not** apply to that class; the free-class carrier price is the
   true unknown and is plausibly large (a free frame_0 delta is image-class content).
2. Within-chart tail-targeting is CAPPED at −0.074 S (measured counterfactual) — 15× smaller than
   the −1.10 free-class prize. #404 ratio: the pose-axis decision variable is this 15×, and it
   hinges entirely on whether the free-class delta has a compact carrier.
3. QA43 re-staged (ledger row amended): **stage-0** = compose RUNG P0 (§1) · **stage-1** =
   free-class GN on 8–16 tail pairs, then MEASURE THE DELTA'S STRUCTURE (rank / sparsity / DCT
   energy / cross-pair shareability) = the carrier-existence probe · **stage-2** = full k-sweep
   ONLY if stage-1 finds a compact carrier; else the falsifier fires and **pose-in-burn REQUIRED
   stands** (v10 row-12) — now with a measured reason, not a scope guess.

## §3 THE RANK-1 AXIS — the tail's within-chart correction is ONE near-constant direction

SVD over the 112 tail p_star solves: **98.06% of variance in ONE direction = pure pose-dim-0**
(components [−0.9993, −0.013, 0.036, ~0, ~0, ~0]); mean p_star ≈ [32.6, −0.05, −0.10, 0,0,0] with
coefficient spread only ±5 around 32.6. The chart pushes ONE yaw-class axis hard and uniformly on
turn pairs, then saturates. Adjacencies: (a) this is the sc1 e_p-rank-1 law re-measured on the new
base (#741's e_p↔ξ dual-use question — partial answer: the axis is real, systematic, and near-
constant across the tail); (b) it means the within-chart carrier is ~90 B (1 shared direction + a
tail flag), but §2 caps its value; (c) for the free-class probe (stage-1), the FIRST hypothesis to
test is that the free delta is also low-rank around this axis — if yes, the compact carrier
exists and the −1.10 prize opens. Temporal structure is moderate, not decisive: 24% of tail pairs
in runs ≥4 (69 runs), within-run step/spread ratio 0.487.

## §4 THE REST OF THE UNHARVESTED TABLE

| # | Signal (where it sat) | What it says | Action taken / routed |
|---|---|---|---|
| 1 | xi1's ideal 3-way bound: H(delta\|base,prev,warp) is only **1,617 B** below SMEVR realized | The token stream is CODING-SATURATED at this alphabet — QA08's mixing ceiling ≤1.6 KB (0.001 S) | QA08 ledger row flipped **CEILING-PRICED-EFFECTIVELY-CLOSED**; reopens only with the ≥48×64 granularity re-race (QA24 rider) |
| 2 | QA03 bytes/flip = 2,709/1,866 ≈ **1.45 B/flip** vs the historical water 1.27 | The correction-price invariant reproduces at a third instrument — corrections price at ~the water line, hence ~break-even forever at this base | Constant noted for canonical-equations fold at next quiet boundary; supports the white-jitter law (seg = base-quality game) |
| 3 | QA11 sensitivity spread: median 8.6e-5 vs p99 3.0e-3 (**35×**), 27% exact-zero gradients | A 3-rung {L16,L8,L4} ladder wastes most of the spread — continuous/log bit-allocation dominates | QA07 design amendment (rides QA06); wr1's sensitivity-ordered curve already dominated uniform null-snap — two instruments, same law |
| 4 | QA11 prereg `baseline_full_dseg` 0.013833 vs measured q=0 **0.0038892** (3.6×) | Stale prereg constant or a confound in the S2 protocol lineage — flagged, unresolved | Routed to sb1 adjudication (arm owns QA11 receipt); do not consume the prereg number anywhere |
| 5 | wr1 pose-safety typing exists only as a VETO list (~100 road midband cells) | The same map read in reverse = pose-POSITIVE token refinement targets (spend L16→L32-class bytes where the warp reads) — an unmeasured mirror cell | Rider on the pose-constrained Knee-B gate: emit the mirror ranking from the same atlas pass, $0 |
| 6 | Cross-axis overlap: top-40 seg-yield ∩ tail112 = 13 vs 7.5 expected (**1.7×**); tail pairs avg 20.8 vs 16.3 net flips | Pose-hard pairs are seg-richer — one shared pair-class map amortizes across both streams | Composed-candidate grammar note: single pair map member (~75 B) serves QA43 + seg corrections |
| 7 | QA04 done: +773 flips, ΔS_seg −0.000655 (QA03+QA04 = −0.0022 = 1.6% of the −0.138 ceiling) | Fourth confirmation: seg closes via BASE movement, not stored corrections | sb1 adjudicates; no new routing |

## §5 THE PANTHEON'S FOUR LENSES (what they'd find/feel that we hadn't acted on)

**EXCITED (Schmidhuber, compression-progress):** §1 — the biggest single composable ΔS of the day
was already measured and mislabeled. And §3 — the tail failure compressing to ONE near-constant
axis is maximal description-length progress: "112 hard pairs" → "one yaw-class systematic + a
content limit." **CURIOUS (Shannon/Wyner):** stage-1's structure question — is the free-class
delta low-rank around the dim-0 axis? That single measurement decides 53% of the remaining gap.
**CONFUSED (Fridrich/Contrarian):** row 4 — a 3.6× prereg-vs-measured baseline discrepancy inside
an otherwise-clean receipt; and the verdict-form artifact of §1 (how many OTHER fired falsifiers
buried net-positive rungs? — the answer for today's set: only D2; xi1's and of1's negatives were
re-checked and stay negative). **UN-ACTED-UPON (Dykstra/MDL):** rows 1+3 — a closed ceiling
(QA08) we were still carrying as DUE, and a 35× sensitivity spread we were about to spend through
a 3-rung ladder.

## §6 LEDGER + ROUTING (same commit)
- QA43 row AMENDED (stage-0/1/2 restructure + chart-saturation caveat + price-shape correction).
- QA08 row FLIPPED to CEILING-PRICED-EFFECTIVELY-CLOSED (≤1,617 B ceiling, xi1 receipt).
- QA06 row note: composed candidate + pose-constrained Knee-B gates run WITH RUNG P0 active.
- Task #775 (QA43) description updated to the staged form.
- Fire order unchanged otherwise: sb1 chain (QA05 last) → Knee-A gate → QA43 (stage-0 compose +
  stage-1 probe) → pose-constrained Knee-B (with P0) → composed candidate → local row → Stage-B
  on operator GO.
