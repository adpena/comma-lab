# ddm_pd9 — replace the seg SCREEN with a JOINT seg+pose Lagrange admission on the price-first pool (charter, MAIN 2026-09-17; Opus)

## Why (pd8 memo `.omx/research/ddm_pd8_price_first_pass3_on_move54_20260916.md`; gs3 A69–A70)
Price-first passes 1–3 netted −1.04e-4 → −4.59e-5 → −2.51e-5 with paying pairs 128 → 72 → 44; pass 4 projects below the
2e-5 admit bar. pd8 named the binding gate: the per-pair SEG SCREEN refuses 3,141–3,176 of ~4,291 proposals before any
credit is read, and 411 of 588 pairs "can pay for no cell". Every pass so far admitted a proposal only if it cost no seg
cells (or repaired some), then priced pose credit per real bit. That is two gates in series with different currencies. The
score has ONE currency: S = 100·d_seg + sqrt(10·d_pose) + 25·B/N. A proposal that costs one argmax cell (+1/(600·196,608)
in d_seg ≈ +8.48e-9 d_seg ⇒ +8.48e-7 S) and buys −3e-6 S of resolved-pose credit at zero net bits is a −2.15e-6 S move that
the screen throws away. pd5 measured that pose credit is a coin flip per proposal; pd6–pd8 measured that the coin flip
still pays on 44–128 pairs after re-solve. This arm measures how much the screen has been discarding.

## The rung (pd8's producer and pool; the same real prices; only the admission rule changes)
1. Base = the pointer at start (move 54 or 55 — re-derive from `reports/latest.md`; if pd8's fire lands as move 55
   while you run, re-base on it before byte-close, as pd7 did); pose base gate; pricer proof; rows re-captured.
2. Reuse pd8's cheap-half enumeration on the CURRENT field (re-price; report the rank-0 sheet cost: the decay curve
   2.163 → 3.075 → 3.279 → ?).
3. **Realize the SCREENED proposals too**: for every proposal the seg screen refused with a seg cost ≤ 3 cells (report
   the histogram of refused costs), realize it exactly as admitted ones (re-render from the shipped field; per-pair
   carrier re-solve with frame-0 repair; seg on the frozen argmax = the REAL cell delta, not the screen's estimate) and
   read its resolved-pose credit and real bits.
4. **Joint admission**: per proposal, net = 25·Δbits/8/N + 100·Δd_seg + Δsqrt(10·d_pose) on the resolved pose, all
   measured; admit by Lagrange on the SET (pd5's set pricing, iterated to the fixed point with pd7/pd8's absorb
   discipline), no seg screen — the seg term is just a term. Report separately: the admitted set's seg-costing members
   (count, cells, S), and what the old screen would have admitted from the same pool (the control: the screen's set,
   priced the same way). The falsifier: the joint set's net is not better than the screen's set by ≥ 1e-5 S.
5. Three legs on the shipped bytes' own cold decode (render from the SHIPPED field); band −2e-5 … −6e-5 S; falsifiers:
   joint-vs-screen gain < 1e-5; net > −2e-5; set re-price vs ledger > 10 %; pose projection-vs-decode gap > 1.5×.
   A fired first falsifier CLOSES the joint-admission formulation on this object (verdict_scope: formulation).
6. If it nets: byte-close, twins, cold n600 parse-back, manifest (Catalog #420), census (RLC1 rider), smokes,
   retention ≤ 2 GiB, NORMAL seal inheriting the pointer's leg. MAIN fires — CUDA first, then the sibling.
7. Memo `.omx/research/ddm_pd9_joint_seg_pose_lagrange_admission_20260917.md`; serializer commits; lane
   `ddm_pd9_joint_seg_pose_admission_20260917`; checkpoint `ddm_pd9`.

## Boundaries
As pd8's charter (no Modal, fire, packet, authorize_*; upstream/, PR trees, sealed trees, contract/receiver code,
renderer, basis, prior read-only; pd1–pd8/jrd1/jrx*/psa*/mrs* stores read-only; heavy steps via
`tools/launch_detached_process.py --done-receipt <bare-name>`; never SIGTERM a running detached encode; kill whole
process trees; the pricer refuses restarts over partial checkpoints; APDataStore ≥ 8 GiB + retention (≈ 22 GiB now);
retain ≤ 2 GiB; never write Vertigo; no ScheduleWakeup; two review passes per .py; `[no-triality] [p0-ledger-ok]`;
never a co-author trailer or AI attribution). psa2's exhaustive search runs at nice 10 on the same host. Label
MEASURED / DERIVED / INFERRED / ASSUMED; state the solver behind every pose number; the public smoke must reach its
named gate on both sides (pd8's law).

## OPTIMAL FORM
Reference forms: pd8's producer, pool, sheet pricer, set pricing with absorb, render-from-shipped-field, closed-archive
ladder. Declared delta: the admission RULE (screen → joint Lagrange) — a formulation change, the charter's point; the
screened proposals realized is SCOPE widening (report the compute). Provenance pins: pd8 memo sha 25acb0bb83fb1cc4; pd8 sealed
archive sha ddadf998ddacab9b356b9b6a01a78c845ab3d6f643dd1e3cf8f37840ae550b8a; pd7 packet move 54 (64db93dc9).

## Prior negatives accounted (operator 2026-08-15)
pd5 (credit does not compound with run length — singles + 2-token only); pd4 (selection bias — set pricing); pd6/pd7
(overlay ≠ shipped render); pd7/pd8 (absorb iteration defect; the ledger over-credits and flips sign — closed archives
decide); pd8 (dead bytes closed; smoke gates must be reached); the seg-collateral law m132 (B/H on every realized
proposal); m88 (no prefix stop); the 34.8 B lottery (twins).

Final message: as pd8's, plus the refused-cost histogram, the joint-vs-screen control table, and the count/cells/S of
seg-costing members in the admitted set, ending with the current own-vehicle frontier line (re-derived at start).

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
