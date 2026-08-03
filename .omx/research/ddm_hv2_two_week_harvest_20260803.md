# ddm_hv2 — THE TWO-WEEK HARVEST: drain, not append

**Date:** 2026-08-03 · **Charter:** operator 2026-08-03 ("a lot of signal here for us to harvest")
· **Cost:** $0, zero scorer forwards · **Pointer 0.1910828242 [contest-CPU] UNMOVED. Nothing here is a score.**
· Denominators: live best S=0.7910689 · gap 0.6189279 · 1% of gap = 0.0061893 S = 7,301 flips = 9,295 B · seg leg 0.4311790 (69.7% of gap, unmoved 8 revisions).

**The typed queue ledger is `.omx/research/ddm_hv2_harvest_queue_20260803.jsonl`** (40 rows:
3 FIRED · 5 CLOSED/VERIFIED-CONSUMED · 7 HANDED (incl. the #920 ranking row) · 12 QUEUED_SCORER ·
6 QUEUED_TRAINING · 7 DEFERRED, every row with owner + fire-condition; `.omx/state/` is gitignored
LIVE_STATE per the artifact-lifecycle policy, so the durable tracked home is `.omx/research/`).
This memo is the narrative + the tables that must not stay chat-only.

## §1 — FIRED ($0, receipts)

1. **#869/#766 exact-key prep** — both lexsort axes were ALREADY measured on disk (charter's "~7 min
   compute" was stale: `gsum` precomputed in `rs2_thin_margin_keys.npz`, exact byte marginals in
   `rs2_cell_byte_marginal.npy`). Assembled 4 exact-keyed orders + receipt at
   `/Volumes/VertigoDataTier/pact/ddm_hv2_20260803/ddm_hv2_exact_keyed_orders.{npz,_receipt.json}`.
   Independent reproduction: fm16 zero-damage tranche 486 vs fmrf 342 = the 144 false-safe cells, exactly rs2's number.
   Key CHOICE deliberately deferred to the rs2 §2.3 A/B (Q2).
2. **ss1's OWED (commit 080760d2f1)** — the live archive's pose solver was NOT in git: working-tree
   `ddm_pfs1_ep_warp_pose_solve.py` = 6-DOF fixed-s_t, f16-realized-acceptance GN (pj2 degeneracy cure),
   vs HEAD's ancestor 7-dim co-opt solver. 126 tests across the 5 importing files PASS against the
   working tree and the pg1 realized-acceptance tests REQUIRE it — the committed test suite already
   depended on uncommitted code. Also verified: `pfs1_warp_receiver.py` IS tracked (ss1's sister claim moot).
   The window's "uncommitted-work-outside-git" bottleneck is now closed except tw1 (D1).
3. **oh1 orphan #1 (commit 667485634e)** — the 19-day-open owed anchor: MEASURED Metal wall
   (achieved 0.7078× SLOWDOWN, η 0.3205, ceiling 2.2086× derived-unrealized) appended to
   `custom_sparse_adjoint_achieved_vs_ceiling_v1` with receipt-anchored timestamp; status BLOCKED→MEASURED, append-only history.

## §2 — WHAT THE CHARTER GOT WRONG (refutations, per the charter's own ask)

| charter claim | measured state |
|---|---|
| "#881/#889: route the pose re-solve to the scorer queue as one unit" | **REFUTED before I started** — `ddm_cr2r` (08-02) killed the re-solve at FORMULATION scope: floor ≥ 4.0389 S vs −0.0867 to defend = **46× over even if all 526 unsolved pairs return exactly zero**. The matched control (same solver, celldrop50 base: mean 0.0778 vs ep854 11.5904, ep854 better on 1/74) proves the solver healthy and the BASE pose-illegible. The −0.0866789 S seg+rate prize is recast as T6: fires only on a pose-carrying base (joint in-loop pose). The prerequisite (uncommitted solver) was real → closed as FIRED-2. |
| "#832: verify the $0 test ever RAN (vacuity genus)" | **It ran.** dc1 2026-08-01 (receipt 01:07:19Z); wd1 08-02 independently re-derived, reproducing ba31's row exactly. Adjudicated: the 0.264 S swing compares two prices of DIFFERENT objects; the −0.098530 / 12.44% interpolation-free bound is confirmed AND superseded (label paid ⇒ win LARGER). All prices anchored to the OLD gap 0.7918468 — re-anchor at consumption. |
| "#866: scorer-queue row" | It is a **heavy operator-GO race** (QA86a: matched budget, SMEVR ledger both arms), not a scorer-slot forward. Kill stays pre-registered: >8,201 flips for the −10,441 B step = net loss. → T1. |
| "as1's P-C: if scorer-free, fire it" | **Not scorer-free**: P-C is n600 through-R SegNet forwards with a BINDING memory gate (the #205 +66 GiB batched-scorer class; governed launcher + memory-preflight). → Q10. Premise re-validated though: as1 measured 1.606×/1.681× on the live vehicle, overturning hg1-claim-4's 1.05×-symmetric. |
| "#869 exact keys can be made exact in ~7 min" | Already exact ON DISK; prep was assembly (FIRED-1). |
| "#879's 84 UNKNOWN + QA52, #886/#887's 29+18 — consume the pre-built rankings" | Consumed by content (ids #874–#911 are ABSENT from the repo ledger — m89 measured again): p1a adjudicated 46/86 itself (29 items remain, ~26 routable); p2a's 18 → 15 open; **QA52 is DEAD** (kl1 fired 07-30; ξ-trajectory measured-FALSIFIED white; residue = the −880 B codec BUILT-UNWIRED → D7). |

## §3 — THE LEDGER-SPLIT, QUANTIFIED (#880's finding, this window)

Repo `canonical_task_status.jsonl`: 423 rows / 149 ids / numeric range 383–909. In the charter window
#815–#919 the repo holds **18 ids** ({815–828}∖{823}, 850, 871, 873, 882, 909); **every other
charter-cited id is absent** (#832 #846 #847 #866 #869 #878 #879 #880 #881 #886 #887 #889 #890 #899
#904 #913 #914 #918 #919). Downstream damage already measured: na1 10-of-14 hunts found nothing;
oh1's instrument is 53% blind (185/348 edges UNVERIFIABLE) for exactly this reason. Bridge = D4,
P0-flagged to the operator. Until then m89 binds: **cite content, never bare ids** (this ledger does).

## §4 — #920 RANKING (MAIN's requested fire condition for the freed Fable slot)

Context carried: cg1r's force ledger (ee848e88cd) measured per-flip GT-margin depth
**DIRECTION-SYMMETRIC on every edge** (Road↔Lane 1.074×; all 9 edges 1.011–1.633×) while count
asymmetry runs 1.27–15.88× — the ph4 10.05× barrier discount is **volumetric/verb-level** (a Lane
word costs ~2.5 px of depth to annihilate because Lane has no interior), NOT cheaper-per-flip erasure.
Therefore protection must be **existence/component-level**; per-flip erasure-cost levers aim at an
already-symmetric quantity.

- **R1 — Lane × ANNIHILATE existence primitive (0.157 S = 25.4% of gap).** No existence-carrying
  primitive exists (10,260-file sweep). Build: training-side Lane **word-existence force** on the
  lane_guard betti0 machinery — PREREQ = adjudicate **#822** (its two quantities,
  `realized_lane_s_units` vs `net_betti0_realized_lane_delta`, disagree in sign — handed to bn1x,
  hv2-H2) — plus grammar-side counterpart = **gt2's tracked-word grammar** (births/deaths/motion;
  handed to gt2r, hv2-H4). Build them as ONE object. Constraints from the harvest: dd1 (every Lane
  delta ≤0.32 cells ⇒ existence, not displacement, is the buyable half), as1 (Lane deficit DIFFUSE at
  16×16 ⇒ per-component, not per-cell; asymmetry GEOMETRIC ⇒ depth-weight never margin-weight),
  hs1's open first-$0-read: "is Lane component-decomposable the way Movable is?"
- **R2 — Road↔Lane × PHASE (0.110 S; the edge is 22.1% of the ENTIRE gap, pc2).** Fire **sx2 G5**
  (scorer-free, sx2's own ordering) first; then spec the phase carrier in **token VALUES** — dd1
  measured cell granularity 16.8× too coarse; sx2 refuted the integer-offset form (sub-pixel = phase).
  The displacement carrier is still a specification, not code (as1 §5.1a-3).
- **R3 — σ_cc′ (#382) port-to-TR1** via pt2's `lever_name_join` (the exact pattern pt2 used for the
  four forces). NOT in mg1's dead per-edge-scalar family (MAIN's correction stands: per-EDGE
  coefficient on per-site training GRADIENTS ≠ per-side scalar on flip counts). Resolve the
  Young's-0.377-vs-fragility-1.029 disagreement by A/B, not adjudication.
- **R4 — Movable × GOUGE (0.067 S)** — gated on Q6 (per-component δ vs decoded cx1, 3-memo
  convergent, scorer); carrier spec shared with R2.
- **BUILT_UNFIRED:** lg1 ratchet rides the next trainer launch (its distortion side has never been
  measured inside a real trainer — rs2_orphan); `road_undriv_bulk_field` stays Q10-gated (P-C).

Scorer-queue head remains **bo1's sealed mg1 hinge A/B** — unchanged.

## §5 — SCORER QUEUE (ordered; fires when ph5o lands) — Q-rows in the ledger

1. mg1 hinge-weight A/B (bo1 spec; kills relative-denominator — see T2) — seg leg 69.7% of gap.
2. **rs2 §2.3 byte-matched key A/B** — BUILT (274,631 vs 274,321 B, 310 B residual), 3 falsifiers
   pre-registered; calibrates the exact-key family (selects among FIRED-1's orders), prices every
   future drop rung, folds br1's cell_drop63 (corrected leg −79,177 B). *New at position 2.*
3. **pb3 §5 realized η** (bar η ≥ 0.006124, shipping archive; k=30 first) — the most-cited debt in
   the window (4 memos).
4. dd1 ±1-quantum Lane probe *(displaced from 2 — flag to MAIN)* · 5. mg1 per-site |m| ·
   6. mf1/dd1 per-component δ (3-memo) · 7. cb2 §8 gate (ADOPT if d_seg<0.00501579) ·
   8. hg1 F1/A13 directed Road↔Lane on cx1 · 9. ph4 O1 blind-set aiming · 10. as1 P-C (memory-gated) ·
   11. tl1 A+B · 12. sx2 G4 (ride-along on any cx1 decode).

Training/operator-GO tier (not the slot): T1 #866/QA86a · T2 **the missing ≥2-seed seg noise floor**
(blocks every "≤noise" kill in the window — mg1 §7.1, rt2; the vacuity genus in the gate layer) ·
T3 mt1 code_width 4→2 (−0.117283 S rate MEASURED; after ds32, shares the rate axis) + token_ste dither
ride-along · T4 #824 Arm B′ folded into op2's 2×2 · T5 pt2 four-forces race after bo1 · T6 the cr2
−0.0867 prize on a pose-carrying base.

## §6 — STALE-ROW CATCHES (do not route; supersedes older rankings)

- **hg1 F4 is DEAD** (mg1 §4: 0.0000% advantage to 30% of bytes freed; shuffle control 5.6–65.4%) —
  any inventory calling F4 "highest value" predates mg1.
- **#826 is DEAD as a score row** (qd1: +0.0034632 WORSE re-priced; and cell_drop50 is ALREADY the
  live base — rs2 verified the shipped cx1 lattice == `qa24_grid_keep_mask_50.npy` 768/768). Residue:
  op3's $0 re-encode-within-212-B → D7 bundle.
- CONSUMED chains verified: tw1/zb1 knee re-price → wf2 §4.4 · mt1 #1 ds32 → gd3→gd4→gd5 (in flight) ·
  #878 findings-persistence → 09fca46f37 · lr2 join → pt2 §1 · ra1 §5.4 n600 → pz1 (withdrawn-as-answered) ·
  sx1 G2 → sx2 · rz1 A3 unknown → ph4 §1 (frame_0 IS a warp; A2's premise gone → re-rank, hv2-H3) ·
  ix1/ix2/cp1 container chain → cx1 · #858 STALE (qd1) · every mt1/p1a %-of-gap figure is on a dead
  baseline (three baseline moves since) — re-anchor at consumption.

## §7 — CONVERGENT DEBT (≥2 memos naming the same owed item; sweep of 80 files, 55 with owed blocks)

4-memo: pb3 §5 realized η (→Q3) · ba31 seg/pose split (→bo1 H1) · TaskList↔ledger bridge (→D4) ·
\#766 ranking-key family (→Q2). 3-memo: per-geometry launcher memory floor + STRICT resume-geometry
gate (gd4/gd5/op2 → routed with T-tier) · mf1 per-component δ (→Q6) · token_quant_levels
pre-quantization activations (→T3 family) · #890-class bare-id unconsumability (→D4). 2-memo:
selector∈{0,1} never re-selected · pose backlog re-price 1.73×/1.42× UPWARD (op3/qd2 — fold into any
banked-pose read; wf2: ≥2.22× compound) · corpus index 9,706-vs-7,398 gap · cb2 §8 (→Q7) · bo1 F1 ·
tl1-A (→Q11) · era-frozen gap constants in cv1/gd3/gd4 headers · de1-diff (sx1/sx2) · ra1 §4.2
np.repeat AA re-race · witness_control/g111_* verdicts unread.

## §8 — WHAT I DID **NOT** DRAIN (m50 denominator)

- **Owed-block sweep:** 80 files in the 08-02..03 window, 55 with owed blocks, ~200 items. Routed
  explicitly: the 20 convergent debts + per-memo heads + charter items ≈ 70. **~130 items remain in
  their source memos**, indexed by §7's sweep (files + one-line contents are recoverable by the same
  grep set; ddm_pu2 and ddm_cr1 skipped per charter).
- **Inventories:** p1a's ~26 routable rows and p2a's 15 T1 rows are routed as ONE bundle (D7 spawn
  package), not item-adjudicated. mt1's ~36 NO_OCCUPANCY_DATA rows unrouted. cu1's 135-file wave
  still needs an OWNER (D3). p1a's 21 ALREADY-DONE are **candidate-closed only** (20/21 rest on
  ledger self-report, not receipt checks). **83.9% of the memo corpus (6,194/7,378) sits outside
  p1a's extraction window and has never been adjudicated for follow-on debt by anyone.**
- **Not verified:** SSD-tier receipts behind "never-fired" claims (e.g. KD-from-warm); earlier-window
  (07-20..08-01) owed blocks beyond what fo1/qd1/p1a/p2a already swept; whether completed-status rows
  other than #832 are vacuity-clean (spot-checked one: it was clean).

## §9 — STORES CONSULTED

main_hot_state.md (15:52Z) · canonical_task_status.jsonl (423 rows) · MEMORY.md anchors m36/m45/m50/
m89/m90/m91 · cr2r/rs2/wf2/mg1/pt2/as1/ca1/ss1/dd1/ph4/hg1/rg5/qd1(-via-sweep)/mt1/p1a/p2a/fo1/oh1/cu1/
gt2/pc2/hs1/sx2/op2/op3/mf1/rt2/rz1/tl1/cb2/pb3/cx1/pj2/gd3-5 memos · v8_increment1_design_draft §6 ·
two bounded sub-sweeps (denominators in §7/§8) · cg1r's ee848e88cd ledger relay (MAIN message).
Commits this arm: 080760d2f1 · 667485634e · (this memo + ledger). Artifacts:
/Volumes/VertigoDataTier/pact/ddm_hv2_20260803/.
