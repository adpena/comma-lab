# ddm_qbt2b r6 verdict — THE MARGIN LAW WORKS ON A BORN FIELD (flip 0.0945→0.00972, 9.7× descent; the r2 freeze is formally the unborn-class pathology) AND the Road prior-shift is fully cured (25.77%→0.39%) — but the unweighted flip objective RE-ERASED the born Lane (9.80%→99.81%), the exact σ_cc′/MCF-thin-structure collateral the derivation predicted; r7 route = lane-CONSTRAINED margin (the built #808 λ_Lane guard family), not lane-weighted

STORES CONSULTED: r5 verdict memo (`ddm_qbt2b_r5_balanced_ce_verdict_20260828.md`) · r2/r3/r4
lineage memos · #382 σ_cc′ per-class-pair surface tension (Γ-limit demands per-pair σ; scalar
length/flip terms MCF-erase the thin lane) · #808 ddm_lg1 lane-guard (λ_Lane primal-dual +
born-lane protection mask + margin-floor, built for TR1) · #809 cg1 per-class guard ledger ·
#218 logit-adjustment lever · m131 (Lane = 0.59% area, 90.1% of demand) · m132 (seg dies on
COLLATERAL not targeting; B/H decomposition mandatory) · m110 (pose budget 1.25e-4). All
numbers re-computed from retained r6 payloads this session (m44).

score_claim=false everywhere. All rows [macOS-MPS frozen-scorer advisory, n32
seeded-stratified, seed 20260827]. Pointer UNMOVED (gb1 0.14811799921260607 @ 180,215 B
[contest-CUDA T4 n600]).

## 1. The run (MEASURED)

r6 = revised-gate config (birth = existence event) + r5-cap EMA-basis warm start, the exact
§5 route from the r5 memo. Storage unblocked by the operator-approved certified cold-MOVE
(908 `certified_cold_move.v1` rows, r3/r4 → `/Users/adpena/pact_local_coldstore/`, AP 34 GB
free). Launch counter 693, pid 78175, config sha `a6f97c90…` (authorized file
`AUTHORIZED_N32_R6_5020_20260828.json` sha `0d1db159…`), init
`initialized_r6_from_r5_cap_ema_state.pt` sha `4b40acc5…` (live ∪ EMA-shadow, the ema_scope
verdict basis, 44/44 tensors). rc=0 CLEAN in 11,955 s (3.32 h) through ALL stages
(03a birth → 03 margin+pose 5,000 steps → 04 precision waterfill → 05 admission gate).
Peak RSS 2,474 MiB (ps-RSS; #1306 Metal-blindness caveat applies — jetsam view not
measured). Resume identity bit-faithful ×2 (within-stage AND across the 03a→03 handoff,
`handoff_authorized_by_realized_event: True`). All payloads retained.

## 2. The revised gate FIRED exactly as derived (#404 binding proof)

Existence-mode verdicts (werr < 0.50 ×5) passed 2/2 consecutive at steps 5+10 →
`birth_handoff_authorized=True` at step 10. The step-10 history row carries:
`gate.derived_from` = the r5 §5 majority-correct derivation · `accuracy_watch
{0.20, classes_passing 4/5}` (Road 26.07% the one watch failure — unchanged from r5,
correct: 10 CE steps move nothing) · balanced weights [0.865/33.55/0.40/16.06/0.79] live in
curriculum_state · reencode leg live. Under the legacy accuracy gate this run would have
capped exactly like r5. **The margin+pose stage executed on a born field for the first time
in family history.**

## 3. FALSIFIER ADJUDICATION — neither pre-registered arm fired

Pre-registered (r5 §5): margin freeze (flip slope ~0 over ≥1,000 steps) with Road werr held
>20% ⇒ the r2 freeze is FAMILY-scoped ⇒ weight-anneal arm next. MEASURED:

- `seg_expected_flip_realized`: 0.0945 (step 11) → 0.00972 (step 5010) — **9.7× monotone
  descent**, late plateau ~0.0095–0.0097. NO freeze.
- Road werr endpoint **0.39%** (from 25.77%) — not >20%.

⇒ **The r2 freeze (0.2504 flat) is the unborn-class pathology, now with a matched positive
control on the same law/vehicle/seed lineage.** The margin law is FAMILY-VIABLE given
existence. The §3-prior-shift corrector claim also VERIFIED: unweighted expected-flip
annealing removed the Road→Lane over-paint entirely.

Pose co-finished: pose_mse 0.22 → 5.5e-4 (final training batch; r2 precedent 4.5e-4
reproduced, this time JOINTLY with seg descent). Heldout 32-pair retained eval:
mean d_pose 3.13e-3 (HT d_pose_hat 2.63e-3) — 21–25× above the m110 budget 1.25e-4;
the live-vs-EMA + batch-vs-heldout gap is the next pose measurement, not a wall claim.

## 4. THE NEW NAMED FAILURE — Lane re-erasure by area-priced flips (MEASURED)

Endpoint per-class werr, computed from the 32 retained stage-05 payloads
(`retained/pair_*.npz`, pred vs target argmax):

| class | r5 endpoint | r6 endpoint | verdict |
|---|---|---|---|
| Road       | 25.77 | **0.39**  | prior shift CURED (the §5 prediction) |
| Lane       |  9.80 (BORN) | **99.81** | **RE-ERASED — the collateral** |
| Undrivable |  5.14 | 0.12  | cured |
| Movable    |  0.65 | 10.85 | eroding (2nd-rarest, area rank order) |
| MyCar      |  1.46 | 0.08  | cured |

Mechanism (DERIVED, and the exact #382 prediction): the unweighted expected-flip objective
prices every class at its pixel area. Lane is 0.59% of pixels — erasing the born Lane costs
~0.006 flip mass while curing the Road over-paint releases ~0.05. Erasure is the
FLIP-OPTIMAL move; the aggregate metric structurally cannot see the loss (m132: collateral,
not targeting). Arithmetic: the dead Lane is ~0.0059 of the endpoint d_seg 0.0090 ≈ 65%;
lane-perfect residual ≈ 0.0031 vs the 0.00116 box.

The weight-family 2×2 is now fully populated and CLOSED at both extremes:
unweighted CE (r4) cannot birth Lane · balanced CE (r5) births Lane but over-paints Road ·
unweighted flip (r6) cures Road but erases Lane. **Weights shift the optimum (prior shift);
no-weights erase the tail. The resolution is a CONSTRAINT, not a weight.**

## 5. r7 ROUTE — lane-CONSTRAINED margin stage (single variable, built machinery)

r7 = the r6 config + ONE change: the margin law runs under a per-class existence/werr
CONSTRAINT (λ_Lane primal-dual — minimize unweighted flips subject to Lane werr ≤ bound,
projected/dual-ascent form), warm-started from the SAME r5-cap EMA basis. This is the #808
lane-guard family (born-lane protection, built+reviewed for TR1) adapted to qbt2b — recall,
not invention. A constraint avoids both measured traps: it does not move the interior
optimum (r5's log-ratio boundary shift) and it cannot trade the tail away (r6's erasure).
Movable joins the constraint set (10.85% and eroding). Falsifier (pre-registered): if the
constrained stage stalls at flip ≫ the unconstrained 0.0097 with the Lane bound ACTIVE
(dual λ pinned at its ceiling), the constraint set is infeasible at this capacity/basis —
then the capacity/representation leg (not the schedule) owns Lane, and the m131 Lane-demand
routing (analytic lane carrier #1288 lineage) supersedes in-field training for the Lane orbit.

## 6. Stage-05 gate: honest refusal (unchanged blocker)

S_hat 1.1486 (B_hat 122,330 · d_seg_hat 0.00905 · d_pose_hat 2.63e-3), NOT admitted:
`same_budget_qbw1_control` REFUSED_MISSING (the standing Stage-05 owed control), d_pose and
S_hat gates honestly failed. Rate leg stable and repeat-identical at every stage boundary
(archive 107,114 B, parseback_exact ×4 stages, packet 107,006 B).

## 7. Family state after r6

| leg | status |
|---|---|
| rate | PROVEN through training (~107.1 KB repeat-identical, 4 stage boundaries) |
| pose | joint co-descent + finish PROVEN on a born field (5.5e-4 batch / 3.1e-3 heldout); budget gap 25× = the next measurement |
| seg  | margin law FAMILY-VIABLE (9.7× descent, no freeze) · Road/Undriv/MyCar cured · **Lane = the named residual, owner = r7 constrained margin** |

## 8. Custody

r6: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r6/`
(RESULT.json complete=True · per-stage checkpoints + reencode tars + 32 retained stage-05
payloads + 2 birth verdict npz). Cold-move manifest:
`…/qbt1_trainer/qbt2b_coldmove_20260828_relocation_manifest.jsonl` (908 rows). §3 trajectory
reproducible from `stage_03_end.pt` history; §4 table from `stage_05_same_budget_admission/
retained/pair_*.npz` (~25 lines numpy, executed this session).

## Observability surface
Per-layer: 5,010 per-step history rows (loss/flip/pose/τ) + per-5-step ckpts + stage-boundary
reencodes. Decomposable: §4 per-class from retained argmax; flip native-interface vs realized
both logged. Diffable: r6 vs r5 single-variable (gate revision + init); r7 vs r6
single-variable (constraint). Queryable: RESULT.json + npz/tar on AP. Citeable: config sha
a6f97c90… / authorized 0d1db159… / init 4b40acc5… / counter 693. Counterfactual: any-window
replay via periodic ckpts; the §5 constraint law is testable from the same r5-cap basis
without retraining the birth stage.

— end —

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `mcf_minority_erasure_inevitability_v1` — `tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704` (`tac.canonical_equations`). **Relation:** DOMAIN-EXTENSION CANDIDATE (law's vehicle is `softmax_of_sdf_levelset_witness`).

The margin law descends 9.7× on a born field (flip 0.0945 → 0.00972) and the unweighted flip objective RE-ERASES the born Lane (9.80% → 99.81%) — the memo names the mechanism as σ_cc′/MCF thin-structure collateral, which is this law: the perimeter-gradient flow is motion by mean curvature, so high-curvature thin Lanes erase FIRST and inevitably. The law's registered cure (a per-class area/volume constraint) is what r7's lane-CONSTRAINED margin is.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
