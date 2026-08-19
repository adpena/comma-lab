# The Optimal Vehicle, Derived From Measured Signal — 2026-08-19

**Operator prompt (verbatim):** "As we learn more and more we are gaining a ton of signal into
what optimal representation and carrier and regime and more and related and all would look
like." This memo compiles that signal into ONE design object. Every claim is labeled
**MEASURED** (receipt exists) / **DERIVED** (arithmetic from measured) / **OPEN** (named
measurement owed). Axis: all pose/seg absolutes are DALI-lineage unless stated.
`score_claim: false` — this is a design synthesis, not a row.

Base state: pointer S 0.15652626435208142 @ 176,420 B (archive 7ce46fd7), axis ledger
seg 0.030309 · pose 0.008746 · rate 0.117471 S; gap to 0.15 = 0.006526.

## §1 REPRESENTATION — what the measurements say it should be

1. **Store the task statistic, nearly raw.** The winning body's dominant section (109,792 B,
   62.2%) is a dense 5-class semantic token label map, 99.9985% identical to the DALI GT argmax
   (MEASURED, jg1 S0/S1). The representation question for seg is essentially SOLVED at the
   information level: the labels are right.
2. **The binding seg term is REALIZATION, not description.** 95.9% of the seg leg is
   render/re-segment loss (MEASURED, jg1; confirms rt1 #1075's 95%). Optimal-vehicle
   consequence (DERIVED): marginal seg effort goes to the RECEIVER/renderer path (how stored
   labels survive render→R→uint8→SegNet), and to targeted single-cell label edits that
   pre-compensate that loss — NOT to richer label representations.
3. **Edit granularity is the single cell.** Single-cell coordinate moves repair 1.55 argmax
   cells/changed token; block/dilation moves realize WORSE at every radius (MEASURED, jg1 S1).
   The optimal representation is therefore EDIT-ADDRESSABLE at cell granularity — a property
   the current token map already has and any successor must keep.
4. **The entropy model already prices the boundary.** Boundary tokens cost ~100× the field
   average; field average 0.00745 bits/token; edit surcharge modelled +4.718 bits/token vs a
   15.8-bit budget (MEASURED-on-other-body model; the REAL number is jg2's S1, in flight).
   Coder axis vs own memoryless bound: closed (#996). OPEN: representation-LEVEL rate
   (per-region/adaptive models, #869's −113 KB projection; tx1 re-pricing).

## §2 CARRIER — pose is a refittable function, not stored information

5. **The carrier re-solves against any edited frame at ~0 bytes** (MEASURED, jg1 S2: pose
   destroyed ×387 by seg edits, recovered to mean 1.073×, one pair better than shipped;
   up2/up3: 970 coefficients re-solved at ΔB=0). DERIVED: the optimal carrier is a small
   parametric FUNCTION whose values are always compile-time refit — never treat carrier
   values as fixed payload.
6. **The wall is the basis, and enlargement cannot pay.** 12-dim basis demands 6.4× its span
   at the converged point (MEASURED, up2); the int12 lattice has 1.00–1.02× headroom and rate
   is non-binding (+5 B full range); doubling the basis costs up to +0.008175 S vs a whole
   pose leg of 0.008746 S (MEASURED arithmetic, jg1 S3). The only free basis move is
   RE-ORIENTATION of the existing 12 dims (OPEN — owed, unowned).
7. **Linear frame-0 overlays are a family ceiling** (MEASURED, pk4 — subject to na10's
   lineage check of the heldout instrument). Until overturned: pose headroom lives in the
   joint compile, not in additive overlay carriers.

## §3 REGIME — objective, move class, and compile order

8. **The objective is the DALI objective, specifically.** contest-CUDA and contest-CPU score
   DIFFERENT GT by construction (MEASURED, up2 structural). The pose gap between objectives is
   **ADDITIVE**: C = 1.4061e-04 = the MSE between the two GT pose tables (MEASURED, pi2 08-16,
   confirmed by na10 via four routes to 0.007% — the 19.09× "factor" is only the population
   median of C/d_dali; per-pair ratios span 0.887–1,627 as a consequence). Seg gap 1.43×
   (MEASURED, up3). DERIVED: optimize against DALI GT only; per-pair pose decisions score
   directly on the DALI instrument; no factor transfer ever. Both axes have exact $0 local
   instruments (pose 0.99993×, seg 0.99995× of T4 — MEASURED, up2/jg1). LIVE DEBT: the older
   qs1.GT_POSE tool family (8 consumers) still carries the PyAV table (#1142).
9. **The admitted move class is realized-acceptance lattice coordinate descent.** Gradient LM
   realized worse at every damping while FD proved the gradient correct (MEASURED ×2, up2 +
   jg1); blind search priced out (#930, search-scoped). DERIVED: gradients/models PROPOSE,
   the real decode ACCEPTS. This is the regime's optimization law.
10. **Compile order is a topological sort with downstream refits.** Seg edits → carrier
    re-solve against the edited frames → rate re-encode → container search (MEASURED chain:
    jg1 composition + qs5 in-compile Schur compensation + up3's 48 B archive-vs-payload
    lesson). DERIVED: the optimal vehicle ships from a TERMINAL JOINT COMPILE PASS in which
    every downstream section is refit against upstream edits, and rate is measured at the
    archive layer, never the payload layer.
11. **Statistical regime:** pose bands 13.4× seg at equal n; pose prefix bias 2.5–4.2×
    (MEASURED, fo2h/na2) → pose gates per-pair exact or large-n seeded-random, never small-n
    pooled. Concavity sign law and non-fungible-byte placement law bind allocation
    (MEASURED, sa3/ck1 lineage).

## §4 THE DERIVED OPTIMAL FORM (compact statement)

A **DALI-objective task-statistic compiler**: (a) near-raw 5-class label map, cell-addressable,
under the best representation-level entropy model the statistics license; (b) a minimal
renderer engineered for label SURVIVAL through render→R→uint8→SegNet (the 95.9% term);
(c) a small pose FUNCTION (re-oriented basis) whose values are always terminal-refit;
(d) shipped only after a terminal joint compile: single-cell seg repair under realized
acceptance × carrier re-solve × real-coder rate at the archive layer. The LIVE jg2 chain IS
this form's first full execution on the current body (its result measures how much of the
optimal form the current body can express). DERIVED ceiling arithmetic: pose→0 pays 1.34× the
gap; seg→0 pays 4.6×; the realization term (95.9% of seg) is the largest single addressable
mass on the board.

## §5 GAP TABLE — current body vs optimal form

| Element | Current | Optimal-form delta | Status |
|---|---|---|---|
| Label map info | 99.9985% correct | ~none (solved) | MEASURED |
| Seg realization | 95.9% of seg leg | single-cell pre-compensation (jg2) + receiver survival (OPEN) | IN FLIGHT / OPEN |
| Carrier values | shipped-frame-fit | terminal refit vs edited frames (jg2) | IN FLIGHT |
| Carrier basis | 12-dim, unoriented | free re-orientation (OPEN, owed) | OPEN |
| Rate: coder | at memoryless bound | closed (#996) | MEASURED |
| Rate: representation level | inherited IHS1 model | adaptive/per-region (#869; tx1) | OPEN |
| Objective | DALI-native since up2 | both local instruments live | DONE |

## §6 CONSUMERS
jg2 (#1139, live) executes §4 on the current body · tx1 (#1141, live) prices the §5 OPEN rows ·
na10 (#1140, live) re-checks whether closed families block any §4 element · the 12-dim
re-orientation and receiver-survival rows need owners at the next free slot.

STORES CONSULTED: jg1/up2/up3/fo2h memos + custody manifests · pose_gap memory UPDATES 1–3 ·
task ledger #1134–#1141 · #996/#918/#930/#1075 receipts (via their rows) · concavity +
non-fungible-byte memories. verdict_scope: synthesis of INSTANCE/FORMULATION-scoped
measurements; no new verdicts issued here.
