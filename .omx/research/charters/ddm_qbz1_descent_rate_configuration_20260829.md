# ddm_qbz1_descent_rate_configuration — is the born object's distortion gap CAPACITY or OPTIMIZATION? (task #1324; owning memo `ddm_qbt2b_r10_doubling_adjudication_20260829.md`)

## MANDATE

Operator 2026-08-29 (standing GO, verbatim): *"Recover and respawn and continue with all"* +
*"do whatever it takes... to accomplish frontier score lowering."*
Routed finding: `ddm_qbt2b_r10_doubling_adjudication_20260829.md` measured the ONLY object in the
campaign that clears the sub-0.12 archive ceiling — `B_hat` **121,928 B** vs the 137,986 B ceiling,
**16,058 B to spare**, and **58,287 B under** the gb1 pointer. Every alternative-representation
route (#1193/#1198 swarms, nr1, rc1's successors) died on the rate axis this object already wins.
Its sole problem is distortion. The tie bar is DERIVED from the live pointer, not quoted: gb1
S = 0.14811799921260607 minus this object's own rate 25*121928/37545489 = 0.0811866 leaves
**0.0669314** of distortion allowance; the object's measured distortion is **0.327712**, i.e.
**4.90x** over. For sub-0.12 the same arithmetic gives 0.12 - 0.0811866 = **0.0388134**, i.e.
**8.44x** over. The r10 memo priced MORE STEPS (400k ≈ 8.9 days exclusive
Metal, optimistic; the pessimistic geometric read says never) — but it never asked whether the
schema *can* represent a low-distortion field at all. **Capacity and optimization are different
walls with different cures.** The r10 memo (`925760f81c`) is the BASELINE here: its own §5 records that
it did NOT order the discriminating measurement, and its §4 prices only step count. This arm separates them with a
supervised fit — hours, not days, and no scorer-in-loop descent.

## SCOPE

1. **Re-derive the r10 series at source before consuming** (never from this charter's numbers):
   run dir located by MAIN this turn —
   `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r10`
   (siblings r2/r5/r6/r7/r8/r9 alongside; `stage_05_same_budget_admission/GATE.json` present); confirm `B_hat`, distortion, `d_seg_hat`, `d_pose_hat` for r7/r8/r9/r10 and the
   estimator label (`NO2_SECTION5_HT_COMPLETE`, `selection_count: 32`). Report any figure that
   does not reproduce — the r10 memo is 8 hours old and its own §5 disclaims the LEVEL.
2. **THE CAPACITY PROBE (the arm's primary deliverable).** Fit the qbt2b schema **supervised,
   directly to the DALI-lineage GT partition** (per the #1142 wrong-objective cure — GT table,
   never a PyAV-lineage substitute), at optimal form: full schema capacity, real n, held-out
   split. Then measure the fitted field's **realized** distortion through the real render/R/uint8
   path. This is the schema's capacity CEILING — the best distortion it could ever reach if
   optimization were free.
   **Explicitly NOT fitting to the gb1 realized field**: #894 measured that class dead — a
   realized field is a NOISIER COPY of GT, and GT is already the target. Use GT.
3. **Fork on the measured ceiling:**
   - ceiling ≤ ~0.067 (tie bar) → the gap is **OPTIMIZATION**. The 8.9-day price is an artifact of
     the descent, not a property of the object. Deliverable: the named short route (warm-start /
     supervised pre-fit → scorer-in-loop finish) with a measured step-count estimate.
   - ceiling ≥ ~0.30 (near the live 0.327712) → the gap is **CAPACITY**. The born-object line is
     honestly CLOSED at FAMILY scope, and the 400k-step projection is meaningless — say so
     plainly and record the ceiling as the family's terminal number.
   - in between → report the exact ceiling and the implied step budget; do NOT round to a fork.
4. **Objective-alignment cross-check ($0, from telemetry already on disk).** the CE-mechanism row (task #1089, `ddm_ce1_allocation_ladder_verdict_20260817.md` (`77701c0445`) lineage) measured on a
   sister trainer that **81.19% of the LR budget goes to the worst-aligned objective**, and that
   the split is SCALE-INVARIANT; the sister row (task #1091, same lineage) measured the seg "wall" as **92.7% CONFIGURATION**. Neither
   was ever checked on this trainer. Decompose the r7→r10 budget the same way. If the same
   misalignment is present, the α = 0.70 prefactor is partly a configuration artifact — name the
   specific config change and its predicted effect on the fit, do not just assert it.
5. **Typed exit**: {schema capacity ceiling (realized, through the real path) · fork verdict w/
   scope · alignment decomposition · the named next measurement}. NO launch from the arm.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
  NO heavy Metal launch — fcd3 owns the scorer lane and the governed slot is not this arm's.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0 DEF CON 1000): the fitted field is a payload, not a scalar — persist
  it with sha+bytes. Bulky receipts to `/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/`.
  Check free space BEFORE materializing (AP tier per the disk rules; Vertigo is tight).
- Axis honesty: `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`, `promotable=false`.
  The r10 series is an n32 Horvitz–Thompson estimate with THREE unmet admission gates and
  `control_status = REFUSED_MISSING_REAL_SAME_BUDGET_QBW1_CONTROL` — never present its LEVEL as
  comparable to the gb1 pointer. Only the trajectory and this arm's ceiling are adjudicable.
- Recompute S FROM COMPONENTS (#877), never a rounded display field.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_qbt2b_r10_doubling_adjudication_20260829.md` §Dead-ends: reading this line's LEVEL as
  comparable to the pointer is refused (missing same-budget control); and `B_hat` 121,928 B is
  NOT a bankable rate win — bytes and distortion belong to the same object.
- Task **#894** — `ddm_na10_negative_audit_fresh_laws_20260819.md` (`3225e3a880`) carries this row (LATTICE-SOLVE REBASE REFUTED, FAMILY): a realized field used as teacher is a
  noisier copy of GT, and GT is already in the loss. This charter's §2 is designed around it.
- Task **#1041** — `ddm_pk4_optimal_form_frame0_pose_20260813.md` (`1eaff90422`), the pk4 verdict: every rung LOPO-POSITIVE in the modeled space but heldout
  NEGATIVE-or-zero in reality; pk3's 23/23 in-sample = 0/23 LOO reproduced at optimal form. **Any
  supervised fit here MUST be held out** or it measures nothing.
- Task **#1225** — `ddm_ny1_live_lineage_toy_and_reactivation_audit_20260823.md` (`301f2b4770`) carries the no-toy re-scoping: a FAMILY was closed from TOY rows — the capacity probe must run at the family's
  optimal form (full schema, real n), or it is a toy and produces no family verdict.
- Task **#1087** — `ddm_wj1_cost_error_position_join_20260823.md` (`72975fcaa1`) lineage: a 50-step end-to-end smoke became the campaign's cost model and priced every
  window 4.9× too expensive. Do not price this fit from a smoke.
- Task **#1090** — same window-pricing lineage: 70.4% of a probe run is EVALUATION, not training — budget accordingly.
- Task **#1251** — `ddm_w96a_aligned_config_renderer_window_20260826.md` (`fc915c771f`, the two-seed OFF baseline): the seed has never been varied on the sister trainer; every result is a single
  draw of unknown width. If a single-seed number drives the fork, say so.
- Task **#1250** (owning memo NOT resolvable in `.omx/research/` at charter time — verify at source or treat as unowned): an arm checked its own unchecked conditional and killed its own 8.9% claim —
  check the conditionals in §4 before asserting the alignment transfer.

## OPTIMAL FORM

- Family exemplar: the r7→r10 constrained-margin series is the reference form — memo
  `.omx/research/ddm_qbt2b_r10_doubling_adjudication_20260829.md` (commit `925760f81c`), lineage tasks
  #1315/#1316/#1317/#1318, retained storage projections at
  `/Volumes/APDataStore/pact/ddm_qbt2b_r10_constrained_margin_continuation/R10_STORAGE_PROJECTION_20260829.json`.
  Run that landed schema; do NOT invent a new one.
- SCOPE reductions declared per row (a reduced-n capacity fit is labelled and its n stated).
  MECHANISM reductions FORBIDDEN: full schema capacity, real GT table, real realization path.
- **PRIOR-LAW PREDICTION (falsifiable):** the descent is monotone on all four columns with a
  power-law α = 0.70 and NO sign of an asymptote above the rate floor — a schema at its capacity
  ceiling flattens, it does not keep descending cleanly across three doublings. So the prediction
  is **OPTIMIZATION-LIMITED: the supervised GT-fit ceiling lands well below 0.30**, and plausibly
  below the 0.067 tie bar. FALSIFIER: the supervised GT-fit ceiling lands ≥ 0.30 (within ~10% of
  the live 0.327712) — the schema simply cannot express a low-distortion field, the three
  doublings were descending toward a floor they would never clear, and the born-object line is
  CLOSED at FAMILY scope. That is a decisive and useful finding; count it plainly.

## DELIVERABLE

`.omx/research/ddm_qbz1_descent_rate_configuration_20260829.md` — typed rows: (1) r7–r10
re-derivation table w/ any non-reproducing figure named; (2) the supervised GT-fit capacity
ceiling (realized through the real path, held-out, w/ n and seed stated); (3) the fork verdict
with explicit scope; (4) the objective-alignment decomposition + named config change if present;
(5) NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS. Commit via the serializer. End with the
own-vehicle frontier line.
