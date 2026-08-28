# ddm_qbt2b r5 balanced-CE verdict — LANE IS BORN (werr 99.76% → 9.80%, under the 20% gate); the failure INVERTED: Road degraded 1.28% → 25.77% via the DERIVED weighted-CE prior shift (86.6% of Road errors are Road→Lane over-paint); all 5 classes majority-correct from ~step 250 — the first fully-born field this family has produced; r6 route = REVIEWED gate revision (birth = existence event) → first margin+pose stage on a born field

STORES CONSULTED: r4 verdict memo (`ddm_qbt2b_r4_extended_ce_verdict_20260828.md`, 6f5cf6a371) ·
r3 verdict + charter · #315/#686 derived-schedule law (CE births, margin sharpens) · #218
long-tail logit-adjustment lever (the prior-shift family) · m131 Lane-demand · m132
collateral law · pc2 Road-hub law · m52 never-binary / m85 matched-control. All numbers
re-computed from retained verdict payloads this session (m44).

score_claim=false everywhere. All rows [macOS-Metal/CPU frozen-scorer advisory, n32
seeded-stratified, seed 20260827]. Pointer UNMOVED (gb1 0.14811799921260607).

## 1. The run (MEASURED)

r5 = the exact matched treatment vs r4 (seed/init sha 0bedbd66…/schedule byte-identical;
single variable `birth_class_weight_mode` none→balanced; config sha 307b4e48…). Launch
counter 691, rc=0 clean at the 1,000-step cap, 4,866 s. Status
`BIRTH_STAGE_CAPPED_WITHOUT_EVENT_HANDOFF`. 200 verdicts + 200 ckpts retained (written
DEFLATED at source — the #1313 cure held: r5 consumed ~27 GB vs r4's 52 GB uncompressed).
Balanced weights verified ACTIVE in runtime curriculum_state (Lane 33.55 · Movable 16.06 ·
Road 0.865 · MyCar 0.786 · Undrivable 0.403) — the #404 binding-vs-inert proof, $0.

## 2. THE HEADLINE A/B (MEASURED, verdict_0200_step_001000 vs r4's same-step verdict)

| class | GT % | r4 werr% (unweighted) | r5 werr% (balanced) | verdict |
|---|---|---|---|---|
| Road       | 23.12 | 1.28  | **25.77** (flat: 28.1@250 → 25.8@1000) | DEGRADED 20× — the new gate-blocker |
| Lane       |  0.60 | 99.76 (never born) | **9.80** (falling; first <99.9% at step 75) | **BORN — the campaign's hardest orbit** |
| Undrivable | 49.58 | 0.49  | 5.14  | mildly degraded |
| Movable    |  1.25 | 27.84 | **0.65** | CURED |
| MyCar      | 25.46 | 0.36  | 1.46  | held |

Total flip 0.08941 (r5) vs 0.01573 (r4) — balanced is 5.7× worse on the AGGREGATE while
fixing both rare classes. Gate result: 0 of 200 verdicts pass all-5 werr<0.20 (Road alone
fails from ~step 250 on). Pose co-descended again (pose_mse 0.0040 at cap; r4: 0.0022).

## 3. THE MECHANISM (MEASURED — not starvation, PRIOR SHIFT)

- Road's balanced gradient SHARE is ~20% vs 23.2% unweighted — essentially unchanged. Its
  degradation is NOT starvation.
- Confusion decomposition at the endpoint: **Road→Lane = 86.58%** of Road errors
  (374,776 px); Lane predicted at 6.38% share vs 0.60% GT (**10.7× over-paint**). Movable
  over-predicted 2.8× (3.50% vs 1.25%), eating Undrivable (61.5% of Undriv errors).
- DERIVED: weighted CE shifts the pairwise decision boundary by log(w_c/w_c′) in logit
  space. Road↔Lane ratio 33.55/0.865 = 38.8 → ≈3.66-nat shift toward Lane. The trained
  renderer over-paints lane structure so the frozen SegNet reads Lane — the optimum moved,
  exactly as the long-tail literature (and our #218 logit-adjustment lever) predicts. The
  r4/r5 pair therefore cleanly separates OBJECTIVE (unweighted CE optimum is correct;
  dynamics cannot birth) from DYNAMICS (balanced CE births; optimum over-paints). Neither
  extreme passes the accuracy gate. verdict_scope: INSTANCE ×2 (the two weight extremes at
  this init/window/n32/seed); the prior-shift mechanism is DERIVED + measured here.

## 4. The under-appreciated positive: the field is FULLY BORN

All 5 classes are majority-correct (werr < 50%) from ~step 250-500 onward — **the first
fully-born realized field this family (or any qbt lineage run) has produced.** The #315
gate entangles EXISTENCE (birth) with ACCURACY (sharpness): r4 failed it on existence
(Lane absent), r5 fails it on accuracy (Road over-painted). Per the family's own derived
schedule law, accuracy refinement is the MARGIN stage's role — and the margin+pose stage
has never once executed on a born field (r2 ran it on an unborn field and froze at 0.2504;
that freeze is plausibly the unborn-class pathology, untested on a born field).

## 5. r6 ROUTE (the r3 memo's fork language, upgraded by the r5 result)

REVIEWED GATE REVISION (a deliberate code change with 2-pass review, never a silent
relaxation): split the birth-stage exit event into its two entangled halves —
- BIRTH EVENT (the CE→margin handoff): all 5 classes MAJORITY-CORRECT (within-class error
  < 0.50) for 2 consecutive realized verdicts. Derivation: majority-correct ⇒ each class's
  plurality basin exists in the realized argmax ⇒ the expected-flip margin objective has
  gradient support at every class boundary (the precondition r2's frozen run lacked).
- The werr<0.20 criterion demotes to a WATCH metric (margin-stage progress), not a gate.
r6 then = revised-gate config, initialization_state = r5's cap checkpoint (existence has
held since ~step 250 ⇒ handoff fires within ~2 verdicts) → the margin law runs 5,000 steps
on a born field: expected-flip minimization IS unweighted-flip annealing — the derived
corrector for the §3 prior shift (it directly prices the Road→Lane over-paint as flips) —
plus the r2-proven pose finish (119.84→4.5e-4 precedent; r5 cap pose_mse 0.0040 vs the
m110 pose budget 1.25e-4).
Falsifier (pre-registered): if the margin stage on the born field freezes (flip slope
~0 over ≥1,000 steps) with Road werr held >20%, the r2 freeze is NOT the unborn-class
pathology — the margin law's freeze becomes FAMILY-scoped and the CE weight-ANNEAL
schedule (balanced→uniform after birth) becomes the next single-variable arm instead.

## 6. STORAGE (BLOCKING — operator decision surfaced)

AP free fell to ~6 GB (below the 8.59 GB launch floor) at r5's end. Retained qbt2b custody
on AP ≈ 56 GB (r3 2.6 + r4 26 + r5 27), all deflated, certified, and consumed into verdict
memos; the payload law forbids discard. Vertigo is also 100% full (#1165 round-2 pending).
The ONLY remaining tier is local disk (/ has ~354 GB free), which the storage waterfall
gates on EXPLICIT OPERATOR OPT-IN. r6 cannot fire until either (a) operator approves a
certified cold-MOVE of r3+r4 custody (~28.6 GB) to a local cold-store path with
machine-readable certificates (move, never delete), or (b) an alternative reclaim is
named. This is the single live blocking decision.

## 7. Family state after r5

| leg | status |
|---|---|
| rate | PROVEN through training (~107.5 KB repeat-identical, unchanged) |
| pose | co-descent PROVEN ×4; finish still owed to the margin+pose stage |
| seg  | **ALL 5 CLASSES BORN (majority-correct)** — first time; residue = Road over-paint (prior shift) + sharpening, both the margin stage's derived job |

Gap to a claimable candidate remains large (flip 0.089/0.016 vs the ~0.00116 box; n32
single seed; advisory axis). This memo claims the mechanism separation + the first born
field, not a score.

## 8. Custody

r5: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r5/`
(200 ckpts, 200 deflated verdict npz, cap checkpoint inside reencoded/, RESULT.json,
launch manifest). Config sha 307b4e48e810…. §2/§3 tables reproducible from
`verdict_0200_step_001000.npz` (~20 lines numpy, executed this session).

## Observability surface
Per-layer: per-step history + per-5-step ckpts/verdicts (deflated). Decomposable: §2
per-class A/B + §3 confusion destinations + pose separately. Diffable: r4 vs r5 identical
schema, single-variable by construction. Queryable: npz/JSON on AP + repack manifests.
Citeable: config shas 4f5326a4… (r4) / 307b4e48… (r5), launch counters 689/691.
Counterfactual: any-window replay via 5-step ckpt cadence; the §5 gate revision is
testable from the retained r5 cap checkpoint without retraining.

— end —
