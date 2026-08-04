# ddm_fl2 — force-class-edge ledger recovery: #920/#921 protection-debt head

**Arm:** ddm_fl2 (recovery of the a326693 arm that died on usage-limit, committed nothing).
**Axis:** `[macOS-CPU scorer-free advisory]` — 0 scorer forwards run by this arm. score_claim=false,
promotion_eligible=false. Own-vehicle frontier **UNMOVED** (this arm does not byte-close).
**Deliverable:** the per-class × per-edge × per-force helps/harms/protections ledger, populated and
priced, resolving the #920 (protection-debt head) and #921 (adoption-map head) tasks.
**Surface extended (not rebuilt):** `src/tac/witness_control/force_class_edge_ledger.py`
(the #809 ddm_cg1/cg1r module). JSONL dump: `.omx/research/ddm_fl2_force_class_edge_ledger_20260804.jsonl`
(69 rows; the cg1 20260803 dump preserved as history).

Binding-discipline note (m89 task-ledger split): #920/#921 are **harness TaskList** rows, not repo
`canonical_task_status.jsonl` rows — they resolve by CONTENT, not id (na1: 10/14 ids don't resolve).
Worked entirely from content + primary artifacts (cg1r JSON, ca1, gk1, as1, gt2, pc2, the module).

---

## Deliverable 1 — coverage populated + the HONEST denominator

**Before fl2:** 44 forces, 57 rows, 26/616 cells with explicit rows.
**After fl2:** **48 forces, 69 rows, naive 35/672, honest 35/266.**

### What I added toward coverage
The single highest-value cells were the **8 remaining edges of the LIVE objective**
(`tr1_seg_leg_composite`), which previously covered only Road↔Lane (hub + directed). From cg1r n600
(`ddm_cg1_directed_edge_margin_n600.json`, git 4e108ccb59, scorer_forwards_run=0) the 9 unordered edges
are now all accountable, and they **sum to 0.431 S = m06's whole seg gap** on the live cx1 vehicle:

| edge | flips | S (=100·flips/117,964,800) | count_asym | depth_asym | verdict |
|---|---|---|---|---|---|
| Road↔Lane (hub, pre-existing) | 235,148 | 0.19934 | 3.65× | 1.074× | HARMS |
| Road↔Undrivable | 89,545 | 0.075909 | 1.68× | 1.039× | HARMS |
| Road↔MyCar | 63,027 | 0.053429 | 3.02× | 1.306× | HARMS |
| Undrivable↔Movable | 61,892 | 0.052466 | 2.34× | 1.019× | HARMS |
| Road↔Movable | 57,225 | 0.048510 | 1.61× | 1.011× | HARMS |
| Lane↔MyCar | 903 | 0.000765 | 1.36× | 1.633× | HARMS |
| Lane↔Movable | 681 | 0.000577 | 12.35× | 1.199× | HARMS |
| Movable↔MyCar | 135 | 0.000114 | 15.875× | 1.186× | HARMS |
| Lane↔Undrivable | 84 | 0.0000712 | 1.27× | 1.285× | HARMS |
| **Σ 9 edges** | **508,640** | **0.4312** | — | — | — |

MEASURED (flip mass, count/depth asym). **VERB CAVEAT (verdict-scope INSTANCE):** verb=DISPLACE is the
mass-conserving NULL; the TRANSFER/DISPLACE verb-mass split is gt2's per-edge measurement and is
UNMEASURED for these edges — and `law.asymmetry_vehicle_transfer` forbids inferring TRANSFER from count
asym. Only the flip mass (→ `magnitude_s`, kind=description_gap) and the directed asymmetry are measured.

### The honest denominator (the m50 confound, INVERTED)
The naive 672 = 48 forces × 14 counts a per-class/per-edge cell for EVERY force — but **29 of 48 forces
are structurally GLOBAL/FAMILY** (a metric aggregate, a family verdict, a single global scalar). Their
per-scope cells are **N_A, not owed.** Counting them as UNMEASURED-implicit reports a debt that does not
exist. `coverage()` now reports both:

- **naive 35/672** (retained for continuity; `scope_cells_possible` is test-pinned).
- **honest 35/266** among the **19 forces PROVEN to act per-scope**. The naive count overstates the
  debt **~2.5×**.
- Of the 29 global/family-only forces: **7 are structurally aggregate** (METRIC/FAMILY → per-scope N_A,
  the only true non-debt) and **22 are global-as-applied LEVERs** (per-class-decomposable in principle →
  a genuinely-owed subset, distinct from N_A). Truly-N_A cells = 7×14 = **98**; the in-principle-owed
  universe = 672 − 98 = **574**.

**What remains uncovered (honest):** `tr1_seg_leg_composite` is now the only force with a complete
per-scope map (5/5 classes, 9/9 edges). The 22 global levers (margin_floor, tau_softplus_tau,
seg_focal_gamma, fisher_density, …) are measured globally and their per-class consequences are UNMEASURED
— they need a scorer pass (owned by pt2's duty-to-measure queue; do NOT run here — bz1 holds the slot).
The production forces (region_paint, shared_grid_token, component_existence, stroke) carry 1–2 class cells
each; their remaining class cells are UNMEASURED.

---

## Deliverable 2 — the two #920 head primitives, priced with their REALIZATION half

The cg1r-ranked protection-debt head is two primitives. **Both head magnitudes are DESCRIPTION PRICES
(measured gap/flicker-share = CEILINGS), NOT realized-through-R correction prices.** This is now
structural in the ledger via the new `magnitude_kind` field (`description_gap` vs `realized_through_R`),
and each primitive carries an explicit REALIZATION-HALF row.

### (a) Lane × ANNIHILATE — existence primitive
- **Description price: 0.1575 S** (MEASURED, INSTANCE) = Lane's total flip mass 185,801 × 8.477e-7 S/flip.
  Word-annihilation 58.23% (9,655/16,581 GT Lane comps); 26.90% of Lane's own population lost.
- **Realization half: UNMEASURED, ~0 with every BUILT actuator** (`tr1.lane.annihilate.realization`):
  aggregate re-inflation is MEASURED **+0.2459 S HARMS** (as1.grow_lane, all 10 directed sides p<0.5);
  gated-16×16 area-move on Lane is NEUTRAL (3/160 profitable cells, diffuse); the existence-carrying
  primitive is ABSENT on every path (as1.lane_presence_gap, 10,260-file sweep). **0.1575 is the ceiling
  #934 aims at, not a delta in hand.** Realization is #934 (Lane existence hinge A/B), pose-vetoed (#383).

### (b) Road↔Lane × PHASE — positional DOF
- **Description price: 0.110 S** (MEASURED, INSTANCE) = the flicker-typed 57.6% of the Road↔Lane edge
  (pixels whose GT label changes between frames), the most flicker-typed and least tie-like edge.
- **Realization half: UNMEASURED, ~0 with current carriers**
  (`pc2.tr1_phase_dof.road_lane.realization`): there is NO per-pair positional/contour DOF anywhere in
  the live representation — tokens_delta is 4 AMPLITUDE numbers per 16×16 cell (no phase), and all 49
  config keys have carrier/lane_render_band ABSENT. **0.110 is the ceiling bz1's phase-field aims at.**

**The unifying law (cg1r, corroborated here):** protection is EXISTENCE/positional-DOF level, not
per-flip. Realized per-flip depth is direction-symmetric (Road↔Lane 1.074×; all 9 edges 1.011–1.633×)
while COUNT asym runs 1.27–15.875× — a per-flip-cost lever aims at an already-symmetric quantity. The
debt is that neither existence-carrier (Lane) nor contour-DOF (phase) is built.

---

## Deliverable 3 — #921 adoption-map head: constant provenance

Two constants encoded as accountable FORCE rows so the ledger HOLDS ca1's finding (results → system
intelligence, not a chat/memo). **Neither is on the shipped decode's numeric path today (ca1 §6: 0 live
ΔS).** Sister of m51 (unladdered governance knobs) + gk1.

### `total_archive_ceiling_bytes = Literal[200000]` → UNLADDERED (ladder class 4)
`direct_description_carrier_compose.py:1334,:1441`. **NO derivation exists** — a frozen pydantic
`Literal`, not config-overridable. Sized for the ~52.5 KB-base era (added_budget max 147,456 ⇒ implied
base ≤ 52,544 B); at the live base 353,805 B, `run_ddm_v9_carrier_compose.py:2371` computes
`max(0, 200000−353805) = 0` ⇒ every ladder rung collapses to a 0-byte budget, the composer admits
nothing, and `:2513`'s plateau_falsifier reads that starvation as a genuine plateau → **a false KILL**
(the `a-probe-that-cannot-return-the-negative` genus, with the failure direction toward KILL). 200000 is
also the repo-wide round `approx_receiver_closed_target_bytes` box, ~4.7% **looser** than PR130's real
191,052 B floor. **DORMANT today** (composer blocked, fails own tests, last moved 2026-07-22). **Verdict:
HARMS (armed landmine), INSTANCE.** Provenance rung = **unladdered**. ca1 O2: delete or re-derive from
the live base before the composer is re-fired (I cannot edit — the compose file is edit-protected from
this arm).

### `thr_wall = 2.5e-4` → CORRECT-ARITHMETIC / STALE-TARGET; owed adjudication
`ddm_p3v2_optimal_form_pose_resolve.py:606` (+ `ddm_p3v2_finalize_from_cache.py:110`, no comment).
`sqrt(10·2.5e-4) = 0.050000` recomputes exactly. But **0.05 is 3.275× the PR130 pose bar (0.015268)**,
10.72× looser in d_pose units (bar `d_pose = 0.015268²/10 = 2.3311e-05`) — a config can PASS the wall and
still be 3.275× short of the competitive pose term. Whether the wall is deliberately formulation-scoped
(binding at 0.05) or should bind at the bar is an **owed adjudication (ca1 O4), NOT a measured defect**
→ ledger verdict **UNMEASURED**. gk1: the read-sites live in `experiments/`, OUTSIDE the guard's scanned
subtrees, so the constant is unguarded. Provenance rung = correct-derivation / stale-target.

**Reconciliation with the PR130 pose bar:** the PR130 floor decomposes seg 0.02966 + pose **0.015268** +
rate 0.127214 = 0.172142. thr_wall's 0.05 is 3.275× that pose term. The gap-decomposition equation
(`tac.canonical_equations.gap_decomposition_against_floor_20260802`) is the canonical home for the pose
bar; thr_wall should adopt 2.3311e-05 unless its consumer deliberately wants a looser formulation-scoped
gate.

---

## Deliverable 4 — the consumer (stated explicitly)

These protections feed the **POSE-CARRYING BASE**:
- **#934 (Lane existence hinge A/B)** is the realization of the **Lane × ANNIHILATE** existence primitive
  (ceiling 0.1575 S). It targets the recall failure (ANNIHILATE:BIRTH = 16.4×) that a per-flip lever
  cannot reach; MUST be pose-vetoed (#383: pose after frozen seg).
- **bz1 (phase-field row)** is the realization of the **Road↔Lane × PHASE** positional DOF (ceiling
  0.110 S). It supplies the per-pair contour/positional DOF the live tokens_delta lacks.
Both are ceilings until their carriers are built; neither ΔS is in hand.

---

## STATE THE BOUNDARIES
- **Scorer-free.** 0 forwards. No S moved; no byte-close. The 8 edge magnitudes are cg1r's cached n600
  argmax (advisory), not a fresh eval.
- **INSTANCE-scoped** where labeled: the edge magnitudes are the live cx1 vehicle; they will differ on
  other vehicles (`law.asymmetry_vehicle_transfer`). The verb split for the 8 new edges is **UNMEASURED**
  (DISPLACE is the null placeholder). The realization prices are **UNMEASURED** (carriers unbuilt).
- **Did NOT touch:** the compose ceiling constant itself (edit-protected file); the scorer; the byte-close.
- The honest denominator (266) counts only proven-per-scope forces; the 22 global levers are owed, not
  N_A — I did not measure their per-class channels (needs the scorer slot bz1 holds).
- `total_archive_ceiling_bytes` could not be re-derived here (forbidden file). Flagged + provenance-rung
  assigned; the edit is ca1 O2's, owned by whoever re-fires the composer.

## Follow-ons (fire / fold / queue)
- **QUEUED (bz1's scorer slot):** per-class measurement of the 22 global-lever forces (pt2 duty-to-measure
  queue already holds seg_focal_gamma/fisher_density/head_natural_grad/tau_softplus_tau). Do NOT run a 2nd
  n600 job.
- **FOLDED:** the two #920 realization halves are now the explicit ceilings for #934 and bz1 (rows land).
- **FIRED (this arm):** the #921 adoption-map head is resolved — both constants have a provenance rung in
  the ledger, thr_wall reconciled to the PR130 pose bar (3.275× looser), ceiling flagged unladdered +
  era-stale 6.73×. ca1 O2 (edit/retire the 200000 ceiling) remains owned by the composer re-firer.
- **QUEUED (owner: whoever re-fires the composer):** ca1 O2 — delete or re-derive `total_archive_ceiling_bytes`
  from the live 353,805 B base before the v9 carrier composer is re-fired, else it silently zero-budgets.

---
**Own-vehicle frontier: S = 0.7910689 @ 353,805 B [macOS-CPU advisory] — UNMOVED** (this arm is
scorer-free analysis + ledger build; it does not byte-close).
