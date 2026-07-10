# P1 SEAT S3 — STAGED-TRAINING / RECONCILIATION CONTROL (independent position)

STORES CONSULTED: `t5_crucible3/{CONVENING,DELTA_GROUNDING,ORCHESTRATION_LEDGER}_20260709.md` (the P1
pack) · `docs/operating_manual_craft_handoff.md` (§4 re-derive-don't-recognize; §5 label
MEASURED/DERIVED/INFERRED/ASSUMED; §6 attack-own-conclusion; §7 answer-first-then-risk) · PRIMARY
ARTIFACTS re-read for my charter, not memo-trusted: `SPEC_v8_perclass_decomposition_20260708.md §3/§4/§8`
+ `v8_increment1_design_draft_20260709.md §2/§3/§4/§9` (reconciliation + staged training + the REVISE
folding) · REAL SCAFFOLD SURFACES traced on-disk 2026-07-09: `src/tac/boundary_math/laguerre_logit_offset.py`
(b_c: `menon_logit_adjustment_offsets` / `damped_newton_ot_offsets` / `power_diagram_argmax` /
`hard_cell_masses` — all closed-form, out-of-gradient-loop), `src/tac/boundary_math/margin_conditional_residual.py`
(#226 `conditional_position_bits` / `waterfill_select`), `src/tac/witness_control/jacobian_basin.py`
(pose-conditioning sensor: `render_grad_energy_per_class`, `conditioning`, `JacobianBasinConfig` f_basin/
quorum_q, `would_have_fired`), `src/tac/witness_control/perclass_verdict.py` (`per_class_dseg_fields` →
`d_seg_by_class` + `flip_share_by_class`), `src/tac/canonical_equations/pose_jacobian_basin_conditioning_20260709.py`.
NOT CONSULTED (independence constraint, binding): sibling seat positions S1/S2/S4/S5/S6 of THIS crucible.
crucible-2 S3 read only for the seat PATTERN, not its v7.5 content.

T5 CRUCIBLE-3 (task #380, v8 per-class-decomposition optimal FINAL FORM). Seat charter: costate #247 /
Dykstra #73 — **OPEN Q3** (staged fields-vs-SDF then paint-vs-frozen-scorer; theft-migrates-to-composite
risk-2; b_c-out-of-loop guard; merge→diff→correct fixed point) + **OPEN Q4** (scorer-rule routing:
chroma-first/luma-reserved SCOPED to Road/Undriv; DIFF-frame1-only; MEASURE-pose-on-composites as the real
guard) + **the pose-conditioning gate control law** (Q5 gate on the v8 DECOMPOSED trunk). `$0`, no GPU, run
dirs read-only, #205 STOPPED (box free). Pointer contest-CPU **0.19110 UNMOVED** — everything here is
[macOS advisory · research-signal · NON-PROMOTABLE] MEANS; only a byte-closed `upstream/evaluate.py` n600
row < 0.19110 moves it. Remaining gap to sub-0.15 = **0.0411** (0.19110 − 0.15); all relative-significance
claims below use that denominator. `[no-triality]` (position doc; P2 synthesis owns leg propagation).

---

## ANSWER FIRST (the three design commitments in one paragraph each)

**Q3 — the theft firewall is TWO-LAYER, and the staged split gives a natural increment where the d_seg
BET is MEASURED before the paint stage exists.** Stage-A (fields vs EXACT SDF targets, ∂φ_c/∂θ_{c'}=0)
closes the *training-time gradient* theft. But the SCOPED decoupling (increment-1 review-A, MEASURED shared
Road+Undriv θ_bulk = ~63% of flip mass; global b_c couples all of a class's ties) means the decoupling does
NOT close the *residual composite-argmax reassignment* — that channel re-enters at the PAINT stage (Stage-B).
My design closes it with THREE independent controls: (i) **b_c calibrated OUT of the scorer-gradient loop**
(closed-form flip-weighted OT/Menon — the `laguerre_logit_offset` surface satisfies the review-E guard BY
CONSTRUCTION because it is closed-form, never a training target); (ii) DIFF is **frame1-ONLY + per-class
channel-split** so the paint correction operates inside ONE class's argmax-cell and cannot reassign another
class's cell; (iii) the merge→diff→correct fixed point is a **projection loop**, not a cross-class
optimization — no theft gradient EXISTS in a projection. **The control-optimal increment-1 SEQUENCING falls
out of the staging: increment-1a = Stage-A fields → composite argmax → flip-weighted b_c → d_seg-through-R +
byte-close (NO paint, NO P-C needed — this is the whole decoupling BET, measurable at the cheapest cost);
increment-1b = the paint/Dykstra stage, GATED behind P-C.** This de-risks the apparatus×5 opportunity cost:
v8's central d_seg thesis is falsifiable at Stage-A cost before any paint machinery is committed.

**Q4 — the frame-split is a STRUCTURAL firewall; the channel-split is a SOFT heuristic; do not conflate
them, and route the pose safety through the structural one.** Chroma-first/luma-reserved is correctly scoped
to Road/Undriv paint ONLY (chroma-separable) and is NOT over-claimed in increment-1 because Lane (41% of
Road's flips, luma-separable) is a SEPARATE ANALYTIC carrier, never paint-repaired. But the routing is a
warm-start that changes the Dykstra projection ORDER (hence CONVERGENCE SPEED = a score-neutral, FREE
wall-clock lever), NOT a correctness guarantee. The correctness guard is TWO facts: (a) **∂d_seg/∂ξ ≡ 0
EXACTLY** (P-2 structural: SegNet reads only frame1; ξ shapes only frame0) → the frame0/frame1 split is a
HARD firewall — pose can NEVER disturb d_seg; (b) the (luma,chroma) triangularity is CONDITIONAL (holds only
in the high-spatial-freq annulus band; low-freq chroma recolors pass into pose). So the composite's pose
safety rides the frame-split (structural, exact), never the channel-split (heuristic, conditional). The REAL
guard is **MEASURE pose on the composite at the Dykstra fixed point** — the routing only makes it converge
faster.

**Pose-conditioning gate — on the v8 DECOMPOSED trunk the conditioning readout is PER-CLASS, and because v8
decouples the classes it can fire when the DOMINANT edges are conditioned even while a tiny class lags —
which v7.5's shared trunk structurally could not do.** The gate = **(per-class d_seg BASIN on {Road, Undriv}
= the ~40% + ~8% dominant mass, remaining within-stage descent < 5%, read from `perclass_verdict.
per_class_dseg_fields`) AND (jacobian_basin σ_min gate, f_basin=0.9 entry correction, quorum_q=0.8,
hysteresis H=3)**, both required (σ_min↑ is NECESSARY-not-SUFFICIENT, P-6). Never-reached fallback = ship
banked R1 dxi (d_pose 0.001610 → 0.127, 7.2 KB, n600 authority) → the in-basin pose-finish is an OPTIONAL
improvement over a banked floor, NEVER a launch dependency (this resolves P-5's HONEST FLAG: we never bet
the run on unvalidated in-basin efficacy).

---

## Q3 — STAGED TRAINING + RECONCILIATION CONTROL (the theft firewall)

### Q3.1 The two-layer theft model (why one stage is not enough)

The measured theft SPEC_v8 decouples: **Lane 13.8× / Movable 4.6× stealing Road** through the SHARED-FEATURE
gradient (MEASURED, DELTA §J). ∂φ_c/∂θ_{c'}=0 (decoupled parameters per carrier) closes THAT — it is the
whole decoupling bet. But the increment-1 draft's own review-A (MEASURED, re-read at `v8_increment1 §2`)
scopes the decoupling HONESTLY: it closes the shared-feature gradient theft, **NOT** the residual
composite-argmax reassignment, because (a) the Road+Undriv field shares θ_bulk for ~63% of flip mass and
(b) each b_c is a GLOBAL scalar coupling all of that class's ties at once. So there are **TWO theft channels**:

| channel | mechanism | closed by | verdict_scope |
|---|---|---|---|
| **T-grad (training-time)** | shared conv features → cross-class gradient path | Stage-A decoupled fields (∂φ_c/∂θ_{c'}=0), NO end-to-end | the decoupling BET — UNPROVEN through R until increment-1a |
| **T-comp (composite-time)** | b_c global coupling + shared θ_bulk + paint→R→SegNet gradient | b_c-out-of-loop + frame1-only channel-split DIFF + projection-not-optimization paint | design-closed; MEASURED at paint stage (gated behind P-C) |

The FORBIDDEN pattern is **end-to-end paint→R→SegNet training** — it re-opens T-comp in the score gradient
(SPEC §8(3)). My position CONFIRMS this forbidden and adds: the reason it re-opens theft is that
end-to-end makes b_c and θ_bulk jointly optimized against d_seg, so the global-bias coupling becomes a
theft-like actuator (review-E). The guard is not "train carefully" — it is "b_c NEVER touches the
scorer gradient," which the closed-form surfaces enforce structurally.

### Q3.2 The b_c-out-of-loop guard — SATISFIED BY CONSTRUCTION, but the TARGET is a build gap

**MEASURED (source-inspection, `laguerre_logit_offset.py`):** every b_c path is closed-form and takes
`phi` + `target_masses` as INPUTS, never a d_seg loss — `menon_logit_adjustment_offsets` (the −τ·log(π)
prior heuristic), `damped_newton_ot_offsets` (exact damped-Newton semi-discrete OT to match SOFT cell
masses, Kitagawa–Merigot–Thibert dual). Neither is inside `mx.grad` of a scorer loss. **⇒ the review-E
"b_c never jointly optimized against d_seg" guard is satisfied BY CONSTRUCTION, not by discipline** — this
is the strongest form of the guard (structural, not procedural). GOOD.

**THE BUILD GAP I name (grounding-verified):** the surface matches to `target_masses`, and the only masses
it computes are `hard_cell_masses` / `soft_cell_masses` = **AREA fractions**. **N-1 MEASURED that
area-mass-matching HURTS** (ot_newton 0.00487 WORSE than no_offset 0.00272; OT enlarges the rare-Lane cell
to hit its 0.59% area → over-predicts Lane → SegNet penalises; verdict_scope: **FORMULATION** = mass-matching
to raw GT area as a d_seg surrogate). The OPEN reformulation (DELTA §H N-1, DELTA SETTLED) is
**flip-weighted target masses** — match the argmax to WHERE THE FLIPS ARE, not raw area. **The flip-weighted
target-mass computation does NOT exist in `laguerre_logit_offset` today** (grep confirms no `flip_weight` /
`flip_mass` path in `boundary_math/`). It DOES exist as a sensor: `perclass_verdict.flip_share_by_class`
(= flips_c / total_flips). **My design VALUE:** `target_masses = flip_share_by_class` (from the realized
through-R argmax), NOT `hard_cell_masses`. Provenance: **DERIVED** (N-1 says area HURTS, flip-weighted is
the sole open arm). The wiring `perclass_verdict.flip_share_by_class → laguerre_logit_offset.target_masses`
is an **OWED BUILD** — flagged for P2 (open question #1).

### Q3.3 The merge→diff→correct fixed point — a PROJECTION LOOP (and the honest Dykstra caveat)

The SPEC frames step-4 as "Dykstra alternating projections onto (argmax-cell ∩ pose-tube) in channel-split
coordinates (#73 reborn)." My position DERIVES why this is the right control AND flags where the framing
over-claims:

- **WHY projection is the right control (DERIVED):** a projection loop has **no objective and no cross-class
  gradient** — it repeatedly enforces two feasibility constraints. This is exactly what closes T-comp: there
  is no d_seg loss whose gradient could steal mass between classes; there are only two SET-membership
  enforcements. The paint frame1 RGB is pushed to satisfy: **(SET-1 argmax-cell)** each pixel's painted RGB
  argmaxes through R to its INTENDED class, and **(SET-2 pose-tube)** the luma stays inside the warp-coherent
  band the ξ carrier needs. Iterating the two = the reconciliation.

- **THE HONEST CAVEAT (§6 attack, folded here):** von Neumann/Dykstra convergence is guaranteed for
  **CONVEX** sets. SET-1 = {RGB : argmax_c SegNet(R(RGB))_c == intended} is **NOT convex** (SegNet is a deep
  nonlinear net; the argmax pre-image is a non-convex region). So this is a **LINEARIZED / local** projection
  loop, NOT a globally-convergent convex Dykstra. Calling it "Dykstra" is aspirational precision — the honest
  control is: **linearized alternating projection with (a) a MAX-ITER cap, (b) a per-iteration d_seg
  monotonicity check (reject an iterate that RAISES composite d_seg), and (c) a regression guard** (if the
  fixed point does not beat the pre-paint composite d_seg, KEEP the pre-paint composite = the unpainted
  argmax + counted residual). Provenance: DERIVED (SegNet non-convexity is structural). This is a **standing
  seal check** — the "Dykstra guarantees the fixed point" phrasing is exactly the aspirational-label residue
  the crucible catches (numbered-cross-ref drift class).

- **The unpaintable residual has a REAL consumer (MEASURED surface):** `margin_conditional_residual.py`
  #226 — `waterfill_select` admits only flips with `net_value > 0` AND marginal cost `< 1.27 B/flip`, coding
  position at `conditional_position_bits` = log2 C(|B|,K) over the decoder-KNOWN low-margin boundary set. So
  step-4's "unpaintable residual → counted sidecar" is not a promissory note — the coder exists and is
  bit-exact. What the residual COSTS at the composite is MEASURED-owed (rides the paint stage).

### Q3.4 P-C is the BLOCKING PRECONDITION — and it must report THREE numbers, not one

**P-C gates the entire paint stage (increment-1b).** SPEC §6 gate (b): P-C runs BEFORE the paint stage is
designed. My position: **P-C is the RIGHT decisive measurement AND it is GOVERNED-HEAVY** — it is a full-P
n600-through-R SegNet forward, the exact +66 GiB path that OOM'd #205 (DELTA §CONVENING §4; increment-1 §6
MEMORY GATE). It is **NOT $0** and does NOT run in the P4 recess; it rides `tools/launch_witness_run.py`
with the memory-preflight. The box is free (#205 STOPPED) but the memory-preflight still gates it.

**The three outputs (folding the review-9 sub-finding, which is load-bearing):** flat-paint FAILS (0.0064
floor MEASURED; increment-1 §9) ⇒ adequate texture needs class-typical statistics ⇒ those are VIDEO-DERIVED
= a **COUNTED seed floor** that neither P-A (oracle UPPER bound, real-frame texture) nor a naive P-C has
isolated. So P-C must report, **per class**:

| P-C output | what it decides | provenance |
|---|---|---|
| `flat_fill_dseg[c]` | the pure-geometry floor (zero counted seed) | MEASURED-owed (governed-heavy) |
| `procedural_fill_dseg[c]` | procedural texture, still zero counted seed | MEASURED-owed |
| `counted_seed_bytes[c]` | bytes of class-typical statistics to reach ε of the P-A oracle | MEASURED-owed |

**Go/no-go (my VALUE):** interiors-near-free is CONFIRMED only if `procedural_fill_dseg[c]` lands within
δ_R (0.0196, DELTA SETTLED) of the P-A oracle at zero counted seed. If it needs `counted_seed_bytes > 0`,
that byte cost **enters the rate ledger and is currently UN-ACCOUNTED in the 0.061/0.140 numbers** — a
coupling to S2/S4's residual-coder decision I flag (open question #3). verdict_scope of "interiors
near-free": **CONDITIONAL on P-C** (DELTA §F P-A note — the oracle is an UPPER bound; the generated-paint
floor is UNMEASURED). Relative-significance: the counted-seed-floor, if it lands at even 0.01 S, is
**24% of the 0.0411 remaining gap** — not negligible, must be measured before the paint stage is committed.

---

## Q4 — SCORER-RULE ROUTING (frame-split structural, channel-split heuristic)

### Q4.1 Two firewalls, different strengths — DO NOT conflate

The SPEC bundles "DIFF frame1-only" and "chroma-first/luma-reserved" as one routing story. They are TWO
firewalls of DIFFERENT strength, and the composite's pose safety must ride the strong one:

| firewall | claim | strength | basis |
|---|---|---|---|
| **frame0/frame1 (temporal)** | ∂d_seg/∂ξ ≡ 0; pose (frame0) cannot disturb d_seg (frame1) | **STRUCTURAL / EXACT** | P-2 proof: SegNet reads x[:,-1]=frame1 ONLY (modules.py:108); ξ shapes frame0 |
| **luma/chroma (spectral)** | chroma-first seg repairs don't perturb pose | **CONDITIONAL / SOFT** | triangularity holds ONLY in high-freq annulus band; low-freq chroma recolors pass into pose (review-D) |

**⇒ The composite's pose safety rides the FRAME-split (exact), never the CHANNEL-split (conditional).** This
is my sharpest correction to the routing story: the SPEC's "chroma-routed seams live where PoseNet barely
looks" is a TRUE but WEAK statement (it bounds only the high-freq band); the load-bearing safety is that
frame0 is SegNet-free and frame1 (the painted seg surface) is pose-free — a HARD firewall that no chroma
argument is needed for.

### Q4.2 Chroma-first is a FREE wall-clock lever, not a correctness lever

Because the merge→diff→correct fixed point is a projection loop (Q3.3), the **ORDER of projection does not
change the fixed point** — it changes only the convergence SPEED. Chroma-first ordering (edit the channels
SegNet sees but PoseNet low-passes FIRST) makes the early iterations cheap seg-repairs that don't perturb
the pose-tube projection → faster convergence. **This is a score-NEUTRAL wall-clock win** (operator binding
3: minimize wall-clock without touching score) → it is FREE and MANDATORY, but it is NOT a correctness
guarantee. Provenance: DERIVED (projection-order-invariance of the fixed point) + the #276 chroma-DOF
measured basis for the ordering choice. increment-1 is SAFE regardless of the routing because Lane
(luma-separable, 41% of Road's flips) is a separate ANALYTIC carrier never paint-repaired (review-D) — the
routing only ever touches the Road/Undriv chroma-separable paint.

### Q4.3 The real guard is MEASURE-pose-on-composites (never assert)

DIFF-frame1-only + reserve-frame0/luma-for-pose HOLDS under the composite because of the frame-split
structural firewall (Q4.1). But the composite paint DOES change frame1's luma in the annulus (seg repairs),
and low-freq luma structure feeds the temporal-screw / warp coherence. So the guard is: **at the Dykstra
fixed point, MEASURE d_pose on the composite pair** (the pose-tube projection SET-2 keeps luma warp-coherent,
but the MEASUREMENT is the arbiter, not the projection). The temporal-screw force (P0 FORCE 1, already BUILT)
is the NECESSARY companion — it keeps the frame0→frame1 luma warp coherent so the pose-tube is non-empty.
Provenance: the frame-split is MEASURED/structural (P-2); the "MEASURE not assert" discipline is the SPEC §3
+ risk-5 cure, CONFIRMED here as the real guard.

---

## THE POSE-CONDITIONING GATE CONTROL LAW (the v8-DECOMPOSED twist)

**OPERATOR BINDING (2026-07-09, verbatim):** *"pose must not be fired for joint descent until optimal — it
needs d_seg to be sufficiently conditioned first."* ⇒ pose-finish ENTRY = a d_seg-CONDITIONING EVENT, NEVER
an epoch.

### The v8 novelty: PER-CLASS conditioning on a decoupled trunk

On v7.5's SHARED trunk, "d_seg sufficiently conditioned" could only be read AGGREGATE — one moving class
drags the whole readout. **On the v8 DECOMPOSED trunk the conditioning readout is PER-CLASS** (the
`perclass_verdict.per_class_dseg_fields` surface gives `d_seg_by_class[c]`), AND because v8 DECOUPLES the
classes (∂φ_c/∂θ_{c'}=0), each class's conditioning trajectory is INDEPENDENT. **⇒ the gate can fire when the
DOMINANT edges (Road ~40% + Undriv ~8% of the d_seg mass, DELTA §J) are conditioned even while a tiny class
(Lane 0.59% area) still jitters** — which v7.5's shared trunk structurally could not do. This is a genuine
v8-specific control gain, not a re-skin.

### The gate {quantity, threshold-with-provenance, hysteresis, fallback}

**Quantity (BUILT surfaces):** BOTH of —
1. **d_seg-BASIN (seg-convergence):** per-class within-stage remaining-descent < 5% on {Road, Undriv}
   (`perclass_verdict.d_seg_by_class`, NCDE remaining_descent_frac=0.05 = MEASURED `ncde_trajectory`). VALUE:
   `{Road, Undriv}` (the dominant edges), NOT aggregate, NOT all-5. Provenance: DERIVED (v8 decoupling makes
   per-class independent; dominant-mass gating).
2. **pose-conditioning σ_min (pose-well-posedness):** `jacobian_basin.would_have_fired` —
   `median_sigma_min ≥ f_basin·σ_min^plateau AND basin_frac ≥ quorum_q`, from the SVD of
   `J_ξ = ∂(PoseNet∘R)/∂ξ` (registered eq `pose_jacobian_basin_conditioning_20260709`).

**Threshold-with-provenance:** `f_basin = 0.9` (NOT the built default 1.0). **DERIVED correction:** a LIVE
entry gate with `f_basin=1.0` requires median σ_min to EQUAL its own provisional running-max, which never
fires (a new max resets the target); `0.9` = "within 10% of best-observed conditioning" fires when σ_min
has essentially plateaued. The exact 0.9 is **ASSUMED_AWAITING_VERIFICATION** (owed a sensor-trust A/B; the
dashboard currently reads would-fire=no under 1.0). `quorum_q = 0.8` (BUILT default, MEASURED-tunable).

**Hysteresis:** predicate TRUE for **H = 3 consecutive basin-cadence verdicts** AND `median_sigma_min`
non-decreasing across the window (a transient σ_min spike that reverts fails the non-decreasing test). VALUE:
H=3, provenance DEFAULTED-WITH-RECESS (same persistence philosophy as the classifier `min_sustained_windows`;
recess = tune on the first real conditioning trajectory).

**NECESSARY-not-SUFFICIENT (P-6):** σ_min↑ says the pose basin is CONDITIONED, not that the seg render has
STOPPED. Both required (AND, not OR): the operator's "d_seg sufficiently conditioned first" =
**(per-class seg-BASIN on {Road,Undriv}) AND (σ_min pose-conditioning gate)**.

**Fallback (never-reached):** pose-finish does NOT fire → SHIP the R1-banked dxi (d_pose 0.001610 →
contribution 0.127, ξ_eff 7.2 KB, n600 AUTHORITY, DELTA §I P-1). Because R1 is ALREADY banked, a never-fired
in-basin finish costs ZERO → the in-basin finish is OPTIONAL, never a launch dependency. **This resolves
P-5's HONEST FLAG** (in-basin efficacy UNVALIDATED): the run is never bet on it. **STORE-NOTHING MANDATE
(operator binding 5):** the fresh v8 arm runs the `generated` store-nothing pose path (~1 KB); any keyframe
lineage MUST charge the counted-keyframe rate (P-7 lineage tag) — the banked R1 dxi is the store-nothing
carrier attached to the seg-converged EMA ckpt.

**Regression guard (if pose-finish DOES fire):** if d_pose does not beat the banked 0.001610 in the finish
window → ROLL BACK to pre-finish EMA + ship banked R1. A failed finish must never corrupt the seg-converged
ckpt nor ship a worse dxi.

---

## THE INCREMENT-1 CONTROL SEQUENCING (the deliverable — config-shaped)

The staged training gives a NATURAL two-part increment where the d_seg BET is measured before the paint
machinery is committed. Every value on the provenance ladder; TBD forbidden.

```
INCREMENT-1a  — the DECOUPLING BET (NO paint, NO P-C; the cheapest falsifiable v8 row)
  build   : Road+Undriv edge-centric bulk-field (the ONE new module road_undriv_bulk_field.py)
            + REUSE {hood_static, lane_sdf/analytic_lane_render_band, Movable v7.5 homotopy}
  Stage A : each carrier's field vs EXACT SDF targets (signed_distance_fields, argmax(sdf)==labels);
            ∂φ_c/∂θ_{c'}=0  (decoupled — NO cross-class gradient path)   [closes T-grad]
  b_c     : FLIP-WEIGHTED target masses = perclass_verdict.flip_share_by_class  (NOT hard_cell_masses;
            N-1 area-match HURTS)  → laguerre_logit_offset {menon warm-start, damped_newton_ot exact},
            CLOSED-FORM, OUT of the scorer-gradient loop  [review-E guard SATISFIED BY CONSTRUCTION]
            [OWED BUILD: the flip_share→target_masses wiring — open Q1]
  measure : composite tropical argmax → d_seg-through-R (n600, frozen CPU SegNet) + byte-close each
            carrier (bit-exact-at-decode, resumable, manifest-per-carrier BEFORE composition)
  scaffold FIX (standing catch): road_undriv_bulk_field.bulk_boundary_byte_cost mode="horizon_poly_xi"
            (measure the 426px Road↔Undriv shared edge, NOT the 2228px full Road perimeter) +
            multi-component-Road-aware (Road multi-blob 37.2% of frames)
  VERDICT : this row FALSIFIES-OR-CONFIRMS the v8 d_seg decoupling thesis through R.  If d_seg-through-R
            does NOT beat v7.5.2's per-class Road floor, the decoupling bet failed at Stage-A cost — no
            paint machinery was spent.  verdict_scope of a negative here: FORMULATION (this field
            parametrization / this b_c), NOT the v8 paradigm.

  ---- GATE: P-C (governed-heavy, memory-gated, NOT $0, NOT recess) ----
  P-C     : {flat_fill_dseg[c], procedural_fill_dseg[c], counted_seed_bytes[c]} per class, n600-through-R
  go/no-go: procedural_fill within δ_R (0.0196) of P-A oracle at ZERO counted seed  → paint is near-free
            else counted_seed_bytes[c] ENTERS the rate ledger (couples to S2/S4 residual decision)

INCREMENT-1b  — the PAINT stage (GATED behind P-C go)
  Stage B : merge→diff→correct, frame1-ONLY DIFF, per-class channel-split
  correct : LINEARIZED alternating projection (SET-1 argmax-cell ∩ SET-2 pose-tube) — NOT globally-convex
            "Dykstra"; controls = {max_iter cap (DEFAULTED-WITH-RECESS), per-iter d_seg-monotonicity
            reject, pre-paint-composite regression guard}
  order   : chroma-first (Road/Undriv) — FREE wall-clock lever (projection-order-invariant fixed point),
            NOT a correctness lever
  guard   : MEASURE d_pose on the composite at the fixed point (never assert); temporal-screw P0 FORCE 1
            keeps the pose-tube non-empty
  residual: unpaintable → #226 margin_conditional_residual.waterfill_select (net_value>0, marginal<1.27
            B/flip) — the counted sidecar (REAL coder, bit-exact)

TERMINAL  — pose-conditioning gate (EVENT, never epoch)
  entry   : (per-class d_seg-BASIN {Road,Undriv} remaining<5%) AND (jacobian_basin f_basin=0.9,
            quorum_q=0.8, H=3 hysteresis, σ_min non-decreasing)
  fire    : pose-finish (store-nothing generated path, ~1KB) → regression guard vs banked 0.001610
  fallbk  : never-reached → SHIP banked R1 dxi (0.127 / 7.2KB, n600 authority) — ZERO-cost fallback
```

---

## §6 ATTACK MY OWN CONCLUSION (operating manual §6; risk always, §7)

1. **The "Dykstra" fixed point is NOT convex-convergent.** SET-1 (argmax-cell) is non-convex (SegNet
   nonlinear). If the linearized alternating projection oscillates or diverges, the paint stage has no
   convergence guarantee. MITIGATION: max-iter cap + per-iter d_seg-monotonicity reject + the pre-paint
   regression guard (ship the unpainted composite + counted residual). RISK: if the local projection is
   consistently non-contractive, the paint stage buys nothing and the residual sidecar carries everything —
   which reduces v8 to "dominant generators + full residual coder" (the S2/S4 named enemy). LABEL: the
   projection-loop design is DERIVED; its CONVERGENCE on real composites is ASSUMED_AWAITING_VERIFICATION
   (rides increment-1b, behind P-C). This is the single biggest risk in my design.
2. **The flip-weighted b_c is un-built AND un-A/B'd.** N-1 proved area-match HURTS; flip-weighted is the
   OPEN reformulation, but "flip-weighted helps" is a HYPOTHESIS, not a measurement (DELTA §H: larger-n
   OWED, probe not resumable-chunked). MITIGATION: the b_c-out-of-loop guard makes a WRONG b_c
   non-catastrophic (it perturbs the tie calibration, it cannot steal via gradient); increment-1a MEASURES
   the flip-weighted b_c through R. LABEL: flip-weighted-is-the-open-arm = DERIVED (N-1); flip-weighted-HELPS
   = ASSUMED_AWAITING_VERIFICATION.
3. **P-C is a heavy governed run I am gating the whole paint stage on — and it OOM'd #205's path.** If P-C
   cannot run memory-safe even with the box free, the paint stage is blocked and increment-1 can only ship
   1a (Stage-A fields). MITIGATION: that is actually FINE — increment-1a is a complete, cheaper, falsifiable
   v8 row (the decoupling bet); the paint stage is a strict improvement gated behind evidence. If P-C never
   runs, v8 ships the dominant-only rate with the residual as a measured-owed sidecar (S2/S4's fallback). So
   my sequencing DE-RISKS the P-C dependency rather than being blocked by it.
4. **I am the CONTROL seat, not the CARRIER-COMPOSITION seat (S1/S2) nor the STRUCTURE-BLIND seat (S6).**
   My increment-1a/1b split PRESUPPOSES the SPEC_v8 carrier decomposition (Road+Undriv edge-centric + 4
   reused). If S6's blind derivation DIVERGES on field-count or sharing structure, the decomposition changes
   and my staging attaches to whatever the synthesis adopts — my control laws (staged theft firewall, b_c
   out-of-loop, projection-not-optimization paint, per-class pose gate) transfer to any decomposition. The
   "edge-centric" label is aspirational for full v8 (only the ONE Road/Undriv edge is edge-centric in
   increment-1; the rest are per-class) — I inherit that honest scope, not the 41-edge claim.
5. **Per-class gate on {Road,Undriv} could starve a lagging class into the ship.** If Lane/Movable are still
   jittering when the gate fires pose, the composite ships with a mid-conditioned tiny class. MITIGATION: the
   tiny classes are 0.59%/1.56% area (DELTA §SETTLED order) — their d_seg contribution is bounded small; and
   the pose gate governs POSE, not seg-stage-exit (the seg stages have their own per-class exit events). RISK:
   if a tiny class carries disproportionate FLIP mass (Lane = 19% of flips despite 0.59% area, memory L80),
   gating pose on Road/Undriv alone could fire while Lane flips are still moving — I should ADD Lane's
   flip-share to the gate readout as a WATCH (not a block). LABEL: the {Road,Undriv} dominant gate is
   DERIVED from area mass; the Lane-flip-share WATCH is a folded correction (open for P2).
6. **"Score-neutral chroma-first ordering" assumes the fixed point is truly order-invariant.** That is only
   exactly true for CONVEX Dykstra; with the non-convex linearized loop, ordering CAN change which local
   fixed point you reach. MITIGATION: the d_pose measurement at the fixed point + the regression guard catch
   a bad local basin regardless of order. LABEL: order-invariance is DERIVED-for-convex, INFERRED-for-the-
   linearized-loop — so "chroma-first is score-neutral" is INFERRED, verified by the fixed-point d_pose
   measurement, not asserted.

---

## OPEN QUESTIONS FOR P2 (the synthesis owner resolves)

1. **The flip-weighted b_c target-mass wiring is an OWED BUILD.** `laguerre_logit_offset` matches to
   `target_masses` but only computes AREA masses; the flip-weighted target (`perclass_verdict.
   flip_share_by_class`) exists as a sensor but is NOT wired to the b_c solver. Does increment-1a build the
   `flip_share → target_masses` path, or ship area-match with a measured caveat? (N-1 says area HURTS, so my
   position is BUILD it; the surface is small.)
2. **Does increment-1 stop at Stage-A (1a) or include the paint stage (1b)?** My control position: the
   smallest DECISIVE build is 1a (the decoupling bet, measurable without paint or P-C). The paint stage is a
   strict improvement gated behind P-C. Does the synthesis adopt this 1a/1b split, or does S1/S2's
   carrier-composition analysis demand the paint stage in the first increment?
3. **Does P-C's counted_seed_bytes enter the rate ledger?** The 0.061/0.140 numbers presuppose the paint is
   free; if procedural-fill needs counted class-typical statistics, that byte cost is currently UN-accounted
   and couples directly to S2/S4's residual-coder decision (the 0.079 named enemy). Who owns folding the
   counted-seed-floor into the rate accounting?

---

## PROVENANCE (per manual §5)

- **MEASURED (artifact / source-inspection):** the b_c surfaces closed-form & out-of-gradient-loop
  (`laguerre_logit_offset.py` — menon/damped_newton_ot/power_diagram_argmax/hard_cell_masses); #226 residual
  coder real (`margin_conditional_residual.py` waterfill_select net_value>0 marginal<1.27); the pose-gate
  sensor (`jacobian_basin.py` f_basin/quorum_q/would_have_fired + `render_grad_energy_per_class` "connects to
  the v8 per-class carriers"); per-class d_seg sensor (`perclass_verdict.per_class_dseg_fields` →
  d_seg_by_class + flip_share_by_class); ∂d_seg/∂ξ≡0 (P-2, modules.py:108 SegNet reads x[:,-1]); N-1
  area-match HURTS (0.00487 vs 0.00272); flat-paint fails 0.0064 (increment-1 §9); R1 dxi 0.001610/0.127/
  7.2KB (P-1); δ_R 0.0196 (DELTA SETTLED); scaffold bulk_boundary_byte_cost scope bug (707B full-Road vs
  426px Road↔Undriv).
- **DERIVED:** the two-theft-channel model (T-grad closed by Stage-A, T-comp by the three composite
  controls); the b_c-out-of-loop guard satisfied BY CONSTRUCTION (closed-form); flip-weighted target masses =
  flip_share_by_class (N-1 open arm); the projection-loop-has-no-theft-gradient argument; the SegNet
  non-convexity ⇒ linearized-not-convex-Dykstra caveat; the frame-split-structural vs channel-split-heuristic
  distinction; chroma-first = order-invariant-fixed-point = FREE wall-clock; f_basin=0.9 entry correction;
  the per-class {Road,Undriv} conditioning gate on the decoupled trunk (the v8 twist); the increment-1a/1b
  staging split.
- **INFERRED:** chroma-first order-neutrality UNDER the non-convex loop (verified by the fixed-point d_pose
  measurement, not asserted); the Lane-flip-share-WATCH addition to the gate.
- **ASSUMED_AWAITING_VERIFICATION:** the linearized paint loop CONVERGES on real composites (rides 1b,
  behind P-C) — my single biggest risk; flip-weighted-b_c-HELPS (rides 1a); exact f_basin=0.9 (sensor-trust
  A/B owed); H=3 hysteresis count; the counted-seed-floor magnitude (P-C UNRUN, governed-heavy).

**Pointer 0.19110 UNMOVED. This is MEANS — a control/reconciliation design, not a score. Only a byte-closed
n600 `upstream/evaluate.py` row moves the pointer. Remaining gap to sub-0.15 = 0.0411.**
