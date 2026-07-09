# P1 SEAT S1 — DEEP-MATH / LEVEL-SET ENERGY (independent position, crucible-2 v7.5.2)

**Author:** S1 (deep-math / level-set energy). **Date:** 2026-07-09. **Axis:** all numbers
`[macOS-MLX/CPU advisory · NON-PROMOTABLE]`. **Pointer 0.19110 UNMOVED — MEANS.** Independent: no
cross-read of other seats. Cites `docs/operating_manual_craft_handoff.md` (§4 re-derive, §5 label,
§6 attack-own, §8 mistakes). Works from DELTA_GROUNDING + the deep-math substrate (#284 EUREKA,
#318 GR-action, #302 curriculum, #360 four-forces, FEED-crux-7dim/roadfloor/roadfloorfix).

**Method (the area-Lagrange method, §4):** write the full variational derivative δS_τ/δφ the
multiphase level-set energy DEMANDS, then check each piece against a force present in the config.
A term the energy demands but the code lacks is a MISSING FORCE (this is how the Chan-Vese
area-Lagrange was found — FEED-roadfloorfix). Every lever L-1..L-8 is then classified by WHICH
piece of the energy it is, which decides SAME-TERM vs ANTAGONIST vs ORTHOGONAL (Q1).

---

## 0. THE ENERGY (the object every claim is derived from)

The witness φ (5-class softmax-of-SDF INR) is the **viscosity solution of the natural-gradient
flow of one τ-smoothed action** (DERIVED, #318 FEED-ia + #284 EUREKA, both registered laws):

```
S_τ[φ, ξ]  =  100·D_seg^τ[φ]  +  √(10·D_pose[ξ])  +  25·B[φ]/N          (the contest action)
G(θ) = E_x[ J_xᵀ F_x J_x ],  F_x = diag(p) − ppᵀ   (Fisher pullback of the FROZEN scorer)
flow:  G·θ̇ = −∇_θ S_τ        (natural-gradient)  ;   ∂φ_c/∂s = −δS_τ/δφ_c   (multiphase level-set)
```

The REPRESENTATION half (what makes φ a level-set field, active where the scorer is flat) is the
Mumford-Shah / multiphase-Modica-Mortola / eikonal functional (DERIVED, Ch.4 τ=ε=ħ, registered):

```
E[φ] = ∫[ W(φ)/ε + (ε/2)|∇φ|² ]dx        # diffuse-interface;  length term = Γ-limit PERIMETER
     + Σ_{c<c'} σ_{cc'} · Perim_{cc'}[φ]  # multiphase SURFACE TENSION (Baldo/Sternberg, K=5 wells)
     + Σ_c (λ_c/2)·relu(A_c − A_c^GT)²    # CHAN-VESE area-Lagrange (birth-balance)
     + μ ∫ (|∇φ| − 1)²                    # EIKONAL (|∇φ|=1 ⇒ interface half-width EXACTLY τ/2)
     + ∮_Σ ‖t_wit − t_GT‖²               # TIE-LOCUS free-boundary Σ={φ_top1=φ_top2} placement
     + ∫_annulus ‖φ(f1) − Warp_ξ(φ(f0))‖² # TRANSPORT  ∂φ/∂t + V·∇φ = 0  (V = ego-screw ξ)
```

τ = ε = ħ is ONE scalar (Maslov/Modica-Mortola/mirror-descent Planck constant, PROVEN Ch.4). The
curriculum is **mirror-descent GNC continuation along τ** (coarse→fine = Morse-Smale persistence
order = the anneal). This is the whole apparatus. Everything below reads off it.

---

## 1. Q1 — LEVER COMPOSITION FROM THE ENERGY (the central question)

**Classify each lever by which piece of δS_τ/δφ it is. That decides compose-vs-isolate.**
The confound §9 fears is GRADIENT OVERLAP IN G: two levers that touch the SAME piece of δS_τ/δφ
confound attribution (you cannot tell which moved d_seg). Levers touching DISJOINT pieces are
(near-)orthogonal in G and compose WITHOUT confounding. This is the deep-math resolution of the
"one-per-increment" tension — it is not a blanket rule, it is an **orthogonality-in-G rule**.

| lever | which piece of the energy | class | compose rule |
|---|---|---|---|
| **L-7 amber** (#378 eps_floor/clip/normalize) | **the METRIC G itself** (not a term) — regularizes the √-caustic + trust-region + block-G balance | **PRECONDITION** | ON, launch-blocking for any joint arm (Q3) |
| **L-3 step-native** | the ACTIVATION / function class of φ (representation) | REPRESENTATION | compose freely (0 loss-share) |
| **L-1 #121 taper** | the BASIS conditioning (curvelet capacity → margin band) — reshapes G, 0-byte | REPRESENTATION | compose freely; **⊥ excludes L-6** |
| **L-5 #276 chroma** | an ORTHOGONAL DOF of φ (chroma = rgb−luma, LUMA-INVARIANT) | ORTHOGONAL DOF | compose freely (0 loss-share) |
| eikonal-raise 0.05 | the EIKONAL term μ∫(|∇φ|−1)² | ENERGY TERM (regularizer) | compose (interface-width, DERIVED) |
| area-Lagrange (incumbent) | the Σλ_c relu(A_c−A_GT)² term | ENERGY TERM | ON (un-floors Road) |
| **FORCE-1 L-4** (#360 temporal) | the TRANSPORT term ∂φ/∂t+V·∇φ=0 | ENERGY TERM (new) | one-per-increment (loss-share) |
| **FORCE-2 (#360 satisfice)** + **L-2 (#169 horizon)** | **THE SAME UNIWARD HINGE** relu(m_safe−m_wit) on the shared `_signed` field | **SAME TERM** | **UNIFY into one; do NOT compose as two** |
| **FORCE-3 L-tie** (#360) | the TIE-LOCUS placement ∮_Σ‖t−t_GT‖² | ENERGY TERM (new) | one-per-increment; AFTER L-1 |
| **L-6 #220 AA** | the RENDER OPERATOR R (coverage supersample = anti-Gibbs) | RENDER OP | **ANTAGONIST of L-1** (measured fail-closed) — separate arm |
| **L-8 #341 head GN** | a SOLVE of the terminal affine head (chart near-quadratic) | SOLVE-NOT-TRAIN | replaces terminal train leg |

**Three findings the energy forces (§5-labeled):**

1. **SAME-TERM (DERIVED): FORCE-2 and L-2 (#169) are ONE term, not two.** Both are the one-sided
   satisficing hinge `relu(m_safe − m_wit)` on the SHARED #141 `_signed` margin field. They differ
   only in (a) the threshold m_safe (Force-2: R-floor 3·δ_R=0.06; #169: margin≥0.3/0.5) and (b)
   the spatial weight (Force-2: annulus #333; #169: horizon rows 96-288). The ENERGY says there is
   ONE UNIWARD hinge with spatial weight = **annulus ∩ horizon** and m_safe = the **R-noise floor**
   (the only principled threshold — above it a pixel is R-robustly safe; #169's 0.3/0.5 is an
   oracle-ceiling proxy, not a derived floor). **Composing both as separate levers double-counts
   the hinge and inflates its loss-share toward the term_domination alarm.** Unify.

2. **ORTHOGONAL-IN-G COMPOSES FREELY, AND THE REPRESENTATION SET COSTS ZERO LOSS-SHARE.** L-3
   (activation), L-1 (basis), L-5 (chroma DOF) are NOT loss terms — they reshape the function class
   / metric / DOF, consuming **zero of the §9 ≤15%/≤40% loss-share budget**. The §9 caps bind only
   on ADDED LOSS TERMS (transport, the unified hinge, tie-locus). So the composition SPLITS cleanly:
   **representation/metric levers compose in the FIRST launch (orthogonal in G, no loss-share);
   loss terms activate one-per-increment (they share the seg-gradient budget).** This honors §9
   exactly while composing the orthogonal set — the §9 rule is a loss-share rule, and representation
   levers are not in its domain.

3. **SYNERGY ORDER = the energy's natural sequence: representation → placement → stabilize.** When
   the loss terms DO activate, the order is forced by δS_τ/δφ: **L-1 taper (basis capacity on the
   band) BEFORE Force-3 (place the tie) BEFORE Force-1 (temporally stabilize the placed tie).**
   Basis-before-capacity-before-placement-before-transport. (crux-7dim: direction/scale is PRIOR to
   place; FEED-PA interaction matrix: #2 frees budget → #3 spends it → #1 stabilizes.) **ANTAGONIST:
   L-6 (#220 AA) vs L-1 (taper) is MEASURED fail-closed (both touch the sampling/basis) — mutually
   exclusive arms, never in the same launch.**

**Net Q1 recommendation:** FIRST v7.5.2 launch composes {amber + step-native + taper + chroma +
eikonal-0.05 + incumbent area-Lagrange} (metric+representation+regularizer, zero new loss-share
confound). Then activate ONE new loss term per increment in synergy order: **(i) the UNIFIED margin
hinge, (ii) tie-locus w_e=pa_flipmass, (iii) temporal-screw** — with A/B isolation, because these
three DO overlap the seg-gradient budget. L-6 (#220) is a SEPARATE arm vs the taper arm.

---

## 2. Q3 — eps_floor IS A DERIVED PRECONDITION FOR JOINT DESCENT (not a lever to A/B)

**DERIVED, and it is the sharpest thing this seat has to say.** The pose term is √(10·D_pose). Its
gradient in the pose state ξ is

```
∂/∂ξ √(10·D_pose) = 5/√(10·D_pose) · ∂D_pose/∂ξ      →  ∞   as D_pose → 0.
```

This is a **caustic in the metric G** (FEED-ia: "√ = concave field-dependent coupling, marginal→∞
as d_pose→0"). Per-pair batch=1 EXPOSES it: one easy pair with D_pose≈1e-8 gives a gradient of
~5/√(10·1e-8) ≈ 5e4 (exactly the L-7 diagnosis #2). The catastrophe is not random — it fires
**precisely when d_seg is converging**: converging d_seg → richer boundary normals → σ_min of
J_ξ=∂(PoseNet∘R)/∂ξ rises → pose becomes solvable → D_pose plummets toward 0 → the √-caustic blows
up (this is the P-6 σ_min↔conditioning derivation read the other way). So the joint 7-dim descent
DIVERGES at the moment it starts to succeed. That is the run-1/v6 collapse.

`eps_floor(C) = (5/C)²` is exactly the **smoothing of the √-caustic**: flooring D_pose caps the
pose-gradient magnitude at C. The other three amber pieces are metric-conditioning too — grad-clip
= trust-region on the natural-gradient step in G; per-param-normalize = diagonal preconditioning of
G; w_seg stage-boundary guard = balancing the block-diagonal G (w_seg=100 ⇒ a 100× seg block that
dominates the flow). **All four fix the CONDITIONING of G, not the energy.** A flow in a singular G
diverges; L-7 makes G well-posed.

**Verdict (Q3): amber is default-ON and launch-BLOCKING for any arm that composes ≥2 loss terms OR
turns on pose (the regimes that excite the caustic / sharpen the boundary → richer normals → the
divergence regime).** It is NOT a lever to A/B for on-vs-off — the DERIVATION says a joint flow is
ill-posed without it. The legitimate A/B is over the **VALUE of C** (the clip level / floor
magnitude) as a fail-safe. P-5's honest flag stands (efficacy of the specific constants UN-A/B'd),
but "regularize the singular metric" is not optional; "which C" is the open question. It is a
PRECONDITION, not a parallel arm — for a single-term arm that doesn't excite the caustic (pure
seg, pose off, one term) it can be a parallel arm, but the moment you compose the joint stack it is
launch-blocking. (§6 self-attack: could the caustic be avoided by never letting D_pose→0 during
seg training? Yes — that is the pose-BLIND incumbent, w_pose=0. But the whole v7.5.2 thesis is the
terminal JOINT pose-finish (D.9), which by construction drives D_pose→0 while d_seg holds — so the
caustic is on the critical path and amber is required there specifically.)

---

## 3. Q4 — A STAGE-EXIT EVENT IS AN ENERGY-VS-MEASURE DIVERGENCE (the decoupling, made precise)

The training loss is **S_τ** (the τ-smoothed SURROGATE action). The verdict is **D_seg^{τ→0}** (the
hard-argmax MEASURE). In the Γ-limit they agree (Modica-Mortola: the soft continuation lands on the
hard perimeter-minimizer as τ→0, PROVEN Ch.4). At FINITE τ they are DIFFERENT functionals. The R-4
false-green is the sign divergence:

```
DECOUPLING EVENT  ⟺   dS_τ/ds < 0   ∧   dD_seg^{τ=0}/ds ≥ 0
```

**Energy meaning (DERIVED):** the current stage's flow is descending the τ-smoothed energy in a
direction that INCREASES the hard-argmax energy. This happens **exactly when the residual d_seg
lives in structures FINER than the diffuse-interface width ε=τ.** The thin 6px lane, below the
τ-scale, is being ERASED by the perimeter/mean-curvature term (MCF minority-erasure inevitability,
registered law #7) while the τ-smoothed CE keeps descending on the bulk. This IS the FEED-04b
nucleation signature ("un-nucleated lane erased; d_seg creeps 0.00475→0.00657 while ep_loss falls
148→134") and the R-4 signature ("Lane 0.349→0.381 while train loss descended") — ONE mechanism.

**So the stage-exit EVENT is the SCALE-SATURATION detector:** the current τ (interface width) can no
longer resolve the residual — the residual has moved to a finer scale. The cure is NOT "more of the
same gradient" (the operator's "more CE isn't going to help" — DERIVED-correct: more CE optimizes
the WRONG scale). The cure is a **SCALE TRANSITION (lower τ, the coarse→fine persistence anneal) OR
a TOPOLOGY EVENT (nucleate the sub-critical class the flow is erasing).** The decoupling PRECEDES a
train-loss plateau (train loss is still descending when the measure has already stalled), so it is a
STRICTLY EARLIER and more informative stage-exit trigger than a plateau.

**Distinguishing decoupling from EMA-lag (R-2) — DERIVED discriminator:**
- **EMA-lag** is a MEASUREMENT DELAY: the verdict reads the EMA shadow, which lags live weights by
  the EMA window N=1/(1−ρ) steps (per-step ρ=0.997 ⇒ N≈333 steps ≈ the measured 78× early lag).
  Live D_seg is already descending; the shadow hasn't caught up. It RESOLVES within one EMA window
  and is CLASS-UNIFORM (all classes' verdicts lag equally).
- **Genuine decoupling** PERSISTS beyond one EMA window AND shows the SCALE signature: the RISING
  flip mass concentrates in the **sub-τ (thin / high-curvature) classes** (Lane, per MCF-erasure),
  not uniformly. This is measurable from the per-class annulus flip-frac already logged.

```
DECOUPLING CONFIRMED  ⟺  (dS_τ/ds<0 ∧ dD_seg/ds≥0) persists > N_ema steps
                          ∧  rising-flip-mass concentrated in high-curvature/thin classes (Lane)
EMA-LAG (benign)      ⟺  same sign-divergence but < N_ema steps  ∧  class-uniform
```

**Q4 recommendation:** make the DECOUPLING event a first-class stage-exit trigger (fire CE→tau on
the decoupling, NOT on epoch 300 and NOT merely on train-plateau). Gate the min-stage by the
critical-slowing relaxation time τ_relax (S-4, DERIVED-AT-CONFIG — near a stage transition the flow
critically slows, relaxation ∝ 1/gap²; min-stage ≥ τ_relax so a post-transition transient flat isn't
misread). Keep the epoch caps (Muon 726, Polyak 2546) as FAIL-SAFES only. The event must carry the
EMA-window + class-concentration discriminator so it never fires on the EMA-lag artifact.

---

## 4. MISSING-TERM ANALYSIS (the area-Lagrange method: energy-demanded vs code-present)

Walk δS_τ/δφ_c = 0 piece by piece; flag every term the energy demands that the config lacks.

| # | energy term (δS_τ/δφ piece) | in config? | verdict |
|---|---|---|---|
| 1 | FIDELITY −∂D_seg^τ/δφ (CE/tau) | ✓ | present |
| 2 | PERIMETER/curvature κ (length) | ✓ (0.001, kept small) | present |
| 3 | EIKONAL |∇φ|=1 | ✓ (raise 0.01→0.05) | present, under-set |
| 4 | AREA-LAGRANGE Σλ_c relu(A_c−A_GT)² | ✓ NOW (the found term) | present |
| **5** | **SURFACE TENSION σ_{cc'}·Perim_{cc'} (multiphase Modica-Mortola, per-class-PAIR)** | **✗ MISSING** | **the strong catch** |
| 6 | TRANSPORT ∂φ/∂t+V·∇φ=0 | ✓ NOW (Force-1) | present (pending compose) |
| 7 | TIE-LOCUS ∮_Σ‖t−t_GT‖² | ✓ (subpix+Force-3 w_e) | present (pending compose) |
| 8 | RATE Λ (in-training MDL, reverse-water-filling) | ✗ (#242 never-fired) | LOW-EV for v7.5 (rate DEAD-at-floor) |

**THE MISSING TERM (S1's headline catch, §5-DERIVED, FORMALIZATION_PENDING): per-class-pair
anisotropic SURFACE TENSION σ_{cc'}.** The code's length term is a SCALAR isotropic perimeter. The
multiphase-Modica-Mortola Γ-limit (Baldo/Sternberg, K=5 wells) demands a PER-PAIR σ_{cc'} with
Herring triple-junction angle conditions. This is NOT the same as Force-3:
- Force-3 (tie-locus) is a **DATA/fidelity** term (pull the boundary toward the GT tie position),
  reweighted by a DISCRETE per-straddle flip-density w_e.
- σ_{cc'} is a **REGULARIZER** term (the intrinsic stiffness/curvature-resistance of the c–c'
  interface). Force-3's w_e is the DISCRETE PROXY; σ_{cc'} is its CONTINUOUS energy form.

**Why it is high-EV:** the isotropic length term is exactly what ERASES the thin lane (MCF
minority-erasure, registered law #7). A per-pair σ that is **anisotropic / lowered on the thin-lane
and Road↔Lane interfaces** makes those specific boundaries STIFFER against curvature-shrinkage —
the DERIVED anti-erosion cure. Road↔Lane is 41% of Road's flips (FEED-PA); the isotropic term
under-weights exactly the boundary that dominates. This term was FLAGGED in Ch.4 ("class-pair σ_ij
surface tension (Baldo — attacks erasure crux)") but the phase-2 build shipped Forces 1-3 and NEVER
built σ_{cc'}. It is the unconsumed missing force. **Recommendation: build σ_{cc'} as a DSL Lever
factory (anisotropic per-pair perimeter, low/anti-erosion on {Road↔Lane, thin-lane}); it is the
CONTINUOUS twin of the tie-locus w_e and the DERIVED complement to the area-Lagrange (area fixes
class MASS; σ fixes interface STIFFNESS — together they pin both the volume and the boundary of the
sub-critical lane).** DERIVED; FORMALIZATION_PENDING (register when its n600 A/B lands).

Term 8 (in-training rate MDL) is DERIVED-present in the energy but LOW-EV for v7.5 (pose BANKED,
rate DEAD-at-floor-for-this-vehicle); noted, not recommended for the first arm.

---

## 5. THE crux-7dim JOINT OBJECT (why the costate is the optimizer, not more sweeps)

The residual d_seg is ONE object — a codim-1 boundary-band flip — living in the PRODUCT SPACE of 7
coordinates (scale, res, time, direction, chroma, luma, place; FEED-crux-7dim). Each lever is an
actuator on one coordinate. The energy says the flip is a JOINT stationary point: δS_τ/δ(each
coordinate)=0. The costate λ (#247) IS ∂(terminal d_seg)/∂(each coordinate) — the Pontryagin
sensitivity — so **the costate controller is the JOINT optimizer over the 7 dims** (this is the
"capacity-sweep reflex" antidote, §8-mistake-2: do NOT design a big per-dim sweep; the costate
already joint-optimizes over the built actuators). Curvelet self-similarity across scale makes the
{direction×place} placement SCALE-RECURSIVE (the "recursive-fractal": same stationarity at each
octave). **The collapse-fix (Q3) is precisely what makes this joint flow CONVERGE** — the AMBER
top-AIML result (d_seg 0.00337, boundary_band_flip 0.079 = half the polynomial wall) was joint
through-R descent; it was shelved for the now-DIAGNOSED collapse bug (X-2), not a wall. So the
energy's verdict on the whole stack: **condition G (amber) → put capacity on the fragile coordinates
in synergy order → let the costate joint-optimize; the sub-0.15 boundary-placement path is
completion(recall) + tie-locus(precision) + σ_{cc'}(anti-erosion) + margin-satisfice(budget), the
four terms that all aim at FEED-PA's "boundary placement is 100% of the floor."**

---

## 5b. Q5 — POSE ENGAGEMENT IS A CONDITIONING-GATED EVENT (operator binding, DERIVED as a LAW)

**Operator binding (2026-07-09):** *"pose must not be fired for joint descent until optimal — it
needs d_seg to be sufficiently conditioned first."* Derive from the energy what "sufficiently
conditioned" MEANS and the bar as a LAW, not a hand value.

**The mechanism (DERIVED — why early pose destabilizes d_seg even though P-2 says ∂d_seg/∂ξ≡0).**
The direct coupling is exactly zero (SegNet reads only f1; ξ shapes only the seg-free f0, P-2). The
destabilization is INDIRECT, through the SHARED TRUNK θ. The pose natural-gradient step is
`θ̇_pose = −G_pose⁻¹ ∇_θ√(10·D_pose)`, and its magnitude is the product of the TWO singular factors
of the pose subproblem:

```
‖pose NG step‖  ~   (5 / √(10·D_pose))   ·   1 / (σ_min²(J_ξ)·λ_min(F))
                     └── √-CAUSTIC ──┘        └── JACOBIAN DEGENERACY ──┘
     J_ξ = ∂(PoseNet∘R)/∂ξ ∈ ℝ^{6×6}  (P-6) ;  σ_min = smallest singular value ;  G_pose = J_ξᵀ F J_ξ
```

The step blows up when EITHER factor is singular. **Q3's amber/eps_floor tames the NUMERATOR** (caps
5/√(10·D_pose) at C). **The conditioning gate tames the DENOMINATOR** (delays engagement until
σ_min(J_ξ) is large). They are the TWO HALVES of taming the SAME singular pose step — amber is
always-on for the numerator, the conditioning gate is a delayed-engagement event for the denominator.
When σ_min(J_ξ) is small (unconverged d_seg), G_pose is ill-conditioned and its null directions are
SHARED-θ directions that DO overlap the d_seg boundary-normal representation → a huge pose step
corrupts the separatrix. This is the R1/birth-arm lesson ("early pose became THE blocker"; the flat
1.2–1.8 pose floor from an unconverged basin vs 0.0011 from a converged one, P-6). **σ_min(J_ξ) is
low precisely when d_seg is unconverged** (few/weak boundary normals to constrain the screw) and
RISES as d_seg converges (richer normals, P-6) — so σ_min(J_ξ) is the DIRECT measure of "d_seg
sufficiently conditioned," already logged by the jacobian_basin telemetry (`median_sigma_min`,
`basin_frac`).

**The LAW (DERIVED threshold, not a hand value).** Pose-finish joins the joint descent iff BOTH:

```
(A) IDENTIFIABILITY:  σ_min(J_ξ)  ≥  σ*  =  √( C / (δ_seg · λ_min(F)) )
       C       = the amber pose-gradient cap  (DERIVED, Q3)
       δ_seg   = the seg trust-region  (= the amber grad-clip radius, already set)
       λ_min(F)= the frozen-scorer Fisher floor  (MEASURABLE, F=diag(p)−ppᵀ on the annulus)
   DERIVATION: with the numerator capped at C, the worst-direction pose NG-step is ≤ C/(σ_min²·λ_min(F));
   require it ≤ the seg trust-region δ_seg so the shared-θ step cannot exceed the separatrix's own
   restoring radius ⟹ σ_min² ≥ C/(δ_seg·λ_min(F)) ⟹ σ* as above.  (σ* falls out of amber's OWN
   constants — it is the σ_min at which the DENOMINATOR singularity is no worse than the NUMERATOR
   cap already permits. No hand value.)

(B) SEPARATRIX STABILITY:  the seg flow is at a STABLE critical point, i.e.
       dD_seg^{τ=0}/ds ≈ 0 FROM BELOW  (converged, NOT the Q4 decoupling/erasing regime)
       ∧  the birth/persistence COMPLETION event has fired (islands formed, not still nucleating)
       ∧  basin_frac ≥ b*   (the batch-mean pose gradient is not dominated by degenerate pairs;
                             b* = the fraction at which median_sigma_min is a representative statistic,
                             DERIVED-AT-CONFIG from the jacobian_basin distribution, sister of σ*)
```

**Why a LAW and not muon-cap-726.** D.9's incumbent backstop fires at `_muon_gate.fired` OR epoch
726 — that is a SCHEDULE proxy for conditioning, not the conditioning itself. The energy says the
correct trigger is the CONDITIONING QUANTITY σ_min(J_ξ) crossing σ*, gated by separatrix stability
(B). Muon-cap-726 stays a FAIL-SAFE (never engage pose before it), but the ENGAGEMENT event is the
σ_min law. This is the exact dual of Q4: Q4 makes stage-EXIT an energy-vs-measure event; Q5 makes
pose-ENTRY a conditioning event. Both replace hand epochs with measured/derived triggers.

**Honest bounds (§5/§6).** (1) σ* is DERIVED-FORM; its numeric value needs λ_min(F) MEASURED on the
current ckpt's annulus (owed probe — cheap, reuses the #333/#141 machinery). (2) The identifiability
argument treats the shared-θ overlap as worst-case (full overlap of J_ξ null-space with the d_seg
boundary-normal subspace); the TRUE overlap is measurable (project ∇_θ√(10·D_pose) onto the d_seg
Hessian's low-eigenvalue subspace) and is likely < 1, so σ* is a CONSERVATIVE (safe) bar — it will
engage pose no EARLIER than optimal, which is exactly the operator's directive ("not until
sufficiently conditioned"). Erring conservative here is correct: a too-late pose engagement costs
terminal epochs; a too-early one destabilizes the separatrix (the R1 blocker). (3) σ_min(J_ξ) is
OBSERVABILITY; the pose CONTENT DOF still comes from θ (P-6 "necessary-not-sufficient") — the gate
says WHEN it is SAFE to descend pose, not that descending will succeed (P-5's efficacy flag stands).

---

## 6. CONFIG-SHAPED RECOMMENDATION (every knob labeled by provenance)

Provenance ladder: **DERIVED** (from the energy) · **MEASURED** (n600/probe anchor) · **DERIVED-LIVE**
(computed at config from loaded GT) · **ASSUMED** (owed to A/B). Loss weights at STAGE BOUNDARIES
ONLY, never per-step (L4/L5 confound). This is a CONTROL-LAW shape, not a launchable argv (P2 owns
the exact DSL WitnessProgram); the flags below are the ones the energy demands ON/OFF/UNIFIED.

### 6.1 PRECONDITION — metric well-posedness (ON, launch-blocking for the joint stack)
```
--stability-preset amber                          # DERIVED (Q3): regularizes the √-caustic + trust-region
    eps_floor(C)=(5/C)²   [DERIVED from ∂√(10·Dpose)/∂ξ = 5/√(10·Dpose)→∞ ; A/B the VALUE C]
    grad-clip = trust-region on the NG step in G  ;  per-param-normalize = diag-G precondition
    w_seg stage-boundary guard = block-G balance (w_seg=100 ⇒ 100× seg block)
# LAUNCH-BLOCKING for any arm with ≥2 loss terms OR pose-on. NEVER off for the joint arm.
```

### 6.2 REPRESENTATION + METRIC (compose FREELY in launch-1 — zero loss-share, orthogonal in G)
```
--activation step_native            # L-3  DERIVED (piecewise-const argmax chart, no Gibbs) · MEASURED −4.5% n600
--dseg-aware-taper on               # L-1  DERIVED (basis capacity → margin band; 0-byte rule-118) · INSTANCE-scope
                                    #      re-validate the converged flip (+18% was under-converged) · ⊥ EXCLUDES --supersample
--seg-chroma-boundary on            # L-5  DERIVED (chroma=rgb−luma = ORTHOGONAL luma-invariant DOF) · ADD-BACK ΔS ASSUMED(owed)
```

### 6.3 ENERGY TERMS (δS_τ/δφ pieces)
```
# regularizers (compose in launch-1):
--length-weight 0.001               # DERIVED (MCF-erosion driver → keep SMALL)
--eikonal-weight 0.05               # DERIVED (Ch.4: interface half-width = τ/2; raise 0.01→0.05 keeps thin lane sharp)
area-Lagrange (incumbent)           # DERIVED-LIVE λ_lane 683.8 / λ_movable 322.6 ; equilibrium 1.25×GT un-floors Road

# NEW MISSING TERM — build + queue (S1 catch, §4):
σ_{cc'} anisotropic surface tension # DERIVED (multiphase Modica-Mortola; Baldo) · FORMALIZATION_PENDING
    low/anti-erosion on {Road↔Lane, thin-lane} = the DERIVED anti-MCF-erasure cure
    (continuous twin of Force-3 w_e; complements area-Lagrange: area=MASS, σ=STIFFNESS)
    → BUILD as a DSL Lever factory; highest-EV unconsumed force; its own increment A/B

# added loss terms — ONE PER INCREMENT (these DO share the seg-gradient budget, §9 caps bind):
#  increment (i) — UNIFIED margin hinge (Q1 SAME-TERM: Force-2 ∪ #169, do NOT compose as two):
--seg-margin-satisfice-weight w_s   # w_s=0.2 ASSUMED ; start≥l7 (preserves τ-anneal)
    m_safe = 3·δ_R = 0.06           # MEASURED δ_R=0.0196 (reports/delta_R_noise_floor.json) — the ONLY derived threshold
    spatial-weight = annulus ∩ horizon   # unify #333 annulus with #169 horizon rows 96-288
    (#169's margin≥0.3/0.5 is an oracle-ceiling PROXY, subsumed by the R-floor m_safe)
#  increment (ii) — tie-locus (AFTER L-1: basis before placement):
--seg-subpix-boundary-weight w_tie  # w_tie=0.3 ASSUMED ; start≥l7
--seg-subpix-edge-weight-source pa_flipmass   # w_e from FEED-PA destination matrix (Road↔Lane heaviest)
#  increment (iii) — temporal transport (last: stabilize the placed tie):
--seg-temporal-screw-weight w_t     # w_t cold 0.1 → ramp to grad-share≈0.44 at STAGE BOUNDARIES · DERIVED (transport)
--seg-temporal-screw-xi-source ground_gt   # stop-grad ξ ⇒ PURE seg regularizer, ZERO pose coupling (L68)
--seg-temporal-screw-classes 0,1,2  # GROUND only (homography wrong for Movable/MyCar)
```

### 6.4 SOLVE-NOT-TRAIN (terminal head)
```
--terminal-head-solve gn_cg full_P  # L-8  fire IFF LM ρ re-verify ∈[0.8,1.2] on THIS ckpt, all 600 pairs
                                    # MEASURED ρ 0.847/0.868 ; K=8 subset OVERFITS +5.1% (N-3) → full-P ONLY
                                    # replaces the terminal fine-tune leg (~3h GPU, NOT $0)
```

### 6.5 POSE (banked; ship the R1 dxi) — CONDITIONING-GATED (Q5, operator binding) — HONEST FLAG P-5
```
# ENGAGEMENT = the σ_min(J_ξ) conditioning LAW (Q5), NOT a hand epoch. Operator binding 2026-07-09.
--pose-finish-engage-on conditioning_gate      # DERIVED (Q5): fire pose ONLY when BOTH:
   (A) σ_min(J_ξ) ≥ σ* = √(C/(δ_seg·λ_min(F)))      # identifiability; C,δ_seg from amber; λ_min(F) MEASURED-owed
   (B) separatrix STABLE: dD_seg^{τ=0}/ds≈0 from below (NOT decoupling/erasing) ∧ birth-completion fired
                          ∧ basin_frac ≥ b*           # from jacobian_basin telemetry (median_sigma_min, basin_frac)
--pose-finish-muon-cap 726                     # FAIL-SAFE floor only (never engage BEFORE); NOT the trigger
   ground_gt → carrier-live at terminal joint pose-descent ; ships R1 dxi (d_pose 0.001610 → 0.127, ξ_eff 7.2KB, MEASURED n600)
# σ_min gate = the DENOMINATOR half of the singular pose step; amber eps_floor (§6.1) = the NUMERATOR half — BOTH required.
# HONEST (P-5): efficacy of v7.5 terminal finish converging to R1-class dxi is UNVALIDATED; the gate says WHEN it is
#   SAFE to descend pose (separatrix un-corrupted), not that it WILL succeed (content DOF from θ, P-6 necessary-not-sufficient).
```

### 6.6 SCHEDULE (event-driven; the decoupling event is first-class — Q4)
```
CE→tau  fires on the DECOUPLING event   # DERIVED (Q4): dS_τ/ds<0 ∧ dD_seg^{τ=0}/ds≥0, PERSIST > N_ema ∧ class-concentrated
    (NOT epoch 300, NOT train-plateau; decoupling PRECEDES plateau) — carries the EMA-lag discriminator
min-stage ≥ τ_relax                     # S-4 DERIVED-AT-CONFIG (critical-slowing relaxation, ∝1/gap²) — supersedes hand-250
--tau-anneal-shape geometric  --softmax-temp-end 1.0   # DERIVED (Fisher-Rao geodesic = constant-info-velocity adiabatic)
--stage-transition-rewarmup ... (Muon #270 warm-start-momentum + LR re-warmup + lr-final-frac 0.1)   # MEASURED −32% Muon
Muon cap 726, Polyak 2546 = FAIL-SAFES only    # DERIVED: caps are event-safe backstops, not the trigger
```

### 6.7 EXCLUDED / SEPARATE-ARM
```
--supersample (#220 AA)   # L-6 ANTAGONIST of L-1 taper (MEASURED fail-closed) → its OWN arm vs the taper arm, never same launch
in-training rate MDL (#242) # DERIVED-present but LOW-EV (rate DEAD-at-floor for v7.5) — not launch-1
```

**Launch-1 composition (energy-ordered):** amber → {step-native + taper + chroma} → CE+area-Lagrange
(form regions, birth) → **DECOUPLING** → tau+eikonal-0.05 (sharpen) → [increment loss terms
i→ii→iii, one per A/B] → Muon finish → head GN solve → pose-finish → byte-close. Representation set
composes at launch (0 loss-share); the three loss terms + σ_{cc'} drain the queue one-per-increment.

---

## 7. Self-attack (§6) — where this position is weakest
- **σ_{cc'} is DERIVED not MEASURED** — its ΔS is unproven; the anti-erosion CLAIM rests on the MCF-erasure law
  (registered) + the FEED-PA Road↔Lane dominance (measured), but the through-training gain is owed. Labeled
  FORMALIZATION_PENDING; it is a BUILD+A/B recommendation, not a measured win.
- **The orthogonal-in-G composition claim is a LINEARIZATION** — G-orthogonality of the representation levers is
  argued from their acting on disjoint pieces of δS_τ/δφ; at large weights the cross-terms are not exactly zero.
  Mitigation: the representation levers carry ZERO loss-share (they are basis/DOF, not weighted terms), so the
  cross-confound is second-order; the loss terms (where overlap matters) stay one-per-increment. Still, the
  "compose freely" claim is DERIVED-under-linearization, and P3 red-team should probe the taper×chroma cross-term.
- **The decoupling event (Q4) has ONE clean anchor** (run-1 R-4 + #205 FEED-04b creep); the EMA-lag discriminator
  thresholds (N_ema window, class-concentration cutoff) are DERIVED-form but ASSUMED-value — owed a backtest on
  run-1's own verdict log (SYNTHESIS-REQUIREMENT-B injection test). The event must ship with the 3-part hardened
  detection (backtest + live-path injection + fail-safe cap), not a unit stub (the F-1 launch-path≠config-tests lesson).
- **eps_floor(C)=(5/C)² is the DERIVED FORM; C is ASSUMED** — the derivation says regularize the singular metric;
  it does NOT pin the clip magnitude. Amber-ON is DERIVED; the value is the A/B.

**Pointer 0.19110 UNMOVED — every line here is MEANS until a byte-closed upstream/evaluate.py n600 row < 0.19110.**
