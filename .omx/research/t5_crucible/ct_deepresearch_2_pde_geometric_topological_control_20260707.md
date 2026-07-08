# CT-2 — PDE / GEOMETRIC / TOPOLOGICAL CONTROL OF THE LEVEL-SET OBJECT (T5 crucible, requirement O)

Agent: CT-2 · 2026-07-07 · positive-design contract (L82): every section ends in a DERIVED LAW in
our notation with plugged values, OR a build item, OR a $0 probe with pre-registered band + kill
threshold. Every claim labeled MEASURED / DERIVED / INFERRED / ASSUMED.
review_status: **fresh-research-round-1 (unreviewed)**.

STORES CONSULTED: ORCHESTRATION_LEDGER (full, req A–P incl. the P signal-exposure pin) ·
DRAFT_OPTIMAL_STACK_v3 §0–§2.2d (stage graph · transition law · τ_end=m_q/ln5 derivation ·
crossing arithmetic) · negatives_scale_validity_review_20260707 (full — viscosity REOPENED item 6;
UniWARD asymmetry item 7; TAIL_k law §3) · viscosity_theory_alignment_hunt_20260707 (full — §1
Barles–Souganidis, §3 homogenization EUREKA, §7 junctions; this report does NOT re-derive those,
it composes above them) · corpus_query: "Maslov tropical amortizing the argmax" (deepmath ch.1–6,
`maslov_dequantization_bound_v1`, `tau_eps_hbar_one_dequantization_two_scales_v1`) ·
"adaptive epsilon CFL viscosity eikonal law" (#318/#320, `adaptive_eps_cfl_edge_tracking_v1`,
witness_config_differential_equations_derivation_20260705) · "island birth persistence critical
nucleus homotopy LADDER per-class lambda" (S5 ledger, S2 position, seal_round2 τ* anchors) ·
"S_R reachability" (#268, `margin_saliency_reachability_replaces_texture_proxy_v1`, sweep-B
ledger) · MEMORY.md CURRENT-STATE + L65/L66/L68/L75/L76 · CLAUDE.md (capstone/witness sections).
NOT consulted: raw run dirs beyond memo citations; no training, no n600 spend ($0 report).

Axis discipline: pointer contest-CPU **0.19110 UNMOVED**; everything below is MEANS. Currency
(req J): crossing margin **0.00178 S** · 1e-5 d_seg = 1e-3 S = 56% of margin · 1 KB = 6.82e-4 S ·
decode-gap (R6, MEASURED once) = +1.0427e-4 d_seg = +0.010427 S = 5.86× margin.

---

## §0 EXECUTIVE MAP (field ↔ our object ↔ verdict)

| § | field | our object | verdict |
|---|---|---|---|
| 1 | shape calculus (Hadamard, Delfour–Zolésio) | d_seg IS a shape functional of the predicted separatrix; annulus = the boundary-supported shape gradient's empirical shadow | **IMPORT-NOW** — the signed per-class-pair shape-gradient weight (the principled UNIWARD replacement); τ_end·ln5 ≈ flip-support edge consistency law |
| 2 | level-set method control (Osher–Sethian lineage) | eikonal term = reinitialization; annulus = narrow band; velocity extension | **IMPORT-NOW (1 item)** — velocity-extension reading of the eikonal ramp; topology-preserve = Conley-lite (→§7); rest CONFIRMS the viscosity hunt |
| 3 | PDE-constrained optimization + adjoint | training = discretize-then-optimize of a level-set energy; per-stage ckpts = Griewank | **IMPORT-NOW (framing + 1 signal)** — DtO/OtD mismatch NAMES the checkerboard/jitter pathology; annulus-restricted loss = OtD preconditioning, already ours |
| 4 | PDE backstepping (Krstic; Koga–Krstic Stefan) | analytic lane band = boundary actuator; INR = interior plant | **CAMPAIGN (mostly DEAD for run-1)** — the target-system TRANSFORM idea survives as the band-residual reparametrization; full kernel machinery does not apply (no 1-D spatial causality) |
| 5 | phase-field / Allen–Cahn / MCF control (Γ-convergence) | τ-anneal = phase-field relaxation; islands = thin structures under MCF | **IMPORT-NOW** — Γ-convergence licenses finite-τ control ONLY with τ-indexed laws (clamps-as-τ-laws, confirms negatives-review item 6); critical-nucleus law r* ≈ 0.95σ DERIVED and matches the measured knee |
| 6 | bifurcation control + normal forms + continuation | island birth = fold (saddle-node) of the margin field; LADDER = homotopy continuation | **IMPORT-NOW** — fold-advance law db/dw (birth-epoch control); pseudo-arclength step law gives the 1-Lipschitz easing its constant; washout-filter (transient-only, zero steady-state bias) = the principled M1-quench damper form |
| 7 | Conley index / Morse theory control | islands = isolated invariant sets; τ-descent = parametrized flow | **IMPORT-NOW ($0 certificate)** — persistence-vs-perturbation continuation certificate: pers_i > τ·ln5 + Δ_decode ⇒ island survives; the erasure ∝ 1/persistence law is its empirical shadow |
| 8 | max-plus / tropical control (McEneaney) | argmax = max-plus combination; analytic primitives = max-plus basis | **IMPORT-NOW (naming + 1 law)** — per-class solve-then-max-combine is EXACT at τ=0 and ≤τ·ln5-coupled at finite τ; band/clamp/comb ARE max-plus basis functions (theorem-named architecture); curse-of-dim-free ⊂ our rule-118 free decode |
| 9 | microlocal / wavefront observability | observability of the separatrix through R | **DERIVED bound** — R is all-pass (MEASURED |H_R|≥0.842); the real floor is the uint8 deadzone: δx_min ≈ (1/255)/(0.842·g_I); along-tangent deficit is CHART capacity, not observability |
| 10 | robust / stochastic shape control | decode-gap = plant-model mismatch on the measurement channel | **IMPORT-NOW (1 rule + 1 signal)** — bias-vs-variance split of the gap; sub-margin claims need per-stage parity rows (σ(gap) currently n=1, UNMEASURABLE) |
| 13 | impossibility results | family asymptote (req N) | 5 bounds, each with the MEASUREMENT that decides whether it binds |

---

## §1 SHAPE CALCULUS — the shape derivative of d_seg through R, and the asymmetric per-class-pair law

**Theory (established).** Hadamard structure theorem / Delfour–Zolésio: for a domain functional
J(Ω) with sufficiently regular boundary Γ, the shape derivative under a velocity field V is
boundary-supported with normal pairing: dJ(Ω;V) = ∫_Γ g(x)·V_n(x) ds. Only the NORMAL component
of boundary motion moves J; the density g is the jump of the integrand across Γ. (Sokolowski–
Zolésio, *Introduction to Shape Optimization*; Delfour–Zolésio, *Shapes and Geometries*.)

**Our functional, exactly.** In scorer coordinates (512×384), the per-pair seg distortion is the
area of partition disagreement: d_seg = |Ω_wrong|/|Ω|, Ω_wrong = {x : c_pred(x) ≠ c_gt(x)},
with c_pred = argmax_c ℓ_c(x) of the witness-through-R logits and c_gt the frozen-scorer GT
argmax. Treat the predicted separatrix Γ_pred = ∪_{i<j} Γ_{ij} (class-pair pieces) as the shape
variable. **DERIVED (direct application of the structure theorem):**

    d(d_seg)(Γ_pred; V) = (1/|Ω|) Σ_{ij} ∫_{Γ_ij} s_ij(x) · V_n(x) ds,
    s_ij(x) = +1 if the GT label at x equals the class on the side V_n moves INTO being wrong,
              −1 if the motion corrects x  —  i.e. s_ij(x) = sgn agreement flip, and s_ij
    changes sign EXACTLY where Γ_pred crosses Γ_gt.

Three consequences, each checkable against a measured anchor:

1. **Boundary-supported ⇒ the annulus.** The exact gradient of d_seg lives ONLY on Γ_pred.
   MEASURED shadow: ~97% of d_seg mass in the ~4.7%-area annulus (#333). The 3% off-annulus mass
   is the τ-smoothing tail (below), not a violation.
2. **The τ-smoothed version IS the margin-saliency object.** Replacing argmax by softmax_τ, the
   per-pixel derivative of the smoothed disagreement w.r.t. a logit perturbation concentrates on
   the set {m(x) ≲ τ·ln5} with weight ∝ (1/τ)·sech²-type bump of width τ·ln5 in margin units —
   i.e. **the smoothed Dirac layer δ_τ(m)·|∇m|**. The #141 margin-saliency map is the MAGNITUDE
   of the τ-smoothed shape gradient. **What it is missing is the SIGN field s_ij(x)** — the map
   as built is unsigned/pooled, exactly the estimator shape the asymmetry addendum (req L)
   flags. DERIVED consistency check with plugged values: at v3's τ_end = 0.062, the layer width
   is τ_end·ln5 = 0.062×1.60944 = **0.0998 ≈ 0.10 = the MEASURED flip-support edge**
   (flip-rate 0.764 for m<0.10, ~0 above). v3's τ_end derivation and the shape-calculus layer
   width are the SAME statement seen from two fields — the τ_end=0.062 choice makes the smoothed
   shape-gradient support coincide with the flip population. [DERIVED; anchors MEASURED]
3. **The asymmetric (one-sided) shape derivative per class-pair.** Because s_ij is a SIGN field
   per class-pair per side, the exact loss weight the shape calculus demands is
   **w(x) = σ_ij,dir(x) · δ_τ(m(x)) · |∇m(x)| · s_ij(x)** — a SIGNED one-sided density, where
   σ_ij,dir are per-class-pair per-DIRECTION multipliers (Road→Lane FP ≠ Lane→Road erasure). The
   pooled-unsigned UNIWARD proxy measured "at chance" (Pearson −0.033, L76) is precisely this
   law with the sign field integrated out — the negatives-review's flagship asymmetry case (§1
   item 7 there) is, in shape-calculus terms, *the theorem predicting the measurement*: an
   unsigned pooled estimator of a signed density has zero expectation when the two sides carry
   opposite signs with comparable mass. [DERIVED mechanism for a MEASURED null]

**Through R (the composition).** R = bicubic↑(384→874) ∘ uint8 ∘ bilinear↓(→512×384), scorer S
frozen. R is linear except uint8. The chain rule transports the boundary-supported gradient to
train coordinates by the ADJOINT of the linear parts: g_train = R_lin^T (S^T g_shape); the resize
adjoints smear the boundary Dirac layer by the interpolation-kernel footprint (bicubic support 4,
scale ratio 874/384 ≈ 2.28 ⇒ ~2 px in train coordinates) — **DERIVED: the trainable annulus
width = τ·ln5 margin-layer width ⊕ ~2 px kernel smear**, which is why the measured annulus is
~4.7% of area rather than a 1-px curve. uint8 contributes the STE bias (the ~35% analytic-vs-true
HVP gap, MEASURED, chain-A — point-bound per negatives review 1a). No new pathology: the shape
gradient survives R with its sign structure intact; only its MAGNITUDE calibration is
STE-distorted, which is exactly why the exact through-R reachability S_R (#268, θ-independent,
signed at the flip level) is the right realized weight — S_R is the numerically-exact evaluation
of this section's density where the analytic chain is STE-biased.

**LAW (shape-gradient loss weight, class-(e) fractional/partial per the P2 contract):**

    L_boundary = Σ_{ij} Σ_{dir∈{i→j, j→i}} σ_ij,dir · Σ_x 1[pair(x)=(i,j), dir(x)=dir]
                 · δ_{τ(t)}(m(x)) · |∇m(x)| · hinge_dir(ℓ(x))
    with δ_{τ}(m) of width τ(t)·ln5 (RE-DERIVED each stage — τ-indexed, per §5),
    σ_ij,dir initialized from the $0 per-side re-test already queued (negatives review item 7:
    per-class-pair per-direction ρ from cached S_R + texture fields; kill |ρ|<0.1 both sides).

**Cost/benefit denominated (req J):** this is a loss-shaping lever (0 bytes). Its efficacy bound
is the lane-erasure share: lane = 19% of flips ≈ 0.44 of composed-surface share (v3 §0.0) —
even a 10% relative improvement on the lane leg ≈ 4e-5 d_seg ≈ 4e-3 S ≈ 2.2× crossing margin.
Falsification: A/B signed-hinge vs unsigned margin-gate at matched schedule; kill if Δd_seg < 0
or < attribution floor.

**REQUIRED SIGNAL (req P):** per-class-pair per-DIRECTION one-sided margin histograms + flip
masses at every stage boundary (generated? cached S_R + #333 annulus rows exist for ONE
checkpoint; NOT per-stage) → record as per-stage F-rows (extend req-F #5/#8 family) → consumers:
σ_ij,dir fit, §7 persistence certificates, the τ*_k per-side re-derivation in TAIL_k cycles.

---

## §2 LEVEL-SET METHOD CONTROL — velocity design, reinitialization, narrow band

**Theory.** Osher–Sethian lineage: (i) evolve φ by φ_t + F|∇φ| = 0 with a DESIGNED normal speed
F (velocity design = the control); (ii) REINITIALIZE φ to a signed distance function
(|∇φ|=1) periodically to keep the interface well-conditioned — but reinitialization MOVES the
zero level set slightly (a known artifact; Sussman–Smereka–Osher fixes constrain the interface);
(iii) NARROW-BAND methods (Adalsteinsson–Sethian) update only a tube around the interface —
O(interface) not O(domain) cost; (iv) VELOCITY EXTENSION (Adalsteinsson–Sethian 1999): extend F
off the interface so that ∇F·∇φ = 0, which PRESERVES the signed-distance property during
evolution and reduces the need to reinitialize at all.

**Mapping + what is genuinely new here** (the Barles–Souganidis/monotone-scheme and viscosity
questions are already settled in viscosity_theory_alignment_hunt §1/§5 — composed, not repeated):

1. **Our eikonal term IS soft reinitialization, and the literature's warning transfers.** The
   penalty λ_eik(|∇m|−1)² is the variational form of continuous-in-time reinitialization.
   Sussman-era result: reinitialization helps WHEN the field has flattened (|∇φ|≪1 near the
   interface — the interface becomes numerically invisible) and HURTS (moves the interface) when
   applied strongly to an already-well-conditioned field. Translated: the eikonal weight should
   be gated on the measured conditioning statistic **E_annulus[(|∇m|−1)²]**, not run at constant
   λ. v3 ships eikonal as a RAMP 0.05→0.10 at the tau boundary; the level-set literature's form
   is EVENT-CONDITIONED (class-(d)): raise λ_eik when annulus conditioning degrades, hold floor
   otherwise. [INFERRED transfer; the run-1 eikonal ramp under live guards is the first fair
   test either way — viscosity REOPENED per negatives review item 6.]
2. **Velocity extension = the missing companion of the eikonal term.** Reinitialization keeps
   |∇m|=1; velocity extension keeps it BY CONSTRUCTION by making the update constant along
   normals near Γ. Our gradient updates are INR-parametric (global support) — we cannot impose
   ∇(δm)·∇m = 0 exactly, but we can PENALIZE its violation on the annulus:
   **L_ext = λ_ext · E_annulus[(∇(δm_pred)·∇m)²]** where δm_pred is the per-step field update
   (computable from two consecutive cached fields at verdict cadence — $0 telemetry first, lever
   later). Prediction (pre-registered): runs with lower measured extension-violation show lower
   boundary-jitter share of d_seg. This is a NEW, cheap diagnostic that decides whether a lever
   is warranted before building it. [DERIVED analog; probe $0]
3. **Narrow band ≡ the annulus concentration, already ours.** ~97% of d_seg mass in 4.7% area
   (MEASURED) says the score-relevant computation IS narrow-band; the capacity-routing lever
   (KKT waterfill on margin saliency) is the parametric narrow-band. CONFIRMS-existing; no new
   build. The one import: narrow-band methods RE-BUILD the band as the interface moves — our
   annulus masks must be RE-COMPUTED per stage (stale-band = the classic narrow-band bug);
   verify the #333 annulus rows are per-stage, not frozen (build check, ~0 LOC — an assert).
4. **Topology control (Han–Xu–Prince 2003, simple-point constraint):** level-set segmentation
   preserves known topology by REFUSING updates that change local digital topology (simple-point
   test), at nominal overhead. Inverted for us: we want CONTROLLED topology CHANGE (island
   birth) and PREVENTION of island death. The simple-point machinery gives the O(1)-per-pixel
   test for "this update kills a component"; as a control it is a HARD projection (reject the
   flip) — harsher than our margin-gated island support but implementable at verdict cadence as
   an ALARM: count island-death events where the dying island's persistence exceeded the §7
   certificate threshold (those are controller failures, not physics). [Build: telemetry only.]

**LAW (event-conditioned eikonal, class-(d)):** λ_eik(t) = λ_floor + (λ_max−λ_floor)·
1[E_annulus[(|∇m|−1)²] > c_cond for 2 consecutive windows], with λ_floor = v3's 0.05, λ_max =
0.10, c_cond calibrated from the run-1 trace's first TAU-stage window (first run measures it —
same class-(e) posture as eps_c in v3 §2.2). Fail-safe: the v3 ramp is the cap-path (req B).

**REQUIRED SIGNAL:** per-epoch **annulus conditioning row** E_annulus[(|∇m|−1)²] + per-window
extension-violation E_annulus[(∇δm·∇m)²] (generated? NO — new; cheap: two cached fields at
verdict cadence) → telemetry F-family → consumers: λ_eik event law, §5 clamp-τ-law check,
run-2 velocity-extension lever decision.

---

## §3 PDE-CONSTRAINED OPTIMIZATION + ADJOINT — which discretization order are we, and does the mismatch explain measured pathology?

**Theory.** Two routes to gradients of PDE-constrained objectives: **optimize-then-discretize**
(OtD: derive the continuous adjoint/shape gradient, then discretize it) vs **discretize-then-
optimize** (DtO: discretize the state equation, differentiate the discrete system exactly —
autodiff). Standard results: DtO gradients are EXACT for the discrete objective but can be
INCONSISTENT with the continuous gradient when the discretization is non-monotone/unstable —
DtO faithfully optimizes the discretization's artifacts (spurious high-frequency modes receive
real gradient mass). OtD gradients are consistent with the continuum but not exact for the
discrete objective (optimizer can stall at discrete-level noise). Checkpointing (Griewank–
Walther `revolve`): optimal treeverse schedules for adjoint memory.

**Classification of us.** Training the INR by autodiff through render→R(STE)→frozen scorer is
**pure DtO**. The continuous object it should approximate is §1's boundary-supported shape
gradient. The mismatch predictions and their measured shadows:

| DtO-vs-OtD mismatch prediction | measured shadow |
|---|---|
| gradient mass on discretization artifacts (spectral/collocation checkerboard modes) | ep110 fastest-growing mode at checkerboard k_max (MEASURED, #318) |
| optimization of sub-continuum noise → high-frequency boundary jitter | 97%-annulus JITTER (not region-miss); 44.6% singleton flip confetti (MEASURED) |
| exactness for the wrong (discrete) functional | training-vs-decoded gap +1.0427e-4 (MEASURED, R6) — DtO optimizes the TRAINING-side discrete functional, not the shipped decode |

**The import is a HYBRID, and we already half-own it:** keep DtO (exactness through the real R
and scorer is non-negotiable — the analytic-HVP ≠ true-curvature lesson says continuous models of
this plant mislead), but PRECONDITION with OtD structure: restrict/weight the loss on the §1
boundary layer (margin-gated support, S_R weights, annulus routing). That is precisely what the
measured-effective levers do; this section's contribution is the NAME and the negative space:
**any lever that adds gradient mass OFF the boundary layer is paying the DtO artifact tax with
no continuum counterpart** — use as a review checklist row (sister of the viscosity hunt's
comparison-principle row): *new loss terms declare their continuum shape-gradient reading; a
term with no boundary-supported continuum limit is presumed artifact-feeding until A/B'd.*
[DERIVED discipline; costs 0]

**Checkpointing:** per-stage checkpoints + EMA shadow already exceed Griewank needs for our
1-level adjoint (no through-time BPTT); `revolve` becomes relevant ONLY if a future lever
backprops through a multi-step inner PDE evolution (e.g. trained MBO/MCF inner loops — none in
v3). DEFER with reason. [ASSUMED scope]

**LAW (review discipline, class-(e)):** every new loss term ships with a declared continuum
symbol: {boundary-supported (shape-gradient class) | bulk-proper (§5 comparison-safe class) |
NONE (artifact-risk — requires A/B before adoption)}. ~1 row in the review contract; 0 LOC.

**REQUIRED SIGNAL:** per-stage **off-annulus gradient-mass fraction** (‖g·(1−annulus)‖²/‖g‖² at
verdict cadence; generated? NO — one masked reduction on the existing backward; cheap) →
F-family row → consumers: the artifact-tax audit, §2 conditioning law, DtO-health trend for
run-2 (rising off-annulus mass late in TAU = the checkerboard precursor, an early-warning alarm).

---

## §4 PDE BACKSTEPPING (Krstic; Koga–Krstic Stefan problem) — boundary control of a moving interface

**Theory.** Backstepping stabilizes PDEs by an invertible Volterra (spatially-causal) integral
transformation mapping the plant to a provably-stable TARGET SYSTEM; the kernel solves a
Goursat PDE. Koga–Krstic (*Backstepping Control of the One-Phase Stefan Problem*, 2016;
arXiv:1607.04345, 1703.05814): a diffusion PDE on a time-varying domain coupled to an ODE for
the moving interface — boundary heat-flux control exponentially stabilizes the interface
position with certified physical constraints (melting monotonicity).

**Honest applicability audit.** The Stefan analogy is seductive (our separatrix is a moving
interface; the analytic lane band is a boundary-adjacent actuator) but the machinery needs:
(a) a 1-D spatially-causal structure for the Volterra kernel (our domain is 2-D, interface a
curve network); (b) an actuator AT the boundary of the DOMAIN (ours acts through the loss,
everywhere); (c) a plant we must STABILIZE in time (our "time" is training epochs and the flow
is a descent, already Lyapunov-stable by construction — the loss is the Lyapunov function).
Verdict: **the kernel machinery does NOT transfer. DEAD for run-1 as mechanism.** [DERIVED
negative — cheap to state now, saves a campaign detour.]

**What survives (one idea, real):** backstepping's essence is *re-express the plant in
coordinates where the hard part is already solved, control only the residual*. That is exactly
the band-to-INR hand-off: let m_lane = max(band_analytic(x; poly, w), m_INR(x)) (max-plus form,
§8) and train the INR ONLY on the residual the band does not explain. The "target system" is
the band-subtracted field; its stability/smallness is certifiable (residual d_seg with band
frozen = a MEASURED quantity per stage). The principled hand-off law the charter asked for is
therefore NOT a backstepping kernel but the **max-plus residual decomposition of §8 + the
per-stage residual certificate**: freeze band → measure residual flip mass per class-pair →
train INR against residual only (margin-gated). v3 already trains WITH the band; the delta this
section adds is the certificate row, not a new mechanism.

**LAW:** none new (negative result + pointer to §8). Build item: per-stage **band-residual
row** — flip mass explained by band alone vs composed (one extra masked verdict at stage
boundaries, reuses the with/without machinery of req-F #8).

**REQUIRED SIGNAL:** the band-residual row above (generated? partially — req-F #8 pairs
with/without at ACTIVATION only; extend to every stage boundary) → consumers: hand-off
certificate, §13 asymptote accounting (what fraction of lane mass is analytically explained).

---

## §5 PHASE-FIELD / ALLEN–CAHN / MCF CONTROL — does control designed at finite τ survive τ→0, and what forcing preserves thin structures?

**Theory.** (i) Γ-convergence of CONTROL, not just energy: for phase-field structural/shape
optimization, minimizers of the ε-regularized control problems converge to minimizers of the
sharp-interface problem as ε→0 (Blank–Garcke et al., *Sharp interface limit for a phase field
model in structural optimization*, SICON 2016 / arXiv:1409.7586), including convergence of the
first-variation (the optimality system). Conditions: correct ε-scaling of the perimeter term and
well-posedness at each ε. (ii) MCF extinction: a circular island of radius r dies under MCF in
t_ext = r²/2; thin structures die first (our #284 law-8 anchor). (iii) Forced MCF in
heterogeneous media PINS below threshold forcing (viscosity hunt §3).

**Import 1 — the license and its price (confirms + sharpens negatives-review item 6).** The
Γ-convergence-of-control results LICENSE designing at finite τ and trusting the τ→0 limit,
**but only law-wise, not number-wise**: every control constant tuned at coarse τ must be
re-expressed as a function of τ (the optimality system converges; a frozen constant does not
track it). This is the theorem-form of the negatives-review finding that the adaptive-ε clamps
(0.3/0.7) are "coarse-point constants wearing a law's clothing." **DERIVED rule: every
τ-adjacent constant in the stack {ε clamps, λ_eik floor/max, δ_τ width (§1), island-support
gate margin, c_cond (§2)} ships as c(τ) with its coarse-τ value as c(0.216) and a declared
scaling exponent, re-validated at the τ∈{0.216, 0.12, 0.062} sampling points v3's F12 already
plans.** The $0 clamp-binding check (negatives review item 6) is the first instance; this
section generalizes it to the full constant inventory. [DERIVED discipline]

**Import 2 — the critical-nucleus law, derived and matched to the measured knee.** Under
smoothing of scale σ followed by thresholding (the τ/MCF probe's operative model), a stripe of
half-width r survives iff its center retains majority mass: erf(r/(σ√2)) > 1/2, i.e.
**r* = √2·erfinv(1/2)·σ ≈ 0.674·√2·σ ≈ 0.95σ**. Plugged: the nucleus-knee probe ran σ=1.5 ⇒
r* ≈ 1.43 px. MEASURED knee: native dashes 44.6% survival (⇒ native r ≈ r*, half above half
below), +1 px dilation → 90.0%, +2 px → 98.3% — quantitatively consistent with a survival
boundary at r*≈1.4 px given a native dash-width distribution straddling it. **LAW (island
support sizing, class-(b) ramp): the eased SDF-dilation radius law needs r(t) ≥ r*(t) =
0.95·σ_eff(t) where σ_eff(t) is the CURRENT effective smoothing scale = max(τ(t)-interface
width, ε-viscous cutoff, R-Nyquist ≈ 1 px); as τ anneals 0.216→0.062, r* SHRINKS — the dilation
homotopy's END state may release protection exactly on schedule rather than by a hand-tuned
ramp.** v3's r(t) 1-Lipschitz anneal over 275 ep gets its endpoint from physics: release when
r*(t) falls below the native dash half-width. [DERIVED with plugged values; anchors MEASURED]

**Import 3 — forcing form that preserves thin structures.** Forced MCF V = κ + f preserves a
structure of max curvature κ_max iff f > κ_max on it (else extinction); for a dash of half-width
r, κ_max ≈ 1/r ⇒ **minimum island-forcing weight ∝ 1/r_i per island — which is exactly the
per-island 1/persistence weight v3 R13-gates** (persistence and r are monotonically linked for
near-critical islands). The topology/curvature reading upgrades "1/pers weighting" from
heuristic to the MCF-forcing threshold law. CONFIRMS v3 §0.3b; adds the release condition from
Import 2. [DERIVED]

**REQUIRED SIGNAL:** τ-indexed interface-width telemetry: per-stage measured interface width
(fit of the m-profile transverse to Γ), σ_eff(t) components, and the r*(t) trajectory
(generated? NO — new row; cheap on cached fields) → F-family → consumers: island-release law,
§1 δ_τ width validation, clamp-τ-law re-derivation ($0 probe already queued).

---

## §6 BIFURCATION CONTROL + NORMAL FORMS + CONTINUATION — island birth as a controlled fold, LADDER as continuation, washout filters for transitions

**Theory.** (i) Bifurcation control (Abed–Fu lineage; washout-filter feedback, Wang–Abed 1995):
small feedback can DELAY/ADVANCE a bifurcation or soften its transient; washout filters (high-
pass in the feedback path) act ONLY on transients — zero steady-state offset, so equilibria
(here: converged score) are untouched. (ii) Normal form of the fold: ṁ₀ = μ + m₀² — the local
maximum of the runner-up margin field crosses 0 when the unfolding parameter μ does; a constant
forcing h shifts the fold linearly: μ → μ + h. (iii) Numerical continuation: pseudo-arclength
step-size control Δλ ≤ c/‖dθ/dλ‖ keeps the tracked branch (adiabatic tracking; matches the
viscosity hunt §6's homotopy-not-temperature reading of τ).

**Island birth = a fold of the margin field, and the birth-epoch control law.** A new island of
class c is born where the local max of ℓ_c − ℓ_runner crosses 0. With per-class amplify weight
w_c entering the loss ~linearly in the logit drive near threshold, the fold-advance law is
first-order exact: **db_c/dw_c = −(∂μ_c/∂w_c)/(dμ_c/dt)** — birth epoch b_c moves linearly in
w_c at rate set by the measured per-class λ/logit-drive trajectory slope (dμ_c/dt from
telemetry). This turns v3's per-class amplify weights (w_lane=1.0, w_movable=0.28) from static
constants into a **calibratable birth-SCHEDULER**: pick target birth epochs (lane islands must
be born BEFORE τ(t)·ln5 shrinks below their persistence — the §7 certificate), read dμ_c/dt off
the run-1 trace, set w_c(t) to hit the targets. Run-1 posture: keep v3's constants, RECORD the
trace, fit the law for run-2 (measure→sweep→derive, req M). [DERIVED law; calibration OWED]

**Washout-filter import — the principled M1-quench damper.** The measured cold-Muon/stage-
boundary quenches (M-S2-1: +27.5%; ep300 bump) are TRANSIENTS at controlled transitions. The
bifurcation-control literature's tool for exactly this is transient-only feedback: a high-pass
damper d(t) = k·HP[dL/dt]₊ applied to the effective LR at stage boundaries — acts iff the loss
is spiking, decays to zero, and CANNOT bias the converged score (zero DC gain — the same reason
Wang–Abed used it: equilibria preserved by construction). v3's rewarmup/easing repairs are
open-loop ramps; the washout form is the closed-loop version with a structural no-bias
guarantee, which makes it admissible under the "score-neutral by construction" observability
rule even as an ACTUATOR (it vanishes at steady state). Build: ~20 LOC on the LR path +
injection test (req B). Class-(d) event law. [INFERRED transfer; falsifiable: transition
transient area vs the open-loop ramp at matched schedule.]

**LADDER = homotopy continuation, and the step-size law.** The per-class-λ LADDER homotopy is
numerical continuation in λ; the continuation literature's binding rule is the pseudo-arclength
step bound Δλ_step ≤ c/‖dθ/dλ‖. v3's "1-Lipschitz easing" is this law with the response norm
ASSUMED ≤ 1; the import is to MEASURE the response norm (per-class parameter drift per unit λ
per epoch — available from checkpoint diffs) and let the easing rate adapt: **Δλ(t) =
Δλ_max · min(1, c_arc/‖Δθ‖_window)** — slows the homotopy exactly when the network is
responding violently (the adiabatic guard), speeds it when response is small (recovering the
wasted-epoch margin M4 measured). Class-(c) self-deriving law; fail-safe cap = the v3 fixed
ramp (req B). [DERIVED; constants from run-1 trace]

**REQUIRED SIGNAL:** per-class **fold telemetry**: per-epoch count + max-value of near-threshold
local maxima of (ℓ_c − ℓ_runner) below 0 (the "pre-birth field"), per-class dμ_c/dt, and
window ‖Δθ‖ response norms (generated? NO — pre-birth field is one extra reduction on cached
logits; ‖Δθ‖ from existing ckpt cadence) → F-family → consumers: birth-scheduler fit (run-2),
adaptive easing law, §7 certificates.

---

## §7 CONLEY INDEX / MORSE THEORY CONTROL — persistence certificates for islands through the τ-descent

**Theory.** Conley index theory: an ISOLATED invariant set (one admitting an isolating
neighborhood N whose invariant part is interior) has an index invariant under CONTINUATION —
if a parametrized family of flows keeps N isolating along the path, the invariant set persists
(possibly deformed) with the same index (Conley; Mischaikow–Mrozek; combinatorial/persistence
version: Dey–Mrozek et al., arXiv:2003.05579). The obstruction to continuation is precisely the
invariant set touching ∂N along the parameter path.

**The certificate, derived in our objects.** Take the parametrized flow = the training/anneal
path over stages k (τ_k descending), the invariant set = an island I of class c in the argmax
partition, and the isolating neighborhood = the margin superlevel tube N_I = {x ∈ I-basin :
m(x) ≥ η}. The island fails to continue only if some pixel path exits N_I, i.e. the margin field
along the island is driven below the perturbation scale. The perturbations acting between what
we train and what is scored are (a) the τ-smoothing (bounded by τ·ln5 in logit units — Maslov,
MEASURED-law) and (b) the decode/byte-close channel (R6: +1.0427e-4 d_seg; its LOGIT-unit
magnitude Δ_dec^logit is NOT yet measured — see signal). **DERIVED CERTIFICATE:**

    island I survives stage k and decode  ⟸  pers(I) > τ_k·ln5 + Δ_dec^logit
    (pers(I) = the island's birth-death margin amplitude, our existing persistence value)

Plugged at v3's endpoint: τ_end·ln5 = 0.0998 ⇒ islands with pers > 0.10 + Δ_dec^logit are
CERTIFIED through the final stage; the measured erasure ∝ 1/persistence law (L75, MBO 95.7%
lane) is the empirical shadow of the uncertified population dying first. Two uses:
1. **Controller predicate ($0):** at each stage boundary, count births with pers below the
   NEXT stage's certificate threshold — those are "born to die" and their support loss is
   WASTED unless the island-forcing (§5 import 3) is scheduled to RAISE pers above threshold
   before release. This converts the island curriculum from hope to a per-island pass/fail
   ledger. [DERIVED; consumes existing persistence machinery]
2. **Death-alarm (with §2 item 4):** an island dying WHILE certified = controller/instrument
   failure (not physics) → ALARM row, req-B tested.

**Honest boundary:** the certificate is SUFFICIENT-not-necessary (sub-threshold islands MAY
survive; matches 44.6% native survival), and pers is measured on the smoothed field so the two
sides of the inequality are not fully independent — the $0 backtest below calibrates the
practical threshold. **$0 PROBE (pre-registered):** on the existing birth-death ledger
(`birth_death_persistence_dseg_20260630`), compute survival-vs-pers curves at the two known τ
points; band: certified-survival ≥ 95%; kill: certified-survival < 80% ⇒ the certificate needs
a safety factor s·(τ·ln5), fit s.

**REQUIRED SIGNAL:** (a) per-island birth-death ledger WITH persistence values PER STAGE
BOUNDARY (generated? per-run forensics exist; not per-stage live) → make it a stage-boundary
F-row; (b) **Δ_dec^logit** — the decode-gap expressed in logit/margin units, one masked
comparison at byte-close (generated? NO — R6 measured d_seg units only) → consumers: the
certificate, §10 robust margin, §13 bound M2.

---

## §8 MAX-PLUS / TROPICAL CONTROL (McEneaney, Akian–Gaubert) — superposition, basis methods, and what breaks at finite τ

**Theory.** HJ semigroups are max-plus LINEAR (Maslov; Kolokoltsov–Maslov): the Lax–Oleinik
evolution distributes over max and commutes with scalar (additive) shifts. McEneaney's
curse-of-dimensionality-free method (SICON 2007, 10.1137/040610830; convergence-rate SICON
follow-ups) exploits this: represent the value function as a max of finitely many basis
functions (quadratics), propagate EACH INDEPENDENTLY, recombine by max — the semigroup's
max-plus linearity makes the per-basis propagation exact; cost grows in basis count, not
dimension. Akian–Gaubert max-plus spectral theory characterizes the eigenspaces.

**Superposition for us — exact statement, both directions.** Our terminal object is the argmax
partition c*(x) = argmax_c ℓ_c(x): the partition IS the max-plus combination of the five
per-class logit fields — **per-class solve/train-then-max-combine is EXACT at τ=0, by
definition, with no condition**. What breaks at finite τ: (a) the softmax normalization couples
classes (log-sum-exp is max-plus-nonlinear at τ>0) — coupling bounded by the Maslov law:
|logsumexp_τ − max| ≤ τ·ln5 per pixel; (b) the LOSS (CE) couples classes through the same
normalization, so per-class training signals are not independent above τ=0. **DERIVED
CONSEQUENCE with plugged values: any per-class-decomposed solve/fit whose per-class fields are
each accurate to δ_c yields a combined partition whose decision error is confined to
{m < max_c δ_c + τ·ln5}; at τ_end=0.062 the coupling term is 0.0998 — exactly the flip-support
edge.** So per-class decomposition is FREE for the bulk and exactly-marginal on the flip
population — the design consequence is not "decompose everything" but "decompose the SOLVABLE
per-class pieces and spend the coupled (joint) training budget ONLY on the flip annulus" —
which is the §1 boundary-layer law again, arrived at from the algebra side. [DERIVED]

**The basis-method reading of our architecture (theorem-naming, req K).** The witness's
composed field m_c = max(INR_c, band_c, clamp_c, comb_c ⋯) IS a max-plus basis expansion:
analytic primitives = basis functions, max = the combination, and (rule 118) the basis
GENERATORS are free at decode — **McEneaney's "propagate basis functions independently" is our
"train/derive levers independently, compose by max"**; the curse-of-dimensionality-free
property is the PDE-numerics name for what the FREE-decode column of the rate table already
exploits. Two actionable corollaries:
1. **Solve-don't-train candidate (#342 register):** fit a max-plus expansion (max of K
   quadratics/low-order polynomials in (x,y,pair-index)) DIRECTLY to the frozen scorer's
   per-class logit fields on the annulus — per-basis fitting is small least-squares (each
   region where a given basis element attains the max is fit independently — tropical
   polyhedral decomposition); the INR then only carries the residual. Cost: $0 prototype on
   cached logit fields; kill: K ≤ 64 elements per class fails to reach band-level accuracy on
   the annulus (else the byte cost of K coefficient sets enters the KKT λ_bytes law). This is
   the principled generalization of band+clamp (K=1 elements) to K>1. [Build/probe]
2. **Approximation honesty (feeds §13):** max-plus/tropical approximation is EFFICIENT exactly
   for max-structured (piecewise-smooth, kink-carrying) targets and INEFFICIENT for smooth
   bulk (basis count blows up where curvature range is wide) — the separatrix is the max-plus-
   native object (req K's "native format" for the boundary), the cartoon bulk stays INR/
   curvelet. Our split architecture is the two-semiring split; do not migrate the bulk.

**REQUIRED SIGNAL:** per-class logit fields EXPORTABLE SEPARATELY at stage boundaries + the
annulus-restricted per-class residual after subtracting each analytic primitive (generated?
logit caching exists per verdict; per-class export + per-primitive residual rows are new,
cheap) → consumers: max-plus fit probe, §4 band-residual certificate, per-class solve ledger.

---

## §9 MICROLOCAL / WAVEFRONT OBSERVABILITY — what the R channel lets us see and steer

**Theory (applied, short).** Observability of a perturbation through a linear filter chain +
quantizer: frequencies with |H| > 0 are observable above the quantizer's effective noise floor;
the geometric-control-condition machinery (Bardos–Lebeau–Rauch) is trivially satisfied here
(full-domain observation — every ray meets the observation set), so the binding facts are the
measured filter magnitude and the uint8 deadzone.

**Derived bounds in our numbers.**
- **Linear leg (MEASURED, scale-robust — negatives review item 5):** |H_R| ∈ [0.842, 1.0] to
  render-Nyquist. NO across- or along-boundary spatial frequency is unobservable; deconvolution
  gain is bounded by 1/0.842 (+1.5 dB max) — the along-tangent 3.2× deficit CANNOT be an
  observability deficit; it is chart capacity (CONFIRMS the basis-wall verdict independently).
- **Quantizer leg (DERIVED):** a normal boundary displacement δx produces a pixel-intensity
  change ≈ δx·g_I (g_I = intensity gradient across the edge, normalized units/px); it survives
  uint8 iff δx·g_I·|H_R| > 1/255 (half-LSB deadzone, no dither) ⇒
  **δx_min = (1/255)/(0.842·g_I)**. Plugged: lane-paint edge contrast g_I ≈ 0.2–0.5 ⇒
  δx_min ≈ 0.009–0.023 px. Sub-pixel boundary control IS observable through uint8 wherever
  edge contrast is healthy; the deadzone binds only on LOW-CONTRAST boundaries (far-range lane,
  shadow edges) — which is a per-class-pair, per-range statement (req H): the deadzone
  population should be MEASURED, not assumed (signal below). Dither/phase levers (#149 class)
  are the standard quantizer-linearization repair IF that population is material.
- **Controllability (the STE half):** the training gradient uses d(uint8)/du = 1 (STE) where
  the true derivative is 0 a.e. — controllable-in-training but with the measured ~35% analytic-
  vs-true curvature distortion (chain-A, point-bound). Nothing new to build; the §3 discipline
  covers it.

**LAW (deadzone census, $0 probe, pre-registered):** on cached fields, count flip-annulus pixels
whose required through-R intensity change (from S_R) is < 1/255: band — if that mass converts
to < 0.3× crossing margin (< 5.3e-6 d_seg), the deadzone is NOT binding and #149 stays DEFER;
if > 1× margin, #149 (sub-pixel phase/dither) enters the duty-to-measure queue with a real
prior. Kill threshold for the probe itself: none (census, not a lever).

**REQUIRED SIGNAL:** per-class-pair **edge-contrast g_I histograms on the annulus** + the
deadzone-mass row above (generated? NO — one pass over cached S_R + frames) → consumers: #149
adjudication, §13 bound M1, far-range lane sub-curriculum decision (req H).

---

## §10 ROBUST / STOCHASTIC SHAPE CONTROL — the decode-gap as plant-model mismatch

**Theory.** Small-gain/robust-margin reasoning for a measurement-channel mismatch: if the
design loop optimizes surrogate J_train but ships J_decode = J_train + Δ, improvements claimed
on J_train transfer iff they exceed the VARIATION of Δ over the compared configs — the
SYSTEMATIC part of Δ cancels in matched A/Bs through the same decode; the VARIANCE part does
not, and is the honest uncertainty on every sub-margin claim.

**Our numbers.** Δ measured ONCE (R6): +1.0427e-4 d_seg = +0.010427 S = 5.86× crossing margin.
With n=1 we know the BIAS at one θ and know NOTHING about σ(Δ) across θ/stages/levers.
**DERIVED RULE (bias-variance split of the parity row):** (a) all ABSOLUTE crossing arithmetic
must carry Δ explicitly (v3 already does post-R6 — CONFIRMS); (b) all RELATIVE (A/B) claims at
matched decode inherit only σ(Δ); until σ(Δ) is estimated (n ≥ 3 stage-boundary parity rows),
**every A/B delta below 5.86× margin ~ 0.0104 S is PROVISIONAL against decode drift** — this is
currently the second-largest attribution floor after the self-orient gap (0.015 S, req-F #6).
Pre-registered band: σ(Δ) < 0.5× crossing margin (8.9e-4 S) ⇒ decode drift retires as an
attribution term; σ(Δ) > 2× margin ⇒ the byte-close path itself becomes a bug hunt before any
sub-margin claim is admissible (fail-closed).

**REQUIRED SIGNAL:** the **parity row PER STAGE BOUNDARY** (byte-close → decode → n600 verdict
vs training-side), not once per run (generated? once, R6; per-stage is the same machinery on
per-stage ckpts — mandated ckpts exist) → consumers: σ(Δ) estimate, §7 Δ_dec^logit, the
attribution-floor ledger.

---

## §11 RANKED ADOPTION LIST (predicted band · cost · $0 probe; PowerPlay-ordered: cheapest-decisive first)

| rank | import | class | predicted band (S-units, req J) | cost | $0 probe / kill |
|---|---|---|---|---|---|
| 1 | **§7 Conley persistence certificate** (pers > τ_k·ln5 + Δ_dec^logit) | controller predicate + alarm | protects the island leg of the crossing triple (lane share 0.44 of composed flips); mis-scheduled births currently unbounded | ~30 LOC on existing persistence machinery | survival-vs-pers backtest on the 0630 ledger; kill: certified-survival < 80% |
| 2 | **§1 signed per-class-pair shape-gradient weight** (σ_ij,dir · δ_τ(m) · signed hinge) | loss lever (0 bytes) | lane-leg 10% relative ⇒ ≈ 4e-3 S ≈ 2.2× margin; DERIVED mechanism for the UNIWARD null | builds on queued per-side ρ re-test + existing margin-gate | $0 per-side ρ (queued); kill: \|ρ\|<0.1 both sides of every major pair |
| 3 | **§5 τ-indexed constants discipline + r* island-release law** (r* = 0.95·σ_eff(t)) | law-form upgrade (0 LOC beyond the clamp checks) | prevents silent re-freeze of adaptive branches at fine τ (the 0.3/0.7 clamp-binding risk — negatives item 6); island release on physics not hand-ramp | $0 arithmetic + the queued clamp-binding check | clamp-binding fraction > 90% at fine τ ⇒ re-derive as c(τ) |
| 4 | **§10 per-stage parity rows → σ(Δ)** | signal (apparatus) | converts a 5.86×-margin attribution blocker into a measured term | reuse R6 machinery per stage | σ(Δ) bands pre-registered above |
| 5 | **§6 washout-filter transition damper** (high-pass LR damping, zero DC gain) | class-(d) actuator | M1 quench was +27.5% transient; damper targets transient area at structurally-zero steady-state bias | ~20 LOC + injection test (req B) | A/B transient area vs open-loop ramp; kill: no reduction |
| 6 | **§6 adaptive continuation step** (Δλ ∝ 1/‖Δθ‖_window) | class-(c) law | recovers part of M4's 76–125 wasted ep; adiabatic guard for LADDER | ~15 LOC; fail-safe = v3 fixed ramp | backtest on run-1 trace first (record-only run-1) |
| 7 | **§8 max-plus annulus fit** (K-element tropical expansion of per-class logits) | #342 solve item | if K≤64 reaches band-accuracy: replaces trained boundary capacity with free-decode generators; bytes enter λ_bytes law | $0 prototype on cached fields | kill: K≤64 fails band-level annulus accuracy |
| 8 | **§2 event-conditioned λ_eik + extension-violation diagnostic** | class-(d) + telemetry | conditioning-gated reinitialization; jitter-share prediction | 2 telemetry rows + gate | prediction: ext-violation ↓ ⇒ jitter-share ↓; else diagnostic only |
| 9 | **§9 deadzone census** | $0 census | adjudicates #149 (sub-pixel/dither) with a real prior | $0 | bands in §9 |
| 10 | **§4 band-residual certificate rows** | signal | per-stage analytic-explained mass; feeds §13 asymptote | reuse req-F #8 machinery | — |

DEAD/CAMPAIGN-ONLY: backstepping kernel machinery (§4, DERIVED-dead for this plant class);
Griewank revolve (§3, out of scope until a through-time inner PDE lever exists); Hajek-log τ
(already refuted, viscosity hunt §6); full Imbert–Monneau flux-limiter formalism (the σ_ij
Young-angle fit — viscosity hunt §7 — is its actionable projection and is already queued there).

---

## §12 WHAT v4 / RUN-1 CHANGES NOW (≤5) + THE ≤5 SIGNALS RUN-1 MUST ADD (req P)

**Config/DSL changes (≤5, exact surfaces):**
1. **Island-support release law**: `--island-dilation-radius-end` bound to r*(t)=0.95·σ_eff(t)
   (class-(c); fail-safe = v3's fixed 275-ep ramp). ~10 LOC in the island Lever factory.
2. **Certificate predicate**: register `island_certificate` costate SENSE row + death-alarm
   (pers > τ_k·ln5 + Δ_dec^logit; ALARM on certified-death). ~30 LOC, advisory-only.
3. **τ-indexed constants**: adaptive-ε clamps + δ_τ width + island gate margin declared as
   c(τ) with exponents; the $0 clamp-binding check runs pre-GO (already queued — this makes it
   binding on three more constants).
4. **Signed hinge arm (gated)**: wire σ_ij,dir slots into the margin-gate Lever (default =
   current unsigned behavior, byte-identical OFF) — fired ONLY if the queued per-side ρ probe
   returns |ρ| ≥ 0.3 on any side. No new flag invented: extends the existing margin-saliency
   lever surface, DSL-first.
5. **Washout transition damper (gated, injection-tested)**: high-pass LR damper at stage
   boundaries; fail-safe = v3 rewarmup ramps unchanged.

**The ≤5 signals run-1 MUST ADD so this layer is realizable (req P; all score-neutral
read-only ⇒ default-ON per the orphan rule):**
1. **Per-stage parity row** (byte-close→decode→verdict delta, d_seg AND logit units Δ_dec^logit)
   → consumers: §7 certificate, §10 σ(Δ), attribution floors.
2. **Per-class-pair per-DIRECTION one-sided margin/flip-mass histograms at stage boundaries**
   → consumers: σ_ij,dir fit, τ*_k per-side re-derivation, §13 M1.
3. **Per-island birth-death ledger with persistence, live at stage boundaries** (not post-hoc
   forensics) → consumers: certificate, birth-scheduler fit (§6), erasure-law tracking.
4. **Interface-geometry row**: measured interface width, σ_eff components, annulus conditioning
   E[(|∇m|−1)²], off-annulus gradient-mass fraction → consumers: §2/§3/§5 laws.
5. **Per-class logit-field export + per-primitive (band/clamp/comb) annulus residuals at stage
   boundaries** → consumers: max-plus fit, band-residual certificate, per-class solve ledger.

---

## §13 FAMILY-ASYMPTOTE BOUNDS FROM IMPOSSIBILITY RESULTS (req N) — each with the measurement that decides bindingness

| # | impossibility result | our form | binding? — THE MEASUREMENT |
|---|---|---|---|
| M1 | **uint8 deadzone** (quantizer information loss; no linear repair) | flips requiring through-R intensity change < 1/255 are unreachable by ANY smooth witness without dither/phase tricks | §9 census: deadzone flip mass in S-units. < 0.3× margin ⇒ not binding; > 1× ⇒ family floor term, #149 mandatory for T_3 |
| M2 | **homogenization/pinning** (sub-δ structure unrecoverable by the coarse flow — viscosity hunt §3, Dirr–Yip class) | dash structure below the (τ, ε, R-Nyquist) crossover needs the comb corrector; NO capacity/epoch budget recovers it | the queued $0 τ-crossover probe (dash-gap FP vs τ/δ_along); comb-corrector A/B. If comb-OFF floors dash residual: binding, corrector mandatory |
| M3 | **Godunov-type monotone barrier** (monotone schemes ≤ 1st-order accurate) | any GLOBAL viscosity/damping fallback pays O(ε) boundary accuracy everywhere — the fixed-ε cost | clamp-binding + jitter-share rows: if the global-ε path is ever active > x% of epochs, its O(ε) d_seg tax is measurable vs the ca-band filtered form |
| M4 | **annealing lower bound** (Hajek: global-basin guarantees need log-slow noise cooling; finite budget forfeits the guarantee) | run-1 CANNOT certify the global partition basin; wrong-basin risk is structural | seed-pair partition-Hamming A/B (the viscosity hunt §2 spec, $0-cheap at two seeds); TAIL_k warm restarts are the repair (already adopted) — measure inter-cycle Hamming to see basin hops |
| M5 | **max-plus approximation lower bounds** (basis count blows up for smooth non-max-structured targets) | the bulk cartoon must stay INR/curvelet; tropical layer only for the separatrix | §8 probe's K-vs-accuracy curve on BULK patches (expected: blow-up) vs ANNULUS (expected: small K) — confirms the two-semiring split is forced, not chosen |
| — | **NOT binding (verified):** Bardos–Lebeau–Rauch geometric control condition | full-domain observation ⇒ observability is unconstrained; the along-tangent deficit is chart capacity | already MEASURED (\|H_R\| ≥ 0.842 all-pass) — no further measurement owed |

**Asymptote composition (INFERRED, honest):** the composed-lever family's d_seg floor ≥
deadzone mass (M1) + uncorrected sub-δ dash mass (M2, →0 with comb) + selection-regime residue
below τ_end·ln5 resolution (bounded by the certificate population). Each term has a named $0
measurement above; the family asymptote estimate req-N asks for is COMPUTABLE from signals
1–5 of §12 plus the two censuses — run-1 generates every input. No term is asserted as a
number here (all currently unmeasured); the DESIGN consequence stands regardless: M1/M2 are
attacked by #149/comb (representation), not by more epochs — consistent with "the wall is
BASIS" (chain-A) from a third independent direction.

---

## Sources (external; corpus sources cited inline)

- Sokolowski–Zolésio, *Introduction to Shape Optimization*; Delfour–Zolésio, *Shapes and
  Geometries* (Hadamard structure theorem) [standard texts].
- McEneaney, *A Curse-of-Dimensionality-Free Numerical Method for Solution of Certain HJB PDEs*,
  [SICON 10.1137/040610830](https://epubs.siam.org/doi/10.1137/040610830); convergence rate
  [pdf](http://maeresearch.ucsd.edu/mceneaney/pubs/codconvpureq_siam.pdf); Riccati-contraction
  analysis [arXiv:1301.4777](https://arxiv.org/pdf/1301.4777).
- Blank–Garcke et al., *Sharp interface limit for a phase field model in structural
  optimization*, [SICON](https://epubs.siam.org/doi/10.1137/140989066) /
  [arXiv:1409.7586](https://arxiv.org/pdf/1409.7586) (Γ-convergence incl. first variation).
- Koga–Krstic, *Backstepping Control of the One-Phase Stefan Problem*,
  [arXiv:1607.04345](https://arxiv.org/pdf/1607.04345);
  [arXiv:1703.05814](https://arxiv.org/pdf/1703.05814) (output feedback / estimation);
  high-order moving boundary [arXiv:2510.06571](https://arxiv.org/pdf/2510.06571).
- Wang–Abed 1995 washout-filter bifurcation control lineage; modern treatments:
  [MDPI Sensors 22:9334](https://www.mdpi.com/1424-8220/22/23/9334),
  [PMC3998009](https://pmc.ncbi.nlm.nih.gov/articles/PMC3998009/).
- Conley index continuation: Dey–Mrozek et al., *Persistence of the Conley Index in
  Combinatorial Dynamical Systems*, [arXiv:2003.05579](https://arxiv.org/abs/2003.05579);
  Mischaikow, Banach Center notes [pdf](http://matwbn.icm.edu.pl/ksiazki/bcp/bcp47/bcp4711.pdf).
- Han–Xu–Prince, *A topology preserving level set method for geometric deformable models*,
  [IEEE PAMI 2003](https://ieeexplore.ieee.org/document/1201824/) /
  [pdf](https://iacl.ece.jhu.edu/~chenyang/research/pami_han_xu_prince.pdf).
- Adalsteinsson–Sethian narrow band + velocity extension; Sussman–Smereka–Osher
  reinitialization [standard level-set references].
- Sister in-house reports composed above (NOT duplicated): viscosity_theory_alignment_hunt
  (Barles–Souganidis · homogenization EUREKA · junctions/σ_ij · weak KAM · selection/WD ·
  Hajek refutation) · negatives_scale_validity_review (viscosity REOPEN · UniWARD asymmetry ·
  TAIL_k) · chain-A terminal verdict · DRAFT v3.

**Triality note:** [no-triality] orchestration state — laws here are SPECS; DSL/equation/DAG
landings occur at P7 integration per requirement G (no flags invented; every lever surface
named above extends an existing Lever factory). Pointer contest-CPU **0.19110 UNMOVED** —
this report is MEANS.
