---
council_tier: T3
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Mallat, Tishby, Hinton, MacKay, Ballé, Hotz, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "You are designing loss surgery before ep50 tells you whether the patient is sick. The witness-alone trajectory has ONE point past init. If ep50-100 descend steeply, every hour spent on focal/active-contour is an hour not spent on the byte-close path. Scope, calibrate, HOLD."
  - member: Hotz
    verbatim: "The active-contour energy-as-loss is beautiful and I don't trust beautiful. It's a new loss landscape on a live vehicle mid-campaign. Focal is 5 lines and composable. Ship the 5 lines, bank the beauty for run 3."
council_assumption_adversary_verdict:
  - assumption: "The seg loss (CE-family) is the binding constraint on witness-alone island formation."
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "ep50/75 slope not yet measured. If the slope is steep, TIME was the constraint, not the loss. The calibration probe measures the gradient-mass fraction on island pixels NOW ($0, decisive for whether amplification is even needed)."
  - assumption: "Amplifying gradient onto islands is net-positive for TOTAL d_seg."
    classification: CARGO-CULTED
    rationale: "The #274 margin/spike-reweight lever measured an AA-washout caveat, and the lane-edge lever's own decision metric was TOTAL d_seg because lane is only ~19% of flips vs class-0's 50% (DAG). Focal-γ up-weights ALL low-margin pixels including bulk boundaries — that may be fine (bulk boundaries are also scored) but the claim 'more island gradient ⟹ lower total d_seg' is unproven; γ must be calibrated against TOTAL d_seg, not island recall."
# Catalog #300 v2-frontmatter backfill 2026-08-25: council_decisions_recorded transcribed
# VERBATIM from this memo's own section "Debate resolution → the program
# (PROCEED_WITH_REVISIONS)" items 1-4. Frontmatter-only addition per the CLAUDE.md
# "Council hierarchy" backward-compatibility clause (NO body mutation).
council_decisions_recorded:
  - "CALIBRATE NOW, $0 (the revision — measurement before surgery): on the live run's ep25/ep50 EMA checkpoint MEASURE (a) island-pixel share of the base-loss gradient under the CURRENT loss, (b) the same under focal γ∈{0.5,1,2,3}, (c) d_seg[islands] vs d_seg[bulk] decomposition and whether island gradient is already large-but-ineffective (→ basis binds, not loss). Output: γ* by Shannon's equalization + the loss-vs-basis verdict. CPU-side, memory-light, NO touch of the live run."
  - "BUILD focal-γ + boundary-distance as default-OFF flags (--seg-focal-gamma 0 = byte-identical; --boundary-distance-weight 0), tested + byte-identity-proven, READY but NOT deployed. Pre-registered fire criterion: ep50→100 witness-alone slope FLATTENS (|Δd_seg| < 0.02 per 25ep window with islands still >50% of residual). If the slope is steep — HOLD (Contrarian)."
  - "Active-contour energy-as-loss = run-3/θ* design item (the unification) under the #218/#78 lineage, NOT a hot patch."
  - "KD-from-teacher on the island band = banked third rung (Hinton), after focal, before the capacity A/B (#299 stays shelved behind BOTH)."
---

# GRAND COUNCIL SYMPOSIUM — the loss geometry of the level-set witness
**Convened 2026-07-05 (operator: "conduct research regarding optimal and deep math against our
level set and measure and calibrate and analyze and debate and propose updates to triality").**
Live context: fixed run pid 39999 at ~ep50, witness-alone d_seg 0.162@ep25, seed annealing →0 by
ep300, tau onset ep300. **Pointer 0.19110 UNMOVED — this symposium is MEANS.**

## The question
CE fits the per-pixel DISTRIBUTION; the contest scores the argmax DECISION on a codim-1 boundary
plus two near-measure-zero island classes, through R, with tau=MCF waiting to erode anything
sub-critical. What is the OPTIMAL loss geometry for this vehicle — and what does it change in the
triality (DAG · DSL · equations)?

## The measured inventory (what we already are — no re-invention)
Margin/hinge (island_amplify #141-lineage) ✓ · persistence/Betti + clDice (topology) ✓ ·
logit-adjust (partial) ✓ · soft-Dice (minor) ✓ · **focal ✗ · boundary-distance ✗ ·
active-contour/level-set energy-as-loss ✗.** The witness-alone island loss (#300) just gave all of
these the correct (deploy-surface) target. We are IN the right family; the question is the three
missing members and their composition.

## The voices (each lens distinct, deep-math)

**Shannon (LEAD) — the gradient-information budget.** Total gradient mass is a budget; CE
allocates it ∝ pixel count × (1−p_y). The islands are ~2% of pixels; once the bulk is
half-learned, the islands' share collapses. Focal-γ reallocates the budget multiplicatively:
share(islands) ∝ Σ_islands (1−p)^γ / Σ_all (1−p)^γ — a CONTROLLABLE concentration knob. But the
budget is conserved: what the islands gain, the bulk boundary loses. γ* is where marginal
d_seg-per-gradient equalizes across regions — a waterfilling condition, MEASURABLE on the ep25
checkpoint ($0). Calibrate, don't guess.

**Daubechies (CO-LEAD) — loss and basis are dual.** The basis determines WHAT the network can
represent cheaply; the loss determines WHERE gradient flows. A focal loss cannot conjure
along-tangent bandwidth the basis lacks (n-dir-freqs 2, the #277 finding); a directional basis
cannot force gradient onto pixels the loss ignores. If ep50 flattens, suspect BOTH axes — the
calibration probe must decompose which binds (gradient-starved vs representation-starved: if
island gradient is already large but d_seg[islands] doesn't move, it's the basis, not the loss).

**Mallat — the scattering view of thin structure.** The lane at 6px with dashes at ~25 cyc/unit is
a high-frequency ORIENTED texture. Per-pixel losses (CE, focal) see it as isolated hard pixels;
what forms it reliably is a loss on an ORIENTED MULTISCALE transform of the mask — which is what
clDice/persistence approximate topologically. The boundary-distance loss is the cheap geometric
sibling: it scores the EDGE LOCATION (distance transform), converting "flip this pixel" into "move
this contour," which is the actual degree of freedom our SDF head owns.

**Chan-Vese lineage (presented by Mallat + Daubechies jointly) — energy≡loss unification.** The
witness IS a level-set flow; Mumford-Shah/Chan-Vese/active-contour losses make the TRAINING
OBJECTIVE the same energy the flow minimizes: E = region-fit + ν·length + eikonal. We already
train with eikonal + length as REGULARIZERS — the active-contour move is to make the DATA term a
region energy too (inside/outside statistics against the SegNet argmax target), instead of
per-pixel CE. Then curriculum = annealing ONE energy (τ its temperature), and tau's MCF is the
gradient flow OF THE LOSS — objective and vehicle become the same object. This is the deep-math
optimum and the largest change; it belongs to the NEXT vehicle design (run 3+/capstone θ*), not a
hot patch (Hotz dissent concurs).

**Tishby — where in the anneal to intervene.** The IB view: CE-with-seed was compressing the
wrong sufficient statistic (the crutch); witness-alone restores the right one. Amplification (γ)
raises the effective β on island bits — do it DURING CE (the encoding phase). Do NOT stack a new
loss mid-tau: changing the objective during annealing invalidates the temperature schedule.
Window: any focal/boundary lever must engage at CE (≤ep300) or wait for run 3.

**Yousfi/Fridrich — the detector's side.** UNIWARD says put signal exactly where the detector is
most sensitive — the small-margin band. Focal(1−p)^γ IS a soft margin-band selector (p ≈ margin);
it's the same object as our #141 margin-saliency, applied to the BASE loss instead of a bolt-on
weight. Elegance: one mechanism, previously two names. But calibrate against TOTAL d_seg — the
detector also reads the 50%-of-flips class-0 boundary (Assumption-Adversary's point).

**Rudin — interpretability of the knob.** γ has a closed-form meaning: the gradient weight ratio
between a p=0.5 pixel and a p=0.9 pixel is ((0.5)/(0.1))^γ = 5^γ. γ=2 ⟹ 25×. Print the measured
island-gradient share at the chosen γ in the run log (observability); no silent reweighting.

**Hinton — don't forget the teacher.** If islands stay unformed by ep100 even with focal, the
cheapest capacity-independent fix is distilling the island logits from the #205 mod-32 teacher
(which formed them) — KD on the island band only. Bank as the third rung.

**MacKay — the MDL check.** All of this is train-time; ZERO archive bytes. Any loss reshaping
that lowers d_seg at fixed rate is pure score. The only cost is train time and risk (Contrarian).

## Debate resolution → the program (PROCEED_WITH_REVISIONS)
1. **CALIBRATE NOW, $0 (the revision — measurement before surgery):** on the live run's ep25/ep50
   EMA checkpoint, MEASURE (a) the island-pixel share of the base-loss gradient under the CURRENT
   loss, (b) under focal γ∈{0.5,1,2,3}, (c) d_seg[islands] vs d_seg[bulk] decomposition and
   whether island gradient is already large-but-ineffective (→ basis binds, not loss). Output:
   γ* by Shannon's equalization + the loss-vs-basis verdict. Runs CPU-side, memory-light,
   NO touch of the live run.
2. **BUILD focal-γ + boundary-distance as default-OFF flags** (`--seg-focal-gamma 0` = byte-
   identical; `--boundary-distance-weight 0`), tested + byte-identity-proven, READY but NOT
   deployed. Fire criterion (pre-registered): ep50→100 witness-alone slope FLATTENS
   (|Δd_seg| < 0.02 per 25ep window with islands still >50% of residual). If slope is steep —
   HOLD (Contrarian), the run is converging on time alone.
3. **Active-contour energy-as-loss = run-3/θ* design item** (the unification), NOT a hot patch.
   Design doc under #218/#78 lineage.
4. **KD-from-teacher on the island band = banked third rung** (Hinton), after focal, before
   capacity A/B (#299 stays shelved behind BOTH).

## Proposed triality updates (the operator's ask — all three legs)
- **equations (append via tools/register_lever_laws):**
  `focal_gradient_concentration_v1`: share_isl(γ) = Σ_isl(1−p)^γ / Σ_all(1−p)^γ, monotone in γ;
  weight-ratio law (p₁/p₂ margin pair) = ((1−p₁)/(1−p₂))^γ — Rudin's 5^γ readback. Anchor =
  the calibration probe's measured shares (pending; register with FORMALIZATION anchor on land).
  `levelset_energy_loss_equivalence_v1` (DESIGN-STAGE): training loss E = ∫ region-fit +
  ν|∇H(φ)| + λ(|∇φ|−1)² makes tau-flow the gradient flow of the LOSS — the objective≡vehicle
  identity for the run-3 vehicle.
- **DSL (gauge.py, APPEND-ONLY):** new `SegLossGauge` component: BASELINE (current margin+topo
  stack, byte-identical) · FOCAL (adds --seg-focal-gamma γ*) · BOUNDARY_DIST (adds
  --boundary-distance-weight) · ACTIVE_CONTOUR (DESIGN-STAGE, run-3). Trainer flags grep-verified
  at build time (never-invent).
- **DAG:** FEED-05e records this symposium + the calibration numbers when they land; the
  pre-registered fire criterion goes in the FEED so the decision is mechanical, not vibes.

## Self-reflection (Catalog #363, Round 2)
Round-1 assumptions carrying `ASSUMED_AWAITING_VERIFICATION`: (i) "focal is implementable in the
MLX seg path without breaking the realized-through-R composition" — the build verifies; (ii) "the
calibration probe can decompose loss-starved vs basis-starved from a checkpoint alone" — if not
cleanly separable, the probe reports `partially_confounded` honestly rather than forcing a verdict.
Round-3: no lever fires from an unmeasured calibration; γ* comes from the probe or the lever waits.

<!-- STORES CONSULTED (2026-08-25 backfill append; this 2026-07-05 council memo predates the recall-evidence discipline #713): MEMORY.md index · .omx/research corrections index (au1) · task ledger #1274/#1275 (frontmatter + observability backfills). Consulted for the backfill-append pass only; the deliberation body is historical, append-only, unchanged. -->
