# P1 SEAT S2 — LEVER COMPOSITION / MEASURED-EV (v7.5.2)

**Seat:** S2 (Fridrich/Yousfi lens). **Charter:** OPEN Q1 (lever composition) primary; Q3 (collapse-fix
as compose-precondition) + Q5 (#341 terminal solve composition) secondary. INDEPENDENT position — no
cross-read of sibling seats. Cites `docs/operating_manual_craft_handoff.md` (label every claim by how
it was obtained; spend depth on the authority path + any default that changed + any word "identical").
Works from `DELTA_GROUNDING_20260709.md` + the code-level compose guards I re-derived from
`src/tac/witness_dsl/curriculum_dsl.py`. Pointer **0.19110 UNMOVED**; everything here is MEANS.

---

## 0. THE ONE-SENTENCE POSITION

The fireable levers are NOT one undifferentiated menu subject to §9 one-per-increment — they split into
**Class A STRUCTURAL PRIORS** (0-byte, rule-118, ep0-structural, add ZERO gradient-share ⇒ CANNOT
term-dominate) and **Class B LOSS-TERM FORCES** (add gradient-share ⇒ §9 binds); the Class-A basis
cluster {directional self-orient + #121 taper + AA-ipe} composes ON together in the FIRST trunk
(basis-BEFORE-capacity, synergistic, and un-isolable-except-by-fresh-run), while the Class-B forces
{horizon-margin, chroma, the 3 P0 forces} stagger one-per-increment with the collapse-fix (#378 amber)
as a LAUNCH-BLOCKING precondition underneath both.

---

## 1. THE DECISIVE DISTINCTION — TWO LEVER CLASSES, TWO COMPOSITION RULES (DERIVED from the compose guards)

I re-derived this from the DSL factories, not from a memo. It is the load-bearing move of this position.

**§9 (SPEC_v75, verbatim in DELTA §B): "the P0 forces default OFF, activate ONE per crucible increment…
turning all 3 on = SPEC VIOLATION, confounds attribution + risks term_domination."** The rationale is
TWO things: (a) attribution confound, (b) term_domination. Term_domination is a LOSS-SHARE property —
the `term_domination` alarm fires when one regularizer term exceeds ~40% of total loss
(`curriculum_dsl.py:2934` cold-start-0.1-"far under the 40% term_domination alarm"). **A lever that adds
NO loss term has NO loss share and CANNOT term-dominate.** So §9's term_domination leg is VACUOUS for
levers that are pure representation/render changes. Only its attribution leg remains — and that has a
cheaper resolution than one-per-increment (§4 below).

| Class | levers | adds a loss term? | gradient-share? | term-dominate? | ep0-structural? | §9 binds? |
|---|---|---|---|---|---|---|
| **A — structural prior** | directional self-orient basis (−48%), #121 taper (73%), StepNative activation (31.6%), AA coverage render | NO (reweights basis / changes render or activation) | 0 | **impossible** | YES (F2-resume REFUSES adding them → fresh-run only) | **NO** (out of §9 scope by construction) |
| **B — loss-term force** | #360 temporal-screw (P0 F1), the 2 other P0 forces, #169 horizon-margin hinge, #276 chroma-match, Chan-Vese area-constraint | YES | >0 | YES | NO (turn-on-able mid-run) | **YES** |

MEASURED anchors for the class split, from `curriculum_dsl.py`:
- `DsegAwareTaper` (#121): "BYTE-NEUTRAL (adds ZERO trainable params)… STRUCTURAL (active from ep0…
  changes the input feats the `in_proj` is trained on)… an end-of-run resume that ADDS/CHANGES it is
  REFUSED by the F2 resume-divergence guard" (L1923–1941). ⇒ Class A, un-isolable except fresh.
- `DirectionalBasisRebalance`: emits `--self-orient/--freq-across/--freq-along`; "window=0 = basis-config
  change, no epoch budget of its own" (L2721,2745). ⇒ Class A structural, ep0.
- `AACoverageRender`: "the AA render is a DETERMINISTIC decode-time op… archive weights+codes are
  UNCHANGED — AA moves d_seg WITHOUT the rate term" (L3243). ⇒ Class A.
- `SegTemporalScrew` (#360): "`weight` cold-start 0.1 (…≈0.1% of total loss 6.7 ⇒ far under the 40%
  term_domination alarm); ramp at STAGE BOUNDARIES ONLY toward gradient-share≈0.44" (L2934). ⇒ Class B,
  explicitly loss-share-managed = the §9 subject.

**Consequence:** the "all-top-on vs minimal-composed" tension in Q1 is a FALSE binary once you class the
levers. The answer is a MIDDLE PATH that is principled, not split-the-difference: **all Class-A basis
priors ON together (they must be — see §2); Class-B forces one-per-increment (§9 respected — see §3).**

---

## 2. THE CLASS-A BASIS CLUSTER — COMPOSE ON TOGETHER IN THE TRUNK (basis-BEFORE-capacity)

Three independent reasons the basis cluster fires together in the FIRST launch, not one-per-increment:

**(i) SYNERGY — they are three views of ONE object (MEASURED, §OPERATOR PRIORITY).** basis-match is
PRIOR to capacity: "Capacity ALONE on an isotropic basis does NOTHING / HURTS (+6%); once the basis is
all-class-directional, modest capacity then pays (n96 −64% combined)." #121 taper REWEIGHTS the *same*
Fourier/curvelet columns the self-orient front-end produces (`DsegAwareTaper` docstring: "the witness's
OWN Fourier basis"). Firing taper on an ISOTROPIC basis is the +6% failure mode — it must ride the
directional basis. They are not additive knobs; they are the basis-match object.

**(ii) UN-ISOLABLE EXCEPT BY FRESH RUN (MEASURED F2 guard).** Both are ep0-structural and the F2
resume-divergence guard REFUSES adding them to a resume. So "isolate directional, then next increment
add taper" is NOT a warm-start increment — it is a SECOND full fresh run. One-per-increment for Class A
means N fresh runs for N levers. That is the opposite of efficient, and it wastes the highest-rel-sig
levers (73% + the −48% #1) on serial isolation when their JOINT effect is the whole point.

**(iii) BASIS-BEFORE-CAPACITY says do these FIRST regardless.** They are the RANK-1 (73%) and #1-measured
(−48%) levers. The trunk's entire job is to establish the correct basis so every downstream Class-B force
operates on a boundary-aligned representation.

**MEASURED interaction constraint I must honor (the AA fork).** `AACoverageRender` supersample (the
AUTHORITY, oracle-R floor **0.00091 @384**) **fail-closes against --self-orient AND --dseg-aware-taper**
(`curriculum_dsl.py:3255,3280` + DELTA L-6). BUT `mode="ipe"` (mip-NeRF cone attenuation) "attenuates the
SAME base curvelet columns AFTER, so it composes" with both. **RESOLUTION: the trunk uses AA in ipe
mode** (compose-compatible with the basis cluster); the supersample AUTHORITY becomes a separate fresh
A/B arm (§5, ladder rung 4) — a genuine REGIME fork (supersample-fidelity-without-basis vs
basis-cluster-with-ipe), not an additive lever.

**Attribution WITHIN the cluster (honest — firing 3 together DOES confound their mutual attribution):**
resolved by structural-cluster A/B, not individual isolation:
- directional −48% is SEPARATELY anchored and is already "the run's OWN A/B" (F-3). MEASURED.
- taper's marginal = {directional+taper} vs {directional-only} — ONE extra fresh run, the RANK-1 lever,
  worth it. Its verdict scope is INSTANCE (the +18% NO-GO was one under-converged run; converged flips to
  −8%~0.03 ESTIMATED) — this A/B is exactly the owed converged re-validation.
- ipe-AA's marginal rides in the cluster; its through-training ΔS is ASSUMED_AWAITING_VERIFICATION
  (oracle-R lane-recall +0.38 is MEASURED, not through-training).

**StepNative is the ONE Class-A lever I HOLD OUT of the first trunk.** Rationale (honest, against my own
"compose Class-A together" rule): it changes the ACTIVATION NONLINEARITY globally (not just the basis),
so it interacts with every other lever's gradient landscape — highest confound risk of the Class-A set;
it is NEVER-FIRED (duty-to-measure #2), lowest rel-sig of the top-3 (31.6%, INSTANCE scope, "modest
−4.5%"); and there is a MEASURED collapse hazard adjacent to it — fixed-β hosc DIVERGES (CLAUDE.md
capstone-trainer note); it must run annealed_hosc β1→8, which is itself a schedule that wants its own
clean read. ⇒ StepNative = ladder rung 2 (fresh A/B after the basis cluster validates).

---

## 3. THE CLASS-B FORCES — §9 ONE-PER-INCREMENT, EVENT-SELF-SEQUENCED

- **Chan-Vese area-constraint + birth-completion + per-class ramp** — ALREADY ON in crucible_v7 (the
  run-1 Road-floor CURE, DERIVED-LIVE λ_lane 683.8 / λ_movable 322.6, R-3). NOT a P0 force; the baseline
  the trunk inherits. STAYS ON. This is the term that returns ~96% of the Road+Undriv deficit run-1
  floored — the reason v7.5.2 ≠ run-1.
- **#360 temporal-screw (P0 FORCE 1)** — the ONE P0 force armed this increment. It is EVENT-governed
  (fires on `annulus_plateau` FORMED-boundary), cold-start w 0.1, ξ=`ground_gt` stop-grad ⇒ PURE seg
  regularizer, ZERO pose coupling (P-2/L68). Event-gating makes it SELF-SEQUENCING — it will not fire
  until the boundary has formed, i.e. after the basis cluster + counter-force have done their work, so
  its attribution window is clean by construction. The **other 2 P0 forces stay OFF** (§9). ✓.
- **#169 horizon-margin satisficing hinge (RANK-2, 43.8%)** — a loss term (gradient-share) BUT one-sided
  SATISFICING: it zeroes once the margin target is met, so its term_domination risk is bounded and it
  "sequences ≥ l7" (SPEC §9). It is NOT one of the 3 P0 forces, so §9's one-per-increment-for-the-forces
  does not literally bind it — but it DOES add gradient-share, so I stagger it as **ladder rung 1**
  (fire at a stage boundary AFTER the basis cluster, dense verdict window around activation). EXIT
  CONDITION (MEASURED risk, L-2): surviving flips must shift to HIGHER GT margin — else it is chasing the
  IRREDUCIBLE frozen-SegNet label-noise `<lo` band → terminal-finding, kill.
- **#276 chroma-match (UNMEASURED add-back)** — loss term, BUT ORTHOGONAL-BY-CONSTRUCTION to every luma
  lever: chroma := rgb − BT.601-luma is LUMA-INVARIANT (DELTA L-5). Orthogonality means it CAN fire in a
  PARALLEL arm without confounding the luma cluster's attribution (per-channel telemetry: luma-margin-
  energy 78.8% vs chroma 21.2% separates them). BUT (operating-manual "identical/equivalent is a claim"):
  its ΔS is ablation-MEASURED (constant-luma FLIPS 7.54% Lane→Road) NOT add-back-MEASURED — **ablation ≠
  add-back**. ⇒ ladder rung 3 (parallel arm, must MEASURE add-back ΔS, never assume it).

---

## 4. THE ATTRIBUTION SCHEME (the charter's explicit ask) — 3 MECHANISMS, matched to lever class

The confound is: N levers at ep0 ⇒ ΔS un-attributable. §9's answer (one fresh run per lever) is correct
but O(N) fresh runs. Here is the taxonomy that lets multiple levers fire per run WITHOUT confounding:

1. **STRUCTURAL-CLUSTER A/B (for Class-A, un-isolable-mid-run).** A/B the CLUSTER against baseline, not
   the individuals; recover per-lever marginals from a small SET of fresh runs whose diffs isolate one
   lever each: {baseline} / {directional} / {directional+taper} / {directional+taper+ipe-AA}. 4 runs →
   3 clean marginals (directional, taper, ipe-AA) + the joint. directional's marginal is already banked
   (−48%), so effectively 3 NEW runs. This is the Class-A attribution answer.
2. **STAGGERED-ACTIVATION VERDICT-WINDOW (for Class-B, turn-on-able mid-run).** Fire each loss-term force
   at a DISTINCT event/stage boundary with a DENSE verdict cadence bracketing the boundary; the per-class
   d_seg delta ACROSS the activation IS the per-lever attribution — inside ONE run. Requires the
   handoff-legibility telemetry (S-3, default-ON) so a transition can't hide between sparse verdict rows,
   AND the R-4 train-verdict-decoupling guard so a rise is not misread (S3 owns that control law). This
   is why temporal-screw (event-gated) and horizon-margin (stage-boundary) can share a run.
3. **ORTHOGONALITY-BY-CONSTRUCTION (channel isolation, simultaneous).** Provably-disjoint-subspace levers
   fire together with per-channel telemetry separating them: chroma ⊥ luma (luma-invariance);
   pose ⊥ d_seg (∂d_seg/∂ξ≡0 structural, P-2). No temporal staggering needed — the subspaces don't mix.

**Net attribution budget:** the first trunk carries the Class-A cluster (attributed by mechanism 1 across
the ladder) + counter-force (baseline) + temporal-screw (mechanism 2, event-self-sequenced) + pose
(mechanism 3). That is ≤1 confound to untangle (the cluster), and the ladder untangles it.

---

## 5. Q3 — COLLAPSE-FIX (#378 amber): ON BY DEFAULT, LAUNCH-BLOCKING PRECONDITION (not a parallel arm)

The basis cluster's whole job is to SHARPEN the boundary → richer normals → precisely the regime where
run-1's diagnosed collapse bit (no grad-clip; sqrt-pose-eps grad blowup 5/√(10·pose+1e-8)→5e4 for easy
pairs at batch=1; w_seg=100 ⇒ 100× seg-LR; DELTA L-7). **Composing the basis cluster WITHOUT amber
re-arms the exact collapse.** So amber is not optional-alongside — it is the PRECONDITION that makes the
JOINT 7-dim descent (X-1) converge. It is default-preserving/byte-identical when off (so it does not
corrupt the incumbent), and its own validation is SELF-LOUD: the first v7.5.2 run either converges
(collapse cured) or collapses (loud, not silent) — the binary IS the A/B. Position: **amber ON,
launch-blocking; the trunk does not launch without it.** (I flag for S3/S5: the fix is UN-A/B'd — so the
FIRST run is simultaneously the amber validation AND the cluster validation; if it collapses we cannot
tell amber-insufficient from cluster-too-aggressive. Mitigation: keep the amber levers individually
loud-instrumented — gnorm alarm, spike-guard, pose-eps-floor eps_floor(C)=(5/C)² telemetry — so a
collapse names its cause.)

---

## 6. Q5 (secondary) — #341 TERMINAL HEAD SOLVE composition

- **Fire IFF re-verified.** L-8: head chart near-quadratic CONFIRMED (LM ρ 0.847/0.868 MEASURED) but
  the solve fires ONLY if ρ re-verifies ∈~[0.8,1.2] on the CURRENT terminal ckpt, ALL 600 pairs, exact
  tau-stage loss, --fused-r-kernel bit-identity, verdict through R + frozen CPU SegNet. K=8 subset
  OVERFITS +5.1% n600 (N-3, MEASURED NO-GO) ⇒ **full-P (P=600) ONLY**, ~3h GPU (NOT $0, governed).
- **Composition = solve-don't-train REPLACEMENT of the terminal fine-tune leg, sequenced.** It fires at
  the TERMINAL tau-best EMA ckpt — AFTER the trunk + ladder converge. Sequence: converge trunk → head
  solve (if ρ ok) → THEN pose-descend (D.9 terminal pose-finish). **Solve-head-THEN-pose, NOT joint:** the
  ~791-param affine head solve (out_sdf/out_tex/palette; FiLM EXCLUDED non-affine) assumes a FROZEN basin
  (GN/CG in the quadratic bowl); pose-finish then rides the solved SDF/tex output. Joint would break the
  frozen-basin assumption the quadratic solve depends on.
- Does it compose with pose-finish or replace it? It REPLACES the terminal SEG fine-tune; pose-finish is
  ORTHOGONAL (pose ⊥ d_seg) so it runs AFTER the head solve, unaffected. Both terminal, sequenced.

---

## 7. THE A/B LADDER (what stays OFF the trunk, drained by increment)

Ranked by (rel-sig × readiness ÷ confound-risk):

1. **#169 horizon-margin hinge (43.8%)** — increment 1, staggered activation at a post-cluster stage
   boundary (mechanism 2, in-run). EXIT: surviving flips shift to higher GT margin or KILL (label-noise).
2. **StepNative activation (31.6%)** — increment 2, FRESH A/B (structural nonlinearity, annealed_hosc
   β1→8 NOT fixed-β; mechanism 1). Highest Class-A confound, so isolated.
3. **#276 chroma-match** — parallel arm (mechanism 3, orthogonal). MUST measure add-back ΔS (ablation ≠
   add-back). Cheap to fire (orthogonal), high info (UNMEASURED = pure duty-to-measure).
4. **AA supersample AUTHORITY vs ipe** — fresh A/B REGIME fork: supersample (oracle floor 0.00091, but
   fail-closes → DROPS directional+taper) vs the basis-cluster+ipe trunk. Measures whether AA fidelity
   beats basis reallocation. Mutually exclusive with the trunk basis levers — a fork, not an add.
5. **#341 terminal head solve** — terminal, gated on ρ re-verify (§6), full-P, ~3h governed.
6. **The 2 other P0 forces** — OFF, §9 one-per-later-increment each.
7. **duty-to-measure tail** (FEED-relsig): d18-k90 truncate (2.4% rate), mod32-neutrality (1.2%), #274
   seg-down-weight — rate/formulation, not the d_seg blocker; drain last (near-goal any real byte cut is
   pure S, but they don't move the seg wall).

**Do NOT re-open blind (MEASURED NEGATIVES, DELTA §B):** N-1 OT head offsets (both arms HURT; next form =
flip-weighted target masses, not raw-GT-frequency mass-match); N-2 lane-ξ ego-transport (ENLARGES the
stream — ego-freeze does not transfer to a chart that already absorbed the ego DOF); N-3 #341 K=8 subset.

---

## 8. CONFIG-SHAPED BLOCK — the v7.5.2 ON-SET + STAGING + A/B LADDER

```
# ============ v7.5.2 FIRST-TRUNK LEVER SET (S2 position) ============
# PRECONDITION (launch-blocking, byte-identical-when-off; §5):
amber_collapse_fix        = ON   # #378 --stability-preset amber : grad-clip + pose-eps-floor
                                 #   eps_floor(C)=(5/C)^2 + per-param grad-normalize + stage LR/w_seg guard.
                                 #   ENABLES the joint 7-dim descent the basis cluster demands. Self-loud A/B.

# CLASS-A STRUCTURAL PRIORS — compose ON together, fresh, ep0 (§2; basis-BEFORE-capacity):
directional_self_orient   = ON   # DirectionalBasisRebalance(regime="lane_carried")
                                 #   --self-orient --n-dir-freqs 4 --freq-across 32 --freq-along 26
                                 #   (fixes the MEASURED 3.2x along-tangent deficit; mod32cap ships along=8 BACKWARDS)
                                 #   COMPILE-COUPLED to --lane-render-band (fail-loud if absent).
dseg_aware_taper          = ON   # DsegAwareTaper(strength=1.0, scale=0.0[AUTO=median|margin|], floor=0.05)
                                 #   reweights the SAME directional basis toward the margin annulus. RANK-1 (73%).
aa_coverage_render        = ON   # AACoverageRender(mode="ipe", grid_h=384, grid_w=512)   <-- ipe, NOT supersample
                                 #   (supersample fail-closes vs self-orient/taper; ipe composes). ~0-rate.
step_native_activation    = OFF  # HOLD -> ladder rung 2 (global nonlinearity, highest Class-A confound)

# CLASS-B LOSS-TERM FORCES — §9 one-per-increment (§3):
chan_vese_counterforce    = ON   # ALREADY crucible_v7 baseline: area-constraint + birth-completion +
                                 #   per-class ramp (lambda_lane 683.8 / lambda_movable 322.6 DERIVED-LIVE).
                                 #   The run-1 Road-floor CURE. Not a P0 force.
seg_temporal_screw        = ON   # #360 P0 FORCE 1 : EVENT-governed (annulus_plateau FORMED), cold-start w 0.1,
                                 #   xi_source="ground_gt" stop-grad (ZERO pose coupling). THE ONE P0 force this incr.
p0_force_2, p0_force_3     = OFF  # §9 : one-per-increment. Stay off.
horizon_margin_hinge      = OFF  # -> ladder rung 1 (satisficing, staggered stage-boundary activation)
chroma_boundary_match     = OFF  # -> ladder rung 3 (orthogonal parallel arm; MEASURE add-back, not ablation)

# POSE (orthogonal; S4 owns detail):
w_pose                    = ON (terminal-gated)   # D.9 pose-finish fires at muon-cap/_muon_gate.fired;
                                                  # ships R1 dxi (0.127, 7.2KB) at export. pose ⊥ d_seg (P-2).

# TERMINAL SOLVE (Q5; after trunk+ladder converge):
head_gn_cg_solve          = GATED # #341 full-P(600) ONLY, fire IFF LM rho re-verify in ~[0.8,1.2] on CURRENT
                                  # terminal EMA ckpt, exact tau-loss, --fused-r-kernel bit-identity, ~3h governed.
                                  # SEQUENCE: converge -> head-solve -> THEN pose-descend (solve-head-then-pose,
                                  # NOT joint; solve assumes frozen basin).

# ============ ATTRIBUTION SCHEME (§4) ============
#  Class-A cluster  -> STRUCTURAL-CLUSTER A/B (mechanism 1): {base}/{dir}/{dir+taper}/{dir+taper+ipe} fresh runs
#  Class-B forces   -> STAGGERED-ACTIVATION verdict-window (mechanism 2): distinct stage/event boundaries,
#                      dense verdict cadence bracketing each activation; needs S-3 legibility + R-4 decoupling guard
#  chroma / pose    -> ORTHOGONALITY-BY-CONSTRUCTION (mechanism 3): simultaneous, per-channel telemetry separates

# ============ A/B LADDER (drain order, §7) ============
#  1. horizon_margin_hinge (43.8%)  in-run staggered  [EXIT: flips->higher GT margin else KILL label-noise]
#  2. step_native (31.6%)           fresh, annealed_hosc beta1->8
#  3. chroma_boundary_match         parallel orthogonal, MEASURE add-back
#  4. AA supersample-authority      fresh REGIME FORK (drops dir+taper; oracle floor 0.00091 vs basis-cluster+ipe)
#  5. head_gn_cg_solve              terminal, rho-gated, full-P, ~3h governed
#  6. p0_force_2 / p0_force_3       OFF, one-per-later-increment (§9)
#  7. duty-to-measure tail          d18-k90(2.4%) / mod32-neutrality(1.2%) / #274 seg-down-weight (rate, last)
#  DO-NOT-REOPEN-BLIND: N-1 OT offsets / N-2 lane-xi transport / N-3 #341 K=8 subset
```

---

## 9. EPISTEMIC LABELS + FLAGS FOR SYNTHESIS/RED-TEAM

- **DERIVED (mine, load-bearing, attack it):** the Class-A/Class-B split and the claim §9 is
  term_domination-vacuous for Class A. This is MY interpretation of §9's rationale, re-derived from the
  compose-guard code — NOT an established SPEC clause. If the CHIEF-DESIGNER/red-team reject it, the
  fallback is strict §9 one-per-increment for ALL levers (O(N) fresh runs, much slower). I believe it
  holds because the term_domination alarm is defined on loss-share and Class-A adds none — but I flag it
  as the single most-attackable claim here.
- **MEASURED (re-derived from code/DELTA):** the AA-supersample ⊥ self-orient/taper fail-close (→ ipe in
  trunk); taper/directional ep0-structural + F2-resume-refuse; §OPERATOR-PRIORITY basis-before-capacity
  (+6% isotropic-capacity, −64% n96 combined); the directional −48%; the K=8 +5.1% overfit; N-1/N-2 signs.
- **ESTIMATED:** taper converged ΔS ~0.03; horizon-margin ΔS 0.012–0.024; StepNative ~0.013.
- **ASSUMED_AWAITING_VERIFICATION:** taper's converged flip (owed A/B is exactly ladder-implicit),
  ipe-AA through-training ΔS, chroma add-back ΔS, amber sufficiency.
- **Honest tension I did NOT resolve:** firing directional+taper together confounds THEIR mutual
  attribution within a single run — I resolve it with the 4-run cluster ladder (mechanism 1), which costs
  3 NEW fresh runs. A minimalist could argue directional-only-trunk + taper-as-rung-1 is cheaper if the
  directional −48% alone already crosses a milestone. I bias to the cluster because basis-before-capacity
  makes the JOINT the point and 73%+(−48%) are the two levers most worth landing together — but that is a
  cost/attribution judgment the synthesis should weigh against S1's energy-necessity read.
- **Q2 (warm-start-vs-fresh) is NOT my seat** but constrains me: ALL Class-A levers are F2-resume-REFUSED,
  so if the run-1 disposition chooses warm-start, the Class-A cluster CANNOT ride it — the basis cluster
  FORCES a fresh run (or a matching-basis resume). This is a hard coupling S3/synthesis must reconcile:
  **the RANK-1 basis cluster and warm-start-from-run-1 are mutually exclusive.** I lean fresh for exactly
  this reason (run-1 is pre-actuation anyway, Q2).
