# Deep-math lens — CHAPTER 6: Optimization dynamics + transition-easing DESIGN

**Task #284 (Chapter 6 of "Amortizing the Argmax"). 2026-07-04.** $0 research + a DESIGN
deliverable. No heavy/paid/GPU launched; #205 untouched (read-only). MEANS — the pointer
(**0.19110**) moves only through a byte-closed `upstream/evaluate.py` n600 exact row. This memo
is a decade-compounding apparatus artifact + the operator's live actionable answer to *"how do we
ease the ep300 stage-transition bump."*

---

## TL;DR — the framing verdict + the ranked lever (read this, skip the rest if rushed)

**FRAMING VERDICT: the saddle-to-saddle / MFLD / RG-critical-slowing lens is CONFIRMED as an
on-point *mechanistic* lens and REJECTED as a literal quantitative map.** It correctly NAMES two
things we measured — (a) the slow power-law d_seg tail = sequential saddle-to-saddle escape of the
lowest-persistence lane-dash index directions (escape time ∝ exp(leap), so the finest dashes lock
in LAST); (b) the stage transition = a critical point of the RG/annealing flow where the effective
relaxation time diverges, so *crossing it fast is the destabilizer*. It is a FALSE FRIEND as a
number: MFLD's λ≈1 is a weight-space entropy coefficient, NOT our softmax/tau output-temperature —
do NOT pin a stage boundary to λ=1. The theory is *proven in our regime* only because **Muon ≈
Stiefel-manifold flow** (Spectral-Flattening, 2605.13079), which is the bridge that makes the
multi-index results apply rather than analogize.

**THE ep300 BUMP IS A NUMERICAL-CONTINUATION FAILURE, not a loss-function failure.** MEASURED
(FEED-ft, 2026-06-27, n200 amort decoder): d_seg **0.0056 → 0.020** (3.4×, PERSISTENT 75+ epochs)
at ep300 — diagnosed HYPOTHESIS B (real transition harm, not EMA-shadow-lag). Root cause (FEED-ft#3):
**three landscape changes stacked at ONE epoch** — CE→tau loss-form switch **+** lane-band render
engage **+** (optionally) AdamW m/v zeroed — hit at **FULL LR with stale momentum** because the
spike-guard was cleared but the LR was NOT re-warmed. The operator's cleaner instance: CE-stage
floor ~0.00475 → 0.00592 across the boundary. Same disease.

**THE RANKED TRANSITION-EASING LEVER SET (the deliverable):**

| # | Lever | Continuation role | Exists? | Next-run config | Honest EV |
|---|-------|-------------------|---------|-----------------|-----------|
| **L1** | **Deconflict the boundary** (stagger tau@300 vs band@350) | take homotopy steps ONE at a time | **CONFIG ONLY** | `--lane-band-start-epoch 350` (or `--tau-softplus-start-epoch 250`) | **HIGHEST — free, removes the *superposition* that FEED-ft named as the cause** |
| **L2** | **LR re-warmup at the boundary** | Newton corrector at reduced step | **BUILT, default-off** | `--stage-transition-rewarmup-epochs 20 --stage-transition-rewarmup-floor 0.1 --stage-transition-rewarmup-shape cosine` | **HIGH — this is literally the FEED-ft#3 fix; just turn it on** |
| **L3** | **CE→tau loss-BLEND** (Γ-convergence path) | continuous homotopy in objective | **BUILD** (loss dispatch hard-switches) | new `--seg-form-blend-epochs 40` | MODEST–HIGH; moderate build |
| **L4** | **Lane-band α-RAMP 0→1** (render homotopy) | continuous homotopy in render param | **BUILD** (band engages as a step) | new `--lane-band-ramp-epochs 40` | MODEST; low build (reuse persistence-anneal) |
| **L5** | **Momentum policy = KEEP-tangent + rewarmup** (not zero) | carry the predictor, correct gently | **CONFIG** (choose among existing) | `--stage-transition-reset-moments` **OFF** + L2 ON | MODEST; A/B owed |
| **L6** | **PRIMING / shadow the new signal in late-CE** | pre-position near the new branch | **BUILD** (band has no pre-engage weight) | new `--lane-band-preheat-weight 1e-3 --lane-band-preheat-start 260` | SPECULATIVE–MODEST |
| **L7** | **MD-Decoupling optimizer** (stable-by-construction) | first-step-safe base flow | **BUILT** (`--optimizer md`) | `--optimizer md` (whole-run ablation) | SPECULATIVE at our scale |

**In-flight live test of the whole principle:** #270 fires the Muon warm-start (`--muon-warm-start-momentum`)
+ Muon LR anneal (`--muon-lr-final-frac 0.1`) at the ep726 boundary on #205 — i.e. **warm@726 vs
#205's cold@300** is already an A/B of L2+L5 on the *same* run. Watch it before committing the
CE→tau-boundary versions.

---

## PART I — THE DYNAMICS, FORMALIZED (with real citations)

The witness trainer minimizes, per epoch, an effective objective that changes across the
curriculum CE → tau_softplus → l7_softplus → Muon-finisher. The right physical picture is a
**variational level-set flow of a Morse-Smale complex** (the project unification) driven by a
stochastic gradient system whose *metric* and *landscape* both change at stage boundaries.

### 1. Saddle-to-saddle / MFLD feature learning — the slow d_seg tail

Amortizing the 5-class SegNet argmax is a **multi-index recovery problem**: the ~8-dim lane-orbit
manifold is the "base" of hidden index directions; each inter-class boundary component ≈ one index
direction. Multi-index gradient dynamics turn these directions ON **sequentially** — the system
sits near a saddle, escapes along the highest-signal direction, sits near the next saddle, etc.
(**saddle-to-saddle** dynamics). Each escape time is governed by the direction's **leap exponent**
and grows *exponentially* in it.

- **Abbe–Boix-Adserà–Misiakiewicz, COLT'23** ("leap complexity"): directions are learned in
  increasing leap order; escape is a staircase of plateaus.
- **Ben Arous–Gheissari–Jagannath** (summary statistics / BBP escape): the saddle-escape time
  from a random start is set by the signal-to-noise of the informative direction — the canonical
  "long plateau then sudden drop" law.
- **Mousavi-Hosseini–Wu–Erdoğdu, 2408.07254 (ICLR'25)**: multi-index recovery *via MFLD*; a
  **positive-curvature / sphere (Stiefel) constraint converts EXPONENTIAL escape → POLYNOMIAL**.
- **Chizat, 2202.01009**: MFLD annealing; a log-decaying noise schedule reaches the global min of
  the *unregularized* objective.
- **Nitanda–Wu–Suzuki 2022**: convex MFLD / propagation-of-chaos (the mean-field limit is convex
  in the distribution).
- **Shang–Wakayama–Lecué–Suzuki, 2606.31429** (operator-flagged): base-fiber geometry of MFLD
  feature learning — the stationary measure concentrates on the BASE (indices), Lévy-Milman
  UNIFORM on the FIBER. This is the formal justification for spending ZERO codec rate on the
  pose-null fiber (our task-space quotient, #155).

**Why it applies to US and is not analogy: Muon ≈ Stiefel-manifold updates.** Muon's Newton-Schulz
orthogonalization projects each 2-D weight update onto (approximately) the Stiefel manifold
(**Spectral Flattening Is All Muon Needs, 2605.13079**, research-verified real). The multi-index /
positive-curvature-→-polynomial-escape results are *proven in exactly the Stiefel regime Muon
realizes*. This is the mechanistic license for the **leap-residual sub-stage (#217)**: a post-Muon
micro-stage that (i) margin-reweights the surviving lowest-persistence dash pixels (Damian
landscape-smoothing, 2022–23), (ii) keeps Muon's Stiefel orthogonalization ON (the
curvature-→-polynomial-escape lever), (iii) optional log-decay SGLD noise. It targets EXACTLY the
"d_seg down but very slowly" tail.

**MEASURED anchor:** Muon descended d_seg **−32%** vs AdamW from an identical stage-4 fork, gap
widening monotonically (`muon_vs_adamw_from_stage4_convergence_arm_20260622`) — consistent with
Muon being the κ-buster (polar Newton-Schulz → σ=1 → O(ln 1/ε) vs AdamW O(κ·ln 1/ε) on the
correlated boundary Hessian, κ≈19).

### 2. STE = a Clarke subgradient selection through the non-smooth argmax

The d_seg signal passes through `argmax`/`max` (SegNet class decision) and through uint8/round
quantization in the R-roundtrip — both **non-smooth**. The straight-through estimator (STE)
substitutes a surrogate Jacobian for the (a.e. zero, boundary-set-valued) true derivative.

- The correct object is the **Clarke generalized subdifferential** ∂_C f (Clarke, *Optimization
  and Nonsmooth Analysis*, 1990). `max` is **Clarke-regular**; ∂_C(max) = conv{e_i : i ∈ argmax}.
  On the interior (unique argmax) it is a singleton (zero for the losing classes) — this is the
  *flat-interior* the Fisher metric sees as "dark." On the codim-1 **separatrix** (ties) it is
  **set-valued** — a segment, not a point.
- **Bengio–Léonard–Courville, 2013** (STE) and **Yin et al., 1903.05662** ("Understanding STE"):
  STE picks a *specific element* of this set-valued map as the descent direction; convergence
  holds when the selection is a *conservative field* (Bolte–Pauwels) consistent with ∂_C.
- **Consequence for us:** the boundary-pixel **flicker** is where ∂_C is set-valued — a
  measure-zero set that is nonetheless the ENTIRE d_seg signal (the argmax is flat everywhere
  else). The STE selection is well-posed *away* from the tie set and *ambiguous ON it*; the flicker
  is the dynamical footprint of that ambiguity, not a bug. This is why linear "store-the-flip-pixel"
  sidecars NO-GO'd ×3: the error lives on the set where the subgradient is genuinely multivalued.

### 3. The curriculum = RG flow = annealing = critical slowing down

CE → tau_softplus → l7 → Muon is simultaneously **coarse-to-fine** (curvelet scale), **temperature
annealing** (softmax-temp hi→lo, hosc-β low→high), and a **renormalization-group (RG) flow** on the
effective boundary sharpness. The softmax-temp anneal `_softmax_temp_for_epoch` (cosine/geometric)
IS the temperature schedule; the seg-form curriculum IS the coarse-graining schedule.

- **Simulated annealing ↔ RG** (classical; Bradde–Bialek 2017 "PCA meets RG" for the ML-facing
  version): lowering temperature = integrating out fine scales = flowing toward the sharp fixed
  point.
- **Critical slowing down** (dynamical-systems universality; Scheffer et al. 2009 for the
  power-law-near-transition signature): near a bifurcation / phase transition the leading
  relaxation eigenvalue → 0, so the **relaxation time diverges as a power law**. Concretely: as
  tau → 0 the softmax sharpens, the boundary Hessian stiffens, and the per-step progress near the
  transition slows critically. **This is why `--tau-anneal-shape geometric` (log-spaced) exists —
  it spends MORE epochs at small tau, respecting the critical slowdown** instead of taking one
  large near-tau→0 continuation step (which drove the measured late-tau d_seg volatility).
- **The keystone corollary (the operator's question):** *a stage transition is a critical point of
  this flow.* Crossing a critical point requires SMALL steps (the relaxation time is large there).
  Cold-reset-plus-step-shocks cross it in ONE epoch — maximally violating the critical-slowing
  scale. Every transition-easing lever in Part IV is a way to respect the divergent relaxation time
  at the boundary.

### 4. The flicker floor = the SDE's irreducible-noise floor

At convergence the trainer is a Langevin-type SDE: dθ = −∇L·dt + √(2T_eff)·dW, whose stationary
distribution has variance ∝ T_eff (LR·batch-noise). The residual boundary-pixel **flicker** is the
STATIONARY FLUCTUATION of this SDE, not un-converged bias. #205 measured *onto* this floor (the
d_seg flicker stopped decreasing while the loss kept micro-adjusting). **This BOUNDS what
transition-easing can buy:** easing removes the TRANSIENT bump (a bias excursion off the branch),
NOT the floor (the variance). So the honest ceiling of the ep300 work is "recover the ~0.0011–0.014
that the bump ADDED," not "beat the flicker floor" — the latter needs representation (curvelet-finest,
step-native) or the leap-residual escape, not transition-easing.

### 5. Muon as natural-gradient / Stiefel flow (ties 1+3)

Muon's orthogonalized update is a **spectral natural gradient**: it whitens the update's singular
spectrum to σ=1, which is a Riemannian (Stiefel) step, not a Euclidean one. Under a metric change
at a stage boundary, a Euclidean momentum buffer is a velocity in the OLD metric; Muon's Stiefel
projection partially *re-metrizes* the carried velocity — which is exactly why the Muon switch is
*more* forgiving of warm momentum than a bare AdamW→AdamW boundary, and why **#217's leap-residual
belongs AFTER Muon** (the Stiefel curvature is what turns exp→poly escape).

---

## PART II — THE ep300 BUMP, DISSECTED (grounded in FEED-ft)

**MEASURED (FEED-ft, 2026-06-27, `levelset_amort_decoder_n200`, macOS-CPU advisory, live-vs-EMA
both realized-through-R):** at ep300, d_seg jumped **0.0056 → 0.020** (3.4×) and stayed elevated
75+ epochs. Ruled OUT HYPOTHESIS A (EMA-shadow-lag artifact) by measuring live AND EMA — both rose
→ **HYPOTHESIS B: real transition harm.** FEED-ft#3 root cause: the **tau + lane-edge confound
fired SIMULTANEOUSLY @ep300 with UNDER-treatment.** The operator's cleaner instance: CE floor
~0.00475 → 0.00592 (the `CE ep299 | 0.005927` row corroborates the CE-stage level).

**What the trainer does at ep300 by default (verified in code,
`experiments/train_levelset_witness_realized_through_R_mlx.py`):**

1. `_seg_form_for_epoch(300)` returns `tau_softplus` — a **HARD switch** ce→tau (line 968-975).
2. `--lane-band-start-epoch` **defaults to 300** — the render-band engages as a **STEP** at the
   SAME epoch (`_band_start <= ep`, line 3614). **THE COLLISION IS BUILT INTO THE DEFAULTS.**
3. The spike-guard IS re-treated (`recent_losses.clear()`, lines 3482-3486 + 3615-3618) — good.
4. BUT the LR re-warmup (`_stage_rewarmup_factor`) is **DEFAULT-OFF** (`--stage-transition-rewarmup-epochs 0`
   → returns exactly 1.0, line 1065) — so the boundary is hit at **FULL scheduled LR**.
5. AdamW moment reset is **DEFAULT-OFF** (`--stage-transition-reset-moments`), so momentum is
   **stale** (a velocity in the pre-boundary metric) — pushed through the landscape change.

So the default #205-class config commits *exactly* the FEED-ft#3 failure: **two simultaneous
branch-jumps (tau + band) crossed at full LR with a stale predictor.** The fix machinery (FEED-fw /
BUILD 1: `_stage_rewarmup_factor` + reset-moments) was BUILT but ships default-off byte-identical —
it has never been turned on in a launched n600 config. **Part IV turns it on and adds the two
missing homotopies (loss-blend, band-ramp).**

---

## PART III — THE TRANSITION-AS-NUMERICAL-CONTINUATION FRAME (the keystone)

A stage transition is a **homotopy** H(θ, s), s ∈ [0,1], deforming the old-stage objective (s=0)
into the new-stage objective (s=1). The solution branch θ*(s) is the moving minimizer. Standard
**numerical continuation / homotopy** (Allgower–Georg; pseudo-arclength; predictor-corrector)
crosses such a deformation by:

1. **Small steps in s** (don't jump the homotopy parameter) → **RAMP the perturbation** (L3 loss-blend,
   L4 band-α-ramp).
2. **Tangent predictor** (extrapolate along dθ*/ds) → **CARRY momentum**, corrected for the metric
   change (L5 keep-tangent; the Muon warm-start #270 is the metric-corrected version).
3. **Newton corrector at reduced step size** until back on the branch → **LR re-warmup from a floor**
   (L2).
4. **One continuation parameter at a time** (a codim-2 crossing is ill-conditioned) → **deconflict**
   simultaneous boundaries (L1).

Cold-reset-plus-step-shocks does the OPPOSITE of all four: it jumps s: 0→1 in one epoch, discards
the predictor (zeroes momentum), applies full-LR corrector immediately, and crosses TWO parameters
(tau + band) at once. The ep300 bump is the predictor/corrector diverging off the branch — a
transient the branch re-converges from over ~75 epochs (the persistence FEED-ft measured). **The
entire lever design is: turn cold-reset-plus-step-shocks into pseudo-arclength continuation.**

Critical-slowing (Part I.3) is *why* the small-step discipline is non-optional: at the transition
the relaxation time is large, so a step of fixed size travels a large *number of relaxation times*
off-branch — precisely where a naive step overshoots.

---

## PART IV — THE RANKED TRANSITION-EASING LEVER DESIGN (deliverable)

For each: the deep-math justification, exists-vs-build, the exact next-run config, honest EV.
Ranked by (EV × certainty) / build-cost. All are FINISHING-adjacent MEANS; none moves the pointer
without a byte-closed n600 exact row through R.

### L1 — DECONFLICT the boundary (stagger tau vs band). CONFIG-ONLY. **HIGHEST EV.**
- **Deep-math:** numerical-continuation principle #4 — cross ONE continuation parameter at a time.
  The CE→tau switch and the lane-band engage are two *independent* deformations; superimposing them
  is a codim-2 crossing (ill-conditioned; the Jacobian of the joint move is rank-deficient in the
  worst case). FEED-ft explicitly named the SIMULTANEITY (not tau per se) as the cause.
- **Exists:** yes — both start-epochs are already flags; the collision is only in the shared default
  of 300.
- **Next-run config:** `--lane-band-start-epoch 350` (band trails tau by 50 ep so the tau branch
  re-converges first) — OR `--tau-softplus-start-epoch 250` (tau leads). Prefer trailing the band,
  since tau is the sharper landscape change and the band's render-target change is easier to absorb
  once the argmax has re-sharpened.
- **EV:** highest — free, directly removes the measured cause, composes with everything below. The
  ONLY risk is a second (now-separated) mini-bump at 350; L2 covers it.

### L2 — LR RE-WARMUP at the boundary. BUILT (default-off). **HIGH EV.**
- **Deep-math:** the Newton-corrector-at-reduced-step-size. After a metric change the descent
  direction in the NEW metric is mis-estimated for a few steps; full LR on a mis-aligned/stale
  direction overshoots the branch. Ramping LR from a floor gives the (kept or reset) optimizer state
  time to re-warm against the new landscape — "stable by construction" (the function's own docstring,
  line 1059-1063). This IS the FEED-ft#3 fix.
- **Exists:** `_stage_rewarmup_factor` (line 1049) fires on ANY registered AdamW→AdamW boundary
  (curriculum seg-form / lane / msal / thin / band / subpix / chroma). Default `rewarmup_epochs=0`
  → returns 1.0 → bit-identical.
- **Next-run config:** `--stage-transition-rewarmup-epochs 20 --stage-transition-rewarmup-floor 0.1
  --stage-transition-rewarmup-shape cosine`. (Cosine floor→1 over 20 ep; floor 0.1 = start at 10%
  LR. Fires at BOTH the tau boundary and the staggered band boundary automatically.)
- **EV:** high, high-certainty (the machinery was built precisely for this; it's a corrector-step
  fundamental). A/B owed but low-risk. Compose with L1.

### L3 — CE→tau loss-BLEND (Γ-convergence homotopy). BUILD. **MODEST–HIGH EV.**
- **Deep-math:** replace the hard seg-form switch with a convex path of functionals
  `L_s = (1−s)·CE + s·tau_softplus`, s ramping 0→1 over a window. **Γ-convergence** (De Giorgi)
  guarantees minimizers of L_s converge to minimizers of the target as s→1, and a *continuous path*
  of functionals keeps the argmin on a *continuous branch* (no branch-jump). This is the loss-form
  analog of the already-continuous softmax-temp anneal. NOTE the partial overlap: tau_softplus is
  itself a softened objective and the softmax-temp anneal already provides *some* soft→sharp path;
  the residual step is the FORM change (cross-entropy vs margin softplus), which the blend removes.
- **Exists:** NO — `_seg_form_for_epoch` returns a single form per epoch (hard dispatch). The loss
  computation would need to evaluate both forms in the window and convex-combine (a ~2× loss-eval
  cost only inside the ~40-epoch window).
- **Next-run config (after build):** `--seg-form-blend-epochs 40` (blend s: 0→1 linearly over
  ep280→320 around the tau boundary; default 0 = hard switch = bit-identical).
- **Build sketch:** in the loss dispatch, when `blend_epochs>0` and `ep` is within the window
  around `tau_softplus_start_epoch`, compute `s = clip((ep − (start−blend/2))/blend, 0, 1)` and
  return `(1−s)*ce_loss + s*tau_loss`. Guard: outside the window, single form (bit-identical).
- **EV:** modest–high; moderate build. This is the most *principled* lever (Γ-convergence is the
  exact theorem) but its marginal value on top of L1+L2 is uncertain until measured — the softmax
  anneal may already absorb most of the form-change shock. Build it, but rank it after the two
  free/built levers.

### L4 — Lane-band α-RAMP 0→1 (render homotopy). BUILD. **MODEST EV.**
- **Deep-math:** the band changes the RENDERED frame → the R-roundtrip → the scorer input; that is
  a continuation in the render parameter. The band engaging as a STEP is a 0→1 jump in the render
  homotopy; ramping α is stepping the homotopy parameter gradually (principle #1).
- **Exists:** partial. `--lane-band-weight` is a CONSTANT scalar passed once (line 1751); the gate
  is a step (`_band_start <= ep`). The RAMP PATTERN already exists for a sibling lever —
  `persistence_anneal_weight(ep, persist_w, persist_warmup)` (line 3621) does exactly a per-epoch
  linear warm-up — so this is a small, pattern-matched build.
- **Next-run config (after build):** `--lane-band-ramp-epochs 40` (ramp band α 0→`lane-band-weight`
  over 40 ep from `lane-band-start-epoch`; default 0 = step = bit-identical).
- **Build sketch:** wrap the band weight in `band_w_eff = lane_band_weight *
  linear_ramp(ep, band_start, band_ramp_epochs)` at the band-loss callsite; reuse the
  persistence-anneal helper shape.
- **EV:** modest; low build. Composes with L1 (ramp starts at the staggered 350). Lower rank than
  L3 because the render-target change is a gentler landscape change than the loss-form change (FEED-ft
  named tau as the sharper one), but it is cheap and removes the *second* branch-jump entirely.

### L5 — MOMENTUM POLICY: keep-tangent + rewarmup (NOT zero). CONFIG among existing. **MODEST EV; A/B.**
- **Deep-math:** the tangent predictor. Two options at an AdamW→AdamW boundary: (a) **reset** m/v to
  zero (discard the predictor — safe but starts cold, re-warms over ~1/(1−β) steps = the first-step
  wild-direction risk), or (b) **keep** stale m/v (carry the predictor — but it is a velocity in the
  OLD metric). The continuation ideal is neither pure option but a **metric-corrected rescale**;
  since AdamW's second moment v re-adapts within ~1/(1−β₂) steps on its own, the *practical* optimum
  is **keep the tangent + re-warm the LR gently** (L2) so the carried velocity is corrected by small
  steps rather than discarded. Caveat (2605.10468): do NOT crank LR at an Adam→different-regime
  transition (implicit-bias mismatch); the small-LR re-warm respects this.
- **Exists:** yes — `--stage-transition-reset-moments` toggles zero-vs-keep; L2 provides the gentle
  corrector.
- **Next-run config:** `--stage-transition-reset-moments` **OFF** (keep tangent) **+** L2 ON. The
  A/B is {keep+rewarmup} vs {reset+rewarmup}.
- **EV:** modest; the direction is a genuine open question (the theory says keep-corrected wins, but
  our loss is non-smooth so the stale second-moment could be actively harmful). **#270's Muon
  warm-start at ep726 is the metric-corrected version of THIS exact question on the same run** — its
  result should set the default for the AdamW boundaries.

### L6 — PRIMING / shadow the new-stage signal in late-CE. BUILD. **SPECULATIVE–MODEST EV.**
- **Deep-math:** homotopy from a GOOD initial guess needs fewer corrector steps. Pre-position θ near
  the new-stage branch BEFORE the transition by applying the new signal at a tiny weight during the
  prior stage — so the branch-jump at the boundary is a small increment, not a 0→1 shock. This is the
  #208 early-seeding idea (which exists for INIT via `--seed-islands`) generalized to the band's
  *loss* signal.
- **Exists:** NO for the band (seed-islands seeds the initial argmax, not a pre-engage loss weight).
- **Next-run config (after build):** `--lane-band-preheat-weight 1e-3 --lane-band-preheat-start 260`
  (apply the band at 1e-3 from ep260, then L4 ramps it to full from 350). Composes with L4 as its
  pre-window.
- **EV:** speculative–modest; lowest-priority build. The risk is the pre-heat perturbing the CE
  stage's own convergence; keep the weight tiny and gate it late (ep260+). Build ONLY if L1–L4 leave
  a residual bump.

### L7 — MD-Decoupling optimizer (stable-by-construction). BUILT. **SPECULATIVE at our scale.**
- **Deep-math:** MD-Decoupling (2606.25971) is designed so "destabilizing first-step updates never
  appear" — no warmup needed, LR-transfer across width. A stage transition IS a first-step moment,
  so an MD base would make transition entries free by construction (the whole L2/L5 problem
  dissolves).
- **Exists:** yes — `--optimizer md` wired (`MDDecoupledOptimizer`, base trainer line 1992-2000).
- **Next-run config:** `--optimizer md` (whole-run base swap; a clean ablation, NOT a per-boundary
  lever).
- **EV:** speculative at 60–230K params (MD's guarantees are demonstrated at width-scaling / LLM
  regimes; our conv-INR is small). Worth a single clean ablation arm AFTER L1+L2 establish the
  eased-transition baseline — if MD matches the hand-tuned re-treat for free, it's the durable win.

---

## PART V — HONEST CLASSIFICATION (proven / conjectured / false-friend)

**PROVEN (measured or formally established transferable mechanism):**
- Muon −32% d_seg vs AdamW from the same fork (MEASURED, `muon_vs_adamw_from_stage4_convergence_arm_20260622`).
- The ep300 bump is REAL transition harm, not EMA-lag (MEASURED live+EMA, FEED-ft; 0.0056→0.020, 3.4×,
  75+ ep persistent).
- The cause is the tau+band SIMULTANEITY at full-LR + stale momentum (diagnosed FEED-ft#3; verified
  the collision is in the code defaults, tau@300 = band@300).
- Muon ≈ Stiefel flow (2605.13079, research-verified) → multi-index/saddle-to-saddle theory applies
  in our regime (not analogy).
- STE = a Clarke-subgradient selection through a Clarke-regular max (formal; Clarke 1990,
  Bolte–Pauwels conservative fields).
- The LR-re-warmup + reset-moments machinery is BUILT and default-off byte-identical (verified in code).
- Critical-slowing motivates `--tau-anneal-shape geometric` (already built for exactly this reason).

**CONJECTURED (plausible, mechanism-backed, UNMEASURED on our n600 through-R):**
- That the slow d_seg tail IS specifically the low-persistence-dash saddle escape (needs the #216
  signature test: DISCRETE per-component plateaus ⇒ confirmed; a SINGLE smooth power-law ⇒ weaker,
  favor curvelet-finest instead).
- That the leap-residual #217 sub-stage shortens the tail (exp→poly escape).
- Which momentum policy wins (keep-tangent vs reset) — #270 will inform.
- The magnitudes: CE→tau blend window (40 ep?), band ramp (40 ep?), rewarmup floor (0.1?) — all
  need the A/B.
- That L3 (loss-blend) adds value ON TOP of L1+L2 (the softmax anneal may already absorb the form
  change).

**FALSE-FRIEND (do NOT map literally):**
- **MFLD λ≈1 ≠ our softmax/tau temperature.** λ is the weight-space entropy/noise coefficient; tau
  is the output-space class-logit sharpness. Do NOT pin a stage boundary to λ=1 in the existing
  anneal (only meaningful if we add an explicit SGLD/entropy term whose coeff IS λ).
- **"RG fixed point" is an analogy, not a computed fixed point.** Use it for the coarse-to-fine +
  critical-slowing intuition; do NOT claim measured critical exponents.
- **Muon successors (Dion/SOAP/Shampoo/NorMuon/MuonClip) are SCALE-plays** we don't need — not a
  transition lever (settled in the Muon deep-dive).
- **Do NOT crank LR at the Adam→Muon (or any regime) transition** (2605.10468 implicit-bias mismatch)
  — the levers here are all *gentle* (small-LR re-warm), which is the point.

---

## PART VI — ENGINEERING NEXUS

### The next fresh-run transition-easing config block (composable, built + build)

```
# --- built + config-only (land immediately, no code) ---
--tau-softplus-start-epoch 300
--lane-band-start-epoch 350               # L1: deconflict (band trails tau by 50)
--stage-transition-rewarmup-epochs 20     # L2: LR re-warmup at EVERY boundary
--stage-transition-rewarmup-floor 0.1
--stage-transition-rewarmup-shape cosine
--tau-anneal-shape geometric              # (critical-slowing: more epochs at small tau)
# --stage-transition-reset-moments OFF    # L5: keep the tangent (default; do not pass)

# --- needs building (small, pattern-matched, default-off byte-identical) ---
--seg-form-blend-epochs 40                 # L3: Γ-convergence CE->tau loss blend
--lane-band-ramp-epochs 40                 # L4: render-homotopy band alpha-ramp
--lane-band-preheat-weight 1e-3            # L6: prime the band in late-CE (optional)
--lane-band-preheat-start 260

# --- the Muon finishing boundary (already built, #269; the in-flight #270 A/B) ---
--muon-start-epoch 726
--muon-warm-start-momentum                 # L5 at the Muon boundary (metric-corrected tangent)
--muon-lr-final-frac 0.1                   # L2 at the Muon boundary (anneal Muon LR)
```

### The leap-residual #217 finishing micro-stage (separate from transition-easing)
Post-Muon, add a short sub-stage that margin-reweights the surviving lowest-persistence lane-dash
annulus pixels (Damian smoothing) with Muon-Stiefel ON (curvature → poly escape) + optional
log-decay SGLD. This attacks the *floor* (the saddle-escape tail), which transition-easing
CANNOT reach (Part I.4). Gate it on the #216 signature test first (discrete-plateaus verdict).

### The RG-optimal curriculum shape (already partly realized)
- soft→sharp with MORE resolution near the sharp end (`--tau-anneal-shape geometric`) — respects
  critical slowing.
- per-stage re-treatment at EVERY boundary (the operator's 2026-06-26 law), now instantiated by
  L1–L5.
- learn ONLY PR95's d_seg-conditioning subset (CE → margin-softplus → Muon-drop), NOT the 14k
  rate-machinery epochs (the witness has a different, tiny rate story) — a SHORT curriculum.

### The live A/B that tests this design NOW
**#270 = warm@726 vs #205's cold@300 on the SAME run.** #205 crossed ep300 cold (the bump);
#270 crosses ep726 warm (`--muon-warm-start-momentum` + `--muon-lr-final-frac 0.1`). If the warm
Muon boundary shows a smaller/absent transition spike than #205's cold ep300, that is direct
evidence for L2+L5 → promote the CE→tau-boundary versions (L1–L3) into the next fresh run. Watch
#270's transition-epoch d_seg trace before committing the builds.

---

## Cross-refs + provenance
- Measured anchors: FEED-ft / FEED-ft#3 (ep300 bump diagnosis), FEED-fw / BUILD 1 (the re-warmup +
  reset machinery), `muon_vs_adamw_from_stage4_convergence_arm_20260622` (−32%),
  `_stage_rewarmup_factor` / `_seg_form_for_epoch` / `_softmax_temp_for_epoch`
  (`experiments/train_levelset_witness_realized_through_R_mlx.py`), `MDDecoupledOptimizer`
  (`src/tac/optimization/md_decoupling.py`), `build_muon_finisher_optimizer` +
  `_seed_muon_momentum_from_adam` (#269, cba2e4375).
- Memories: `[[mfld-multiindex-leap-saddle-to-saddle-muon-stiefel]]` (#217),
  `[[muon-deep-dive-keep-and-tune-finishing-stage-schedule-not-switch]]`,
  `[[muon-restart-config-change-deterministic-repro-provenance]]` (#270),
  `[[different-stages-need-different-treatment-regardless-carrier-substrate]]`,
  `[[unified-variational-levelset-flow-everything-is-facets]]`.
- Papers (research-verified via the memories above; theory-only, no material OSS beyond Muon):
  2606.31429 (Shang-Wakayama-Lecué-Suzuki, MFLD base-fiber), 2408.07254 (Mousavi-Hosseini-Wu-Erdoğdu,
  multi-index-via-MFLD sphere→poly), 2202.01009 (Chizat, MFLD annealing), 2605.13079 (Spectral-
  Flattening-Muon ≈ Stiefel), 2606.25971 (MD-Decoupling), 2605.10468 (Adam→Muon transition
  implicit-bias), 1903.05662 (Understanding STE), COLT'23 Abbe-Boix-Adserà-Misiakiewicz (leap),
  Ben Arous-Gheissari-Jagannath (saddle escape), Damian 2022-23 (smoothing), Clarke 1990 (nonsmooth
  analysis), Scheffer 2009 (critical slowing). — Do NOT re-research these; ledger'd here.
- Chapter 6 of "Amortizing the Argmax" (task #284). Sisters: Ch.4 (Γ-convergence annealing),
  the θ* lever stack (#201), the triality DAG↔DSL↔equations.

**MEANS. Pointer 0.19110 UNMOVED — only a byte-closed n600 exact row through `upstream/evaluate.py`
moves it. This memo is the DESIGN + the operator's actionable transition-easing answer; the verdict
is the measured A/B on the next fresh run (and #270 as the live preview).**
