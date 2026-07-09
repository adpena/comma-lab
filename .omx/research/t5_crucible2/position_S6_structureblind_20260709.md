# P1 SEAT S6 — STRUCTURE-BLIND DERIVATION of the v7.5.2 training-program SHAPE

**Crucible-2 (task #379). Author: S6 (STRUCTURE-BLIND). 2026-07-09. `[no-triality]` · $0 · #205 untouched.**
Pointer contest-CPU **0.19110 UNMOVED** — this is MEANS. Everything below is a blinded structural
derivation of the *shape* the task NEEDS; it is not a measured claim.

---

## STORES CONSULTED (and the deliberate BLINDING SET)

**READ (permitted):** `CONVENING_20260709.md` · `DELTA_GROUNDING_20260709.md` (measured facts §A–§X;
schedule-shape rows S-1/R-1 treated as *data about a prior experiment*, never a template) ·
`ORCHESTRATION_LEDGER.md` (my charter §49–55 + the **operator POSE ENGAGEMENT GATE §103–111** only) ·
`docs/operating_manual_craft_handoff.md` (craft contract) · the physics carried in
CLAUDE.md/MEMORY (class order + areas + IoU, separatrix=annulus, R operator, scorer read-pattern,
pose⊥d_seg exact-zero, head-quadratic, collapse-fix diagnosis).

**DELIBERATELY NOT READ (the blinding set, per the S6 binding constraint):** SPEC_v75 §2/§4 schedule
constants + curriculum section · `crucible_v7` `_CRUCIBLE_V6_DELTAS` / stage sequence ·
`witness_autoconfig` / `proven_base` / any incumbent launch config · the other five seat positions
(S1–S5) · any FEED that states the incumbent stage ORDER · PR95 curriculum material. I did not open the
incumbent's stage list; where a permitted row *leaked* a fragment of it (S-1 "CE→tau…Muon…pose-finish",
R-1 "CE→tau@257"), I quarantined it as prior-experiment data and derived the skeleton independently, then
compared at the end.

**Honest contamination note (craft-manual "label what you know"):** I cannot un-know that CLAUDE.md/MEMORY
carry the PR95 8-stage skeleton (CE→tau_softplus→smooth→QAT→c1a→lambda→sigma→Muon) in prose. I did NOT use
it as a template. I use it in §8 ONLY as the object my *independent* derivation is compared against — which
is the entire point of the blind seat (agreement = vindication, divergence = cargo-cult catch).

---

## 1. METHOD — derive from the task, not the incumbent

The task is a fixed-point problem: find θ such that `argmax SegNet(D↓·uint8·U↑ · render_θ)` = GT argmax over
n600, and `PoseNet` on the pose-frame ≈ GT pose, minimizing `S = 100·d_seg + √(10·d_pose) + 25·bytes/N`.
A training *program* is a decomposition of this fixed point into sub-goals whose **dependency DAG** and
**natural completion events** dictate the phase skeleton. I derive the DAG from four measured structural
facts, and nothing else:

- **(F-i) Multi-scale class geometry.** 3 large high-persistence classes (Road 22.9%/IoU.955,
  Undrivable 49.3%/.995, MyCar 25.6%/.994) + 2 small low-persistence classes (Movable 1.56%/.903,
  Lane 0.59%/.263, thin, dash-broken, ~8-dim nonlinear orbit). Persistence (Morse–Smale filtration) is
  a **total order on scale**: high-persistence structure is resolvable coarse, low-persistence structure
  appears only at finer scale.
- **(F-ii) The error is codim-1 boundary jitter.** ~97% of d_seg lives in a ~4.7%-area annulus around the
  separatrix; it is *jitter*, not region-miss. The terminal fight is sub-pixel curve placement through R.
- **(F-iii) Birth is recall-without-precision + mass-conserved.** The witness starts near-constant-palette;
  small classes must NUCLEATE, and naive nucleation over-paints Lane 13.8×/Movable 4.6× **into** Road,
  flooring Road d_seg ~0.40 unless an area/mass counter-constraint returns the deficit (R-3 mechanism).
- **(F-iv) Pose is EXACTLY orthogonal to d_seg and conditioning-gated.** ∂d_seg/∂ξ ≡ 0 (SegNet reads only
  the last frame; ξ shapes only frame0). Pose descends monotone from a *conditioned* trunk but WALLS at
  1.2–1.8 from an ill-conditioned one → operator GATE: pose fires only after d_seg is sufficiently
  conditioned (σ_min of J_ξ high enough).

Everything else (which basis, which loss, which optimizer) is then *entailed* per phase.

---

## 2. PHASE COUNT + JOBS (jobs, not names)

The DAG has **four sequential d_seg-training regimes + one in-loop solve + one conditioning-gated
orthogonal appendix = 4 + 1 + 1 = 6 functional blocks.**

- **J1 — FORM the coarse partition.** Establish the high-persistence separatrix segments (Road/Undrivable
  horizon, MyCar/hood static boundary) and nucleate the 3 large basins. Smooth/high-temperature posterior
  energy on a **low-bandwidth, near-isotropic** basis (basins, not edges; sharp loss here manufactures
  Gibbs + premature commitment). This is the region-fill regime.
- **J2 — NUCLEATE the thin/rare structures (Lane, Movable islands) UNDER area-conservation.** Admit finer,
  **directional (curvelet)** basis capacity matched to the lane/movable tangent; turn ON growth forces so
  the low-persistence critical points are born; SIMULTANEOUSLY hold a per-class area-Lagrange constraint so
  birth's recall-without-precision does not cannibalize Road mass (F-iii). Region regime, now
  edge-oriented and constrained.
- **J3 — SHARPEN the separatrix (the annulus / boundary-band fight).** Drive the ~97%-of-d_seg boundary
  flips ACROSS the argmax margin at the finest scale: **step-native** representation (L∞-at-edge, no
  Gibbs), **coverage-integrated render at grid≥384** (place the flip before R's downsample averages it),
  **margin/horizon-weighted satisficing hinge** (not smooth CE — CE is exhausted here, F-ii/R-4),
  **chroma active** (luma-invariant, orthogonal, decides Lane/Movable in the annulus). This is the
  region→boundary handoff; growth forces decay, margin forces peak.
- **J4 — FINISH under an orthogonalized-step optimizer.** Low-noise conditioned polish of the
  already-correct geometry (orthogonalized momentum beats noisy AdamW random-walk once the basin is right).
  This is a distinct regime because it *presupposes* J1–J3 placed the boundary; it explores nothing.
- **J5 — SOLVE the terminal head (not train).** Freeze the feature trunk; the ~791-param **affine** output
  head is a near-quadratic sub-problem → Gauss–Newton/CG **exact solve, full-P (all 600 pairs)**. Replaces
  a terminal head fine-tune. In-loop solve interrupt, not a training phase.
- **J6 — FINISH pose (conditioning-gated, orthogonal appendix).** After d_seg conditioning crosses the
  σ_min threshold, engage w_pose and joint-descend ξ to an R1-class dxi; serialize at export. Because
  pose ⊥ d_seg exactly, this block's ONLY ordering constraint is the conditioning gate — it cannot disturb
  d_seg, so it is an appendix, not an interleave.

---

## 3. ORDERINGS — FORCED vs CONVENTIONAL

**FORCED by dependency logic (not tradition):**

1. **J1 ≺ J2** (form before nucleate) — FORCED twice over: (a) Morse–Smale filtration is monotone in
   persistence, so high-persistence features must resolve before low-persistence ones can be placed
   relative to them; (b) birth is mass-conserved (F-iii) — without existing large basins to conserve
   against, nucleation floods them (the measured Road-floor).
2. **J2 ≺ J3** (nucleate before sharpen) — FORCED: annulus jitter is defined *around an existing
   separatrix*; you cannot sub-pixel-place a boundary of a structure not yet born. Exit of J2
   (all islands present, persistence complete, area at equilibrium) IS the entry to J3.
3. **J3 ≺ J4** (sharpen before finish) — FORCED by optimizer regime: an orthogonalized conditioned polish
   assumes the basin is correct; running it before the margin forces placed the boundary polishes the
   wrong configuration.
4. **J4 ≺ J5** (finish before head-solve) — FORCED: the quadratic-in-head approximation holds only with the
   feature trunk fixed, and the solve is only worth its cost at the trunk's best point. Terminal by
   construction.
5. **J6 after the d_seg-conditioning event** — FORCED by (a) the operator gate and (b) the ill-conditioned
   pose-wall physics (F-iv). It is a *conditioning-gated event*, not an epoch, and (because of exact
   orthogonality) has NO forced order relative to J1–J5 other than "after conditioning is sufficient,"
   which lands it at/after J4.

**CONVENTIONAL (design choice, NOT forced) — I flag these so the synthesis doesn't reify them:**

- **The J2|J3 boundary being a hard stage vs a continuous ramp.** The persistence→scale sweep is
  *continuous*; slicing it into two named stages is a control/legibility convenience. What is forced is the
  monotone hand-off (growth↓, margin↑, capacity↑, temperature↓); the number of *named* stages that hand-off
  is discretized into is conventional.
- **Whether J3 is one stage or splits (region-margin vs chroma/directional fine-finish).** Splitting is
  legibility; the forced content is the term composition, not the stage count.
- **Pose strictly-terminal vs gated-parallel-head.** Exact orthogonality means J6 *could* co-fire the
  moment the gate opens; making it strictly last is a simplicity/attribution choice, not forced.
- **Head-solve as a single terminal event vs periodic re-solve (solve-head / train-trunk-δ / re-solve).**
  Forced is only "trunk fixed at solve time"; cadence is conventional.

---

## 4. EVENT EXITS (defined from the JOB being done, not epoch counts)

Each phase ENDS when its sub-goal is achieved — a measurable predicate, with an epoch CAP only as a
dead-trigger fail-safe (never as the primary exit):

- **J1 exits** on **coarse-class plateau**: Road/Undrivable/MyCar verdict-d_seg slope < ε over a window AND
  coarse posterior entropy settled. (The high-persistence partition has stabilized.)
- **J2 exits** on **birth-completion + area-equilibrium**: all 5 classes present with part_frac above a
  nucleation threshold (Morse–Smale persistence complete) AND per-class area within tolerance of target
  (the R-3 deficit returned). This is literally a persistence-completion event.
- **J3 exits** on **margin-satisfaction OR train-verdict DECOUPLING** (R-4, first-class): either surviving
  flips have shifted to higher GT margin (only frozen-SegNet label-noise remains → terminal) OR train loss
  descends while verdict d_seg rises = the current term is exhausted = "same gradient won't cure it" =
  hand-off. The decoupling detector MUST be distinguished from the EMA-shadow-lag early-verdict artifact
  (R-2) before it is trusted as an exit.
- **J4 exits** on **basin-reached**: within-stage remaining descent < threshold (the NCDE/basin detector's
  "< 5% remaining" → terminal-solve admissible).
- **J5 (head solve) fires** at the terminal EMA ckpt **iff** the quadratic regime RE-VERIFIES on the
  current ckpt (LM ρ ∈ ~[0.8, 1.2], full-P). Otherwise skip (the regime gate is the safety).
- **J6 (pose) fires** on the **conditioning event**: σ_min(J_ξ) ≥ threshold (threshold DERIVED from the
  measured coherence↔conditioning relation, not hand-set), with hysteresis and a never-reached terminal
  backstop. This is the operator's binding gate expressed as a control law.

---

## 5. WHERE SOLVING REPLACES TRAINING

- **Terminal affine head → GN/CG exact solve (J5).** Forced-solvable: near-quadratic loss-vs-head
  (measured ρ~0.85), ~791 affine params, full-P. This is the sharpest solve-don't-train site.
- **Area-constraint multipliers (Chan–Vese λ per class) → dual-ascent / KKT solve, in-loop.** The λ that
  returns the class-area deficit is the solution of the dual problem given the current primal — it should
  be DERIVED-LIVE from the measured mass deficit each step (a solve embedded in J2), not hand-tuned. This
  is "solve the multiplier, train the field."
- **Pose ξ (partly).** Once the conditioned trunk fixes the render, the optimal per-pair 6-dim ξ is a
  small least-squares-shaped sub-problem; its conditioning IS the σ_min basin sensor. J6 is therefore
  partly a solve (the linear/basin part) and partly a conditioned descent (the residual). Flag: partly.
- **Temporal-screw ξ warp → closed-form geometric transform (compute, not train).** H(ξ) on ground-frame
  stop-grad coords is an analytic prior (rule-118 free), not a trained parameter.

Training is reserved for exactly what is provably NOT solvable: the nonlinear feature trunk that carries
the separatrix through R (J1–J4). Everything affine/dual/geometric around it is solved.

---

## 6. TRAJECTORY SHAPE (capacity / basis / loss over the program)

All four trajectories move MONOTONICALLY in the same coarse→fine direction — the single Morse–Smale
persistence anneal, seen four ways:

- **Capacity (effective basis bandwidth / active DOF):** LOW → HIGH, monotone. Start low-bandwidth to form
  the high-persistence partition; admit finer scales as the filtration descends. Capacity must NOT start
  high — **basis-match precedes capacity** (isotropic-high-capacity HURTS; directional-match is the −48%
  lever). This is the curvelet-scale curriculum = persistence order.
- **Basis orientation:** ISOTROPIC → DIRECTIONAL, monotone-rising. Smooth coarse blobs need no
  orientation; the annulus fight demands the basis oriented to the boundary tangent field. Directionality
  peaks in J3.
- **Loss temperature (softmax τ):** SOFT/HOT → HARD/COLD, monotone-annealing. High τ in J1 forms basins
  without Gibbs; τ anneals toward the hard-argmax / step-native / L∞-margin regime by J3–J4.
- **Loss composition:** REGION → BOUNDARY. Growth/birth forces peak in J2 then DECAY; the area-conservation
  constraint peaks in J2 then RELAXES; margin/horizon/chroma forces RISE toward J3. This is the region→
  boundary hand-off.
- **Optimizer:** exploratory noisy (AdamW-class) through J1–J3 → orthogonalized conditioned (Muon-class) at
  J4, with warm-start-momentum + LR re-warmup at the boundary (treat it as a stage boundary) and LR
  annealed to a small final fraction.
- **Substrate constants (NOT trajectories):** grid ≥ 384 + coverage-integrated render is a FLOOR held
  throughout the boundary-relevant phases (a floor, not a ramp). The **collapse-fix** (grad-clip +
  pose-eps-floor + per-param grad-normalize + stage-boundary LR/w_seg guard) is a STABILITY SUBSTRATE that
  must be ON from the first moment multiple boundary-sharpening forces co-fire (J2 onward) — it is the
  precondition that lets the joint descent converge; it is not itself a phase.

---

## 7. DERIVED PROGRAM — stage table

| # | JOB (not a name) | Basis / capacity | Loss regime (temp) | Optimizer | Exit EVENT (cap = fail-safe only) | Solve? |
|---|---|---|---|---|---|---|
| J1 | FORM coarse partition (3 large classes, high-persistence separatrix) | low-bandwidth, near-isotropic | smooth soft posterior, HOT | AdamW | coarse-class d_seg plateau + entropy settled | train |
| J2 | NUCLEATE thin/rare islands (Lane, Movable) UNDER area-conservation | finer, directional (curvelet) | region + growth forces + **area-Lagrange** | AdamW | birth-completion (all 5 present) + area-deficit < tol (persistence complete) | **λ solved live** (dual); field trained |
| J3 | SHARPEN separatrix (annulus margin fight), chroma-active | finest, directional, **step-native**, coverage≥384 | **margin/horizon hinge**, COLD; chroma term | AdamW | margin-satisfied OR **train-verdict decoupling** (≠ EMA-lag) | train |
| J4 | FINISH — low-noise conditioned polish | finest (fixed) | hard margin, coldest | **Muon** (warm-start mom + LR re-warmup, anneal) | basin-reached (<~5% remaining descent) | train |
| J5 | SOLVE terminal affine head | head only (~791 affine) | quadratic (exact) | **GN/CG full-P** | fires iff LM ρ∈[0.8,1.2] re-verified | **SOLVE** |
| J6 | FINISH pose (orthogonal appendix), serialize dxi | ξ carrier (6-dim/pair) | pose MSE, joint | conditioned descent | fires on σ_min(J_ξ)≥thr (derived), hysteresis, backstop | **partly solve** |

Substrate (all phases, not a stage): grid≥384 coverage render · collapse-fix stability levers ON from J2
· train-through-R in-loop.

---

## 8. BLIND-DERIVATION vs INCUMBENT — vindications and cargo-cult catches

Comparing my independent skeleton to what I know of the incumbent/PR95 chain (used ONLY here, per §STORES):

**VINDICATIONS (my structure FORCES a phase the incumbent also has — agreement, not cargo-cult):**
- **J1 smooth-hot region form** ↔ incumbent CE (+ "smooth") stage. Forced by Gibbs-avoidance + basin
  formation + persistence. **And my derivation PREDICTS the incumbent's measured CE-exhaustion (R-4):** CE
  is a region-fill term; once basins form it decouples from the boundary verdict → J3's margin hand-off is
  structurally required, not a patch.
- **J3 temperature anneal (τ soft→hard)** ↔ incumbent tau-softplus. Forced (temperature annealing = the
  level-set anneal). Vindication.
- **J4 orthogonalized finish** ↔ incumbent Muon-final. Forced ("a distinct conditioned-polish optimizer
  regime, last"). Caveat: my structure forces the REGIME, not the specific name Muon (empirical adoption).
- **J2 area-conservation during nucleation** ↔ the post-run-1 Chan–Vese fix. Forced by the birth-over-paint
  mass mechanism (F-iii). Note this is NOT a PR95 echo — PR95 had no from-scratch 5-class nucleation.
- **J6 conditioning-gated terminal pose** ↔ incumbent gated pose-finish + operator gate. Forced by exact
  orthogonality + the ill-conditioned wall.

**DIVERGENCES / suspected PR95 CARGO-CULT residue (the incumbent likely has stages my structure does NOT
force — the elementwise-audit-launders-the-skeleton law made concrete):**
- **QAT stage → NO structural justification for THIS vehicle.** QAT is an FP4-archive-rate concern; the
  witness's trunk weights are FREE (rule-118 generated / store-nothing ~1KB), rate is dead-at-floor. A
  QAT-shaped stage in the d_seg curriculum is a pure PR95 echo. FLAG for the synthesis: if a QAT-like stage
  survives, demand its d_seg-or-rate justification for the *generated* weight regime.
- **c1a / lambda-sweep / sigma-sweep as distinct sequential STAGES → my structure folds them into
  substrate, not stages.** Round-trip-awareness (the "sigma" uint8-noise intent) is a *render/loss property
  held throughout the boundary phases* (train-through-R, coverage-integrate), NOT a late standalone stage.
  "c1a coder-aware" biases weights for brotli — irrelevant for generated/free weights. If these appear as
  separate stages, they are skeleton residue.
- **Stage COUNT.** My structural skeleton is **4 training regimes + 1 solve + 1 pose appendix = 6 blocks**;
  the PR95 chain is 8 linear stages. The delta (QAT + c1a + lambda + sigma as separate stages) is exactly
  the residue the blind seat exists to catch. My count is LOWER, and one incumbent-trained leg (terminal
  head) should be a SOLVE (J5), not a stage — structure says solve where PR95 trained.

**Net:** the *load-bearing* spine (coarse-form → constrained-nucleate → margin-sharpen → orthogonal-finish
→ head-solve → conditioned-pose) is STRUCTURALLY FORCED and my blind derivation converges to it from the
measured physics alone. The residue to interrogate at synthesis is (a) any QAT/c1a/lambda/sigma-shaped
stage that carries no d_seg/rate justification for the generated-weight, rate-at-floor witness, and (b)
any terminal head *fine-tune* leg that should instead be the J5 solve.

---

## 9. TERSE ANSWER (for the synthesis)

- **Phase count/jobs:** 6 blocks — J1 form-coarse · J2 nucleate-under-area-constraint · J3 sharpen-margin
  (step-native/coverage≥384/chroma) · J4 orthogonalized-finish · J5 **SOLVE** affine head (GN/CG full-P) ·
  J6 conditioning-gated pose appendix.
- **FORCED orderings:** J1≺J2≺J3≺J4≺J5 (persistence + mass-conservation + optimizer-regime + trunk-fixed);
  J6 after the σ_min conditioning event (operator gate + ill-conditioned-wall physics).
- **CONVENTIONAL (flag, don't reify):** J2|J3 hard-boundary-vs-ramp; J3 one-vs-split; pose strictly-terminal
  vs gated-parallel; head-solve single vs periodic.
- **Solve-replaces-train:** affine head (GN/CG, J5) · area λ (dual/KKT live, J2) · pose ξ basin (partly,
  J6) · temporal-screw H(ξ) (closed-form, compute-not-train).
- **Cargo-cult catch:** structure forces 6 blocks, PR95 has 8 — QAT + c1a/lambda/sigma-as-stages carry NO
  justification for a generated-weight/rate-at-floor witness; terminal head should be SOLVED not fine-tuned.
