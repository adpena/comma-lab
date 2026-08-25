# THE POSE-LEGIBLE WITNESS — aperture-problem diagnosis + ξ-consistent textured class/depth-stratified render (2026-07-08)

## ⛔ 0a. MEASURED (c2adba9aa, same day): THE APERTURE HYPOTHESIS (§1) IS FALSIFIED — mechanism REFINED to WRONG-FLOW-OBSERVABILITY; the T×D cell is the surviving live bet
A0T probe (n24, through-R, pos-control 2.1e-12): texture on the GLOBAL ground-H flow made d_pose WORSE
(1.685 → 15.14 best-case, →118 at high amp; monotone; NO low-|t| rescue — the aperture signature is
ABSENT). Diagnostic that pins it: warp(frame,ξ) vs itself reads d_pose 166–186 — the global H warps the
WHOLE frame as ground-plane (sky/structure/cars given wrong motion); the flat render's 1.685 was a
weak-residual reading BECAUSE the wrong flow was unreadable. Texture exposed it. Semantic-prior dominance
ALSO ruled out (PoseNet responds 1.7→118 to injected flow — it reads pixels). verdict_scope: formulation —
"texture restores pose on the GLOBAL-ground-H carrier" is the falsified claim; NOT tested/killed: the
**T×D cell** — texture advected by the PER-CELL stratified flow (§2's D layer; every prior arm tested T
and D SEPARATELY: A2+ = D-without-T on a flat render, A0T = T-without-D on the wrong global flow). A1T
probe FIRED (texture × stratified_depth_warp, same harness) with the fallback scale/convention diagnostic
(sweep s·ξ + sign-flips; the 6 scored dims' semantics are INFERRED-unconfirmed) if A1T stays ~flat-level.
§4's prediction is retired; §5's GREEN gate is now keyed to A1T. d_seg guard also failed at A0T amps
(3.7–13% flips) — smaller amps + interior gate owed in A1T.

## ⛔ 0b. MEASURED (7d2784fc9): A1T ALSO FALSIFIED — the T×D cell does not collapse (best 2.608 > A0 1.685 > A2+ 1.223; per-cell flow reads WORSE than global-H, self-pair 183.5 vs 165.6 — the piecewise composite is discontinuous, not rigid motion); no convention bug (s·ξ sweep monotone toward flat floor as s→0). TRIANGULATED: any cheap GEOMETRIC carrier's flow is wrong-as-read; legibility amplifies the error. verdict_scope: formulation — the geometric stratified store-nothing carrier. LIVE: (1) L2 true-depth probe (mono-depth on real_f0 → K·T(ξ)·D·K⁻¹ → PoseNet, $0, FIRED) — separates "our depth models too crude" from "PoseNet needs real scene content"; (2) joint pose-descent (#238 R1 re-validation); (3) HONEST BUDGET CORRECTION: at the measured floor ~1.7, pose contributes √(10·1.7)≈4.1 — "budget item" is NOT viable for sub-0.19; pose must be SOLVED (joint-descent or paid-depth) or this witness architecture is pose-blocked. §5's GREEN gate: RED.

**Operator directive (verbatim, binding):** "dig deeper than ever into pose research and task space SDF
level set research and v7.5 and v8 and openpilot and upstream evaluate.py and modules.py because this
isn't intractable it is a frontier bleeding edge deep math and differentiable geometry and information
and domain problem and we are so close; openpilot and research tell us how to handle the depth and
parallax issues and scorer tells us how to fool the posenet and we may need to tweak our witness and
level set and morse smale and terms to possibly render something more precisely segmented by class and
depth as appropriate for conditioning with the necessary information."

STORES CONSULTED: pose_mladder_depthwarp_measured (70649531f — A0 1.685 / A2 1.486 / A2+ 1.223, Rung-0
anti-correlation −0.45/−0.68, off-plane parallax mass 0.5%) · pose_solve_output_space_inverse (§0a
refutation) · pose_carrier_arms_measured (pair-consistency law: consistent cartoon 1.995, real-f0-mixed
10.42) · council_pose_carrier_optimal_form_symposium_20260703 (the L3 null-texture lever — this memo is
its MEASURED-DIAGNOSIS completion, not a new invention) · FEED-posehard/poseresearch/poseladdermeasured ·
#141 margin-saliency (placement field) · #204 filter-chain (texture band through R) · #212 kernels.
Pointer **0.19110 UNMOVED** — design + $0 probe plan; nothing is a score until byte-closed exact.

## 1. THE DIAGNOSIS the ladder data was carrying (new; the WHY of the 1.2–1.7 floor)
A consistent FLAT cartoon pair floors d_pose ~1.2–1.7 while the real pair scores ~0, with parallax mass
0.5% and d_pose WORST on LOW-|t| pairs (anti-correlation). The discriminating variable is not geometry —
it is **flow OBSERVABILITY. Piecewise-constant cell interiors make optical flow unobservable except
normal-to-boundary (the classical APERTURE PROBLEM).** The flat witness under-informs PoseNet: motion is
depicted only on the sparse boundary set, and small flows (low-|t| pairs) fall below readability first —
exactly the measured anti-correlation. The information deficit is WITHIN-CELL TEXTURE CARRYING THE FLOW.
This also explains the A2 failure mechanistically: on a flat render the ξ→PoseNet-output Jacobian is
near-rank-deficient (perturbing ξ moves nothing inside flat cells) → tiny reachable set → 1.486 local
min. The solve machinery was starved, not wrong.

## 2. THE DESIGN — three layers that only work TOGETHER (each alone measured/known insufficient)
**(T) Texture = the information channel (rule-118 FREE):** seeded procedural texture (deterministic
generator + seed → zero counted bytes for realization) painted into cell interiors of BOTH frames,
ξ-CONSISTENTLY: f1 texture = f0 texture advected by the per-cell model flow. f0 is SegNet-FREE
(modules.py: SegNet reads x[:,-1] only) → unconstrained; f1 texture SUB-MARGIN in SegNet's null (margin
field #141; measured p50≈0.9 → large amplitude headroom) and in a spatial band that survives R (the
#204-measured bicubic↑874→uint8→bilinear↓512×384 chain: mid-band; amplitude ≥1–2 uint8 steps).
**(D) Class/depth-stratified flow = the correctness channel (few counted bytes):** the advecting flow is
per-Morse-cell: ground cells exact H(ξ) (0 params), sky rotation-only, structure/movable per-cell
inverse-depth (openpilot world-model: road plane + lead depth + 3D lanes; the #365 params re-scoped to
their RIGHT job — making the depicted flow correct where cells differ in depth, not steering outputs).
**(S) The 6-DOF solve = the calibration channel (revived):** on a TEXTURED render the ξ→output Jacobian
becomes well-conditioned (dense flow response) → the per-pair ξ_eff solve gets a real reachable set →
closes PoseNet's residual scene-dependence bias. The A2 machinery is reused as-is.
MECHANISM CLAIM (honest): PoseNet is trained on comma driving data to estimate ego-motion from scenes;
a synthetic scene DEPICTING the same ego-motion densely should elicit ≈ the target 6 outputs. This is
legibility, not adversarial steering. The residual = PoseNet's scene-dependence — measured by the probe,
then calibrated by (S).

## 3. WHY THIS IS CONSISTENT WITH EVERY MEASURED NEGATIVE (no anchor discarded)
- Post-hoc warp of a FIXED FLAT render refuted (70649531f) → this changes WHAT IS RENDERED (adds the
  missing information), the exact door verdict_scope left open ("render co-adapts").
- Pair-consistency law (10.42 mixed) → texture is in BOTH frames, consistent by construction (advected).
- #249 image-space stores rate-prohibitive → texture realization is FREE (seeded generic); only ξ (already
  stored) + few depth scalars are counted.
- Photometric self-fit 2.562 (flow-space) → irrelevant here: we do not match the real pixels/flow; we
  depict the same EGO-MOTION and let the trained estimator read it.
- Symposium L3 proposed null-texture; the ladder now supplies its MISSING mechanism (aperture) + placement
  (margin field) + band (filter chain) + the reason it must compose with (D) and (S).

## 4. THE DECISIVE $0 PROBE (T-alone, before any witness-term build)
On the existing ladder harness (same pairs, same through-R, same positive control): **A0T** = A0 (global
ground-H warp, consistent pair) + seeded sub-margin texture advected by the SAME flow. Measure d_pose vs
A0's 1.685 (n24). Graded ladder: texture amplitude × band sweep (small grid). Secondary: A2T = the 6-DOF
solve ON the textured pair (does the solve revive?). d_seg guard: run SegNet argmax on textured f1 through
R, count flips vs untextured (must be ~0 at sub-margin amplitude — measure, don't assume). Prediction
(labeled PREDICTION): A0T ≪ 1.685 if the aperture diagnosis is right; if A0T ≈ A0, the diagnosis is wrong
and the honest fallback is pose-as-budget-item + #238. Either way decisive.

## 5. THE WITNESS-TERM IMPLICATION (v7.5/v8 — build only on probe GREEN)
If GREEN: the witness gains a **pose-legibility term/stage** — render = argmax-partition (d_seg, unchanged)
+ within-cell seeded texture advected by per-cell ξ-flow (d_pose). v8-native: the per-class decomposition
already gives per-cell support; depth augments each cell (class+depth stratification = the operator's
"more precisely segmented by class and depth"). Curriculum: texture is a LATE/decode-side layer (does not
perturb the d_seg training) OR an in-training term if co-adaptation pays — probe decides. DSL Lever
obligation on build. Rate: ~0 (seed + existing ξ + few depth scalars).

## 6. OPEN QUESTIONS FOR THE DEEP DIVES (fired 2026-07-08)
(a) SCORER-SIDE: what does FastViT-T12/PoseNet actually attend to — receptive fields, luma space-to-depth
band-pass, normalization; is it scene-invariant on its training domain; what texture statistics maximize
flow readability per unit amplitude (motion-perception/optical-flow-observability literature; spectral
band optimal for the measured R chain). (b) OPENPILOT/DOMAIN: exactly how openpilot's models produce and
consume depth (lead distance, road plane, 3D lanes), what per-cell depth we can lift at ~0 bytes; plane+
parallax literature for the minimal sufficient depth per cell. (c) WITNESS-TERM: the Morse-Smale-native
formulation of the texture layer (within-cell = interior of the complex, does not touch separatrices),
interaction with tau-anneal/eikonal terms, v8 coupling.

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — §2 "THE DESIGN — three layers that only work TOGETHER (each alone measured/known insufficient)" names the three layers and records that each was measured alone.
2. **Per-signal decomposition** — §0b decomposes the measured floor per arm (best 2.608 vs A0 1.685 vs A2+ 1.223; per-cell flow 183.5 vs global-H 165.6 on the self-pair) rather than reporting one composite.
3. **Run-to-run diff** — §0b records a matched `s·ξ` sweep monotone toward the flat floor as s->0, which is exactly the run-to-run control that ruled out a convention bug.
4. **Post-hoc query** — the authority surfaces are `upstream/evaluate.py` and `modules.py` (frozen CPU-torch PoseNet); the measurements are cited by commit (`c2adba9aa`, `7d2784fc9`).
5. **Cite-chain** — §3 "WHY THIS IS CONSISTENT WITH EVERY MEASURED NEGATIVE (no anchor discarded)" is the explicit reconciliation chain across prior anchors.
6. **Counterfactual hooks** — §4 "THE DECISIVE $0 PROBE (T-alone, before any witness-term build)" is the pre-registered discriminator, and §5's GREEN gate is recorded as having returned RED.

**Scope honesty (from the memo's own §0a/§0b):** the aperture hypothesis is FALSIFIED and A1T is ALSO falsified; `verdict_scope: formulation`. This memo is retained as a superseded design plus its measured refutation, not as a live design.
