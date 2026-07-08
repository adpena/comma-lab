# THE POSE-LEGIBLE WITNESS — aperture-problem diagnosis + ξ-consistent textured class/depth-stratified render (2026-07-08)

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
