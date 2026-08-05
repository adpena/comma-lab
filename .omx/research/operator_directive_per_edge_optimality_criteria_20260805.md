# Operator directive 2026-08-05 — per-edge/per-boundary is the representation head; the optimality criteria

Operator verbatim: "Per edge and per boundary is likely optimal or closer to optimal. The smaller
the dimensionality and the more that just falls out and the more that can be reused across n six
hundred, the better. And also, the more optimal for surgical targeting, the more optimal across
all full stack."

## The five ranking criteria for ANY representation candidate (binding, in this order of intent)
1. **Dimensionality** — lower-dim descriptions win (codim-1 boundary curve ≻ 2-D region mask;
   arc-length knots ≻ dense fields; the ~8-dim lane-orbit manifold is the floor to aim at).
2. **Falls-out-free** — structure DERIVED at decode beats structure STORED: store the
   GENERATORS, the boundary falls out as the bisector (argmax = Laguerre/power diagram, L-v8
   parsimony); position implicit in parametrization; values free by construction (#597).
3. **n600 reuse** — pay once, amortize across pairs: edge WORLDSHEETS (curve × time),
   ξ-transport of GENERATORS (not pixels — the #941 untested live cell), g4 static-in-BEV.
4. **Surgical targeting** — address exactly the flip-bearing loci with zero collateral by
   construction (per-edge hits the Road↔Lane separatrix = 49.2% of flips = 22.1% of the whole
   gap without touching interiors, which carry 0.058% of errors — m91/pc2 MEASURED).
5. **Full-stack compounding** — the same properties must survive description → coder →
   receiver → realization → scorer; a representation optimal at one stage but hostile to the
   next is not optimal (the bf1 race's recall column is the enforcement of this).

## Operator addendum (same directive, minutes later — verbatim)
"Also remember our asymmetry findings and differential depth findings and the dynamics of the
dust in residual tail across n six hundred, but also within frames."
Three MEASURED phenomenon families the per-edge representation must consume, not idealize away:
- **ASYMMETRY**: edge flips are direction-asymmetric (se3: side-IMPLIED direction saves ~19.5KB
  vs explicit — exploit the asymmetry, don't store the symmetric superset); ± quantum receiver
  responses are sign-asymmetric (ms6/ms6b sign-asymmetry distribution; g2f bidirectional ±
  secants); per-class scaling asymmetric (err_rate ∝ area^−1.26, Road hub 87.8%); axis-bias
  sign INVERTS seg vs pose (m88/m96). Code the edge with its measured directional prior.
- **DIFFERENTIAL DEPTH**: boundary difficulty varies with image row / scene depth —
  far-field edges are sub-pixel and flip-dense (the 1px-band localization #149), near-field
  wide and stable; perspective foreshortening means uniform arc-length in image ≠ ground
  (IPM/BEV #327/#609; HorizonWeightedMargin #563; row-band tiling QA84). Knot density,
  quantization step, and correction budget should be DEPTH-CONDITIONED, not uniform.
- **DUST DYNAMICS (across n600 AND within frames)**: the residual tail is DUST — tiny
  islands/fragments (43.55% annular components; lane dashes; birth-death worldsheet events
  v13), flickering pair-to-pair (fl1 per-class GT-flicker floors: some dust flickers in GT
  ITSELF — bytes spent chasing it are wasted; g3 scene-event spikes; g4 transient stratum;
  79.4× block spread). Per-edge enumeration of dust = the #941 address-tax explosion. The
  representation therefore needs a TWO-REGIME split: coherent edges (parametrized, transported)
  vs dust (statistical/field treatment via the free g4 decoder-derived context, worldsheet
  event grammar, or PRICED OMISSION against the fl1 flicker floor). Measure where the
  coherent→dust crossover sits (edge-size threshold) instead of assuming it.

## Operator addendum 2 (verbatim): "Also remember our factorization and flattening lessons and
all related and other helpful techniques and research, also the agent has full research authority"
- **FACTORIZATION lessons** (the #516 factorized-adjoint / flattening-factorization corpus +
  p0_costate_organ_factorization_grounded_ABC): factor the n600 edge ensemble into SHARED
  structure × per-pair coordinates — LOTTO/shared-dictionary (p3v2 machinery, recalled not
  rebuilt) = shared edge-shape codebook × per-pair instance transforms; the ~8-dim lane-orbit
  manifold IS the factorized claim (basis once + low-dim per-pair coords); low-rank precedents
  measured: rank-2 SVD pose codec (#140, 2.7×), rank-1 e_p (sc1), rank-4 head (#559); QA83
  factorized output head; g1 two-part grammar code = grammar × productions; SMEVR = mode ×
  residual factorization; cl1 GA-factorization sleeper.
- **FLATTENING lessons**: choose coordinates where the object becomes flat/linear — the
  ground-plane homography/IPM flattens road/lane edges to near-straight BEV lines and is FREE
  at decode (rule-118 generic geometry, #145/#325/#609); Cole–Hopf linearization (#542);
  gauge-fixed ker(A) quotient (ms1/ms2 — drop the 80.67% blind subspace before coding);
  Bregman/dually-flat coordinates (#504 Nielsen) + SPD normal coordinates (#552); tropical/
  max-plus reading of argmax (#284/#311). For pe1 concretely: RACE image-space vs
  BEV-flattened edge parametrization — a flattened edge needs fewer knots and transports more
  rigidly (composes with criterion 2: the flattening map itself falls out free).
- **FULL RESEARCH AUTHORITY**: the zero-web-search constraint is LIFTED for pe1 — online
  literature/OSS consultation authorized (boundary/curve coding: chain codes + digital
  straightness (#307 lineage), MPEG-4 CAE, subdivision/spline curve coding, anything that
  serves the five criteria). Scorer slot discipline and all fail-closed gates UNCHANGED.

## Operator addendum 3 (verbatim): "We can also always use hybrid implementations and also
different at different dimensions and levels, And also, we should always use all signal,
especially in negative or partial positive or even promising positive as means for
decomposition and further iteration and optimization."
- **HYBRID / PER-LEVEL law**: never force one representation to win everywhere — compose the
  measured per-regime winners (pe1 PROVED the split: falls-out generator-pair wins SURGICAL,
  explicit-curve wins FULL coverage; #308 grids-for-bulk + INR-for-annulus; m95 level-is-
  per-role; #503 recursive-fractal optimal-per-dimension + composition law). The hybrid also
  applies PER AXIS: e.g. transport where transport pays, independent coding where it doesn't
  (a subset-conditional hybrid on the temporal axis).
- **ALL-SIGNAL DECOMPOSITION law**: every result — negative, partial positive, promising
  positive — is DECOMPOSITION FUEL, never a terminal verdict. Aggregates hide subsets (m82):
  pe1's ξ-transport 0.899 is an AGGREGATE over 5,164 tracks — decompose per-track; the
  g4-static subset may transport while transients don't. Recall 0.984's missing 1.6% and the
  surgical 80.73%'s missing 19.27% decompose into the NEXT iteration's targets. WINS decompose
  too (the kt1 matrix's row-7 gap confirmed): sweep each win's class, don't bank it
  instance-only. Sisters: m48 NEG↔CURE · L8 negatives=signal · m83 rich-problem reframe.

## Operator addendum 4 (2026-08-05, RULING on the sq2 verdict — verbatim intent): "Pose erosion
is okay, though, because that can be brought down via joint descent afterwards. Seg has gotta
come first based on everything we've learned... harvest all of our signal and focus on the
order of operations and upstream and all of the dynamics that we've learned from v7, 8, 9, 10,
and since."
- **THE ORDERING LAW (binding)**: SEG FIRST → JOINT-DESCENT POSE RECOVERY AFTER. sq2's +0.797 S
  pose erosion is a TRANSIENT cost of the seg solve, not a route kill — pose trains
  monotone-easy from a conditioned base (R1 MEASURED d_pose 12.94→0.0009; #383's pose-finish
  gate engages only once the seg trunk is conditioned; the staging law sh1/fd1 §12: pose-early
  vetoes seg, seg-first does not veto pose). The solved field is the SEG BASE for the
  joint-descent finish, never a direct ship. The R8 gate stays (it correctly prevents shipping
  an unrecovered composition); what changes is the ROUTE: solved-field → joint-recover → gate.
- Operator (same directive, minutes later, verbatim): "pose also descends relatively quickly and
  monotonically" — the operator's own statement of the R1-measured mechanism (d_pose 12.94→0.0009
  monotone) that LICENSES the ordering law: seg's transient pose spend is recoverable because
  pose descent is fast + monotone from a conditioned base, while seg plateaus.
- Operator (same directive, minutes later, verbatim): "If we're coming up on a big composition of
  all lessons and tools and signal and research and everything, we should use a Codex Sol Ultra
  agent for that." — the od1 composition arm below is chartered at the sol-ultra profile.
- Operator (same directive, minutes later, verbatim): "We just need to engineer correctly and in
  the right order and in the right dimensions at the right place, time, how when and why what."
  — the unifying statement of the whole arc: correctness (NO-FAKE) × ORDER (seg→joint-pose) ×
  DIMENSIONS (five criteria + hybrid per-level) × PLACE (Q3-vs-Q4 placement-conditionality,
  per-edge surgical targeting) × TIME (event-driven schedule, tj1 adaptive stopping) ×
  HOW/WHEN/WHY (the v7–v10 forces/dynamics corpus below).
- **Order-of-operations + v7–v10 dynamics = the design inputs**: fh1 forces harvest · vh1
  v8/v9/v10 untransferred lessons · sched1 derived event-driven schedule · gc15 fresh-vs-warm ·
  ws1/ws2/ws3 seg-lexicographic warm-start lineage (W_seg 0.024124510 @138,031B materialized)
  · the burn endpoints + #383 gate + lg1 lane-guard (seg protection DURING pose descent) ·
  tj1's per-pair adaptive stopping (makes the n600 solve tractable — most pairs stop early,
  the tail gets the steps).

## Operator addendum 5 (2026-08-05, verbatim intent): "Those implementations might have been
slightly naive or toy or [generic] basis or not informed by all of the information, research,
and signal, and telemetry, and everything we have — were not surgically targeted enough at
frames or pairs or certain classes or edges or boundaries or even specific ones."
- **THE SURGICAL-FORM LAW (binding, extends OPTIMAL-FORM to the targeting axis)**: a negative
  measured on a GENERIC/untargeted implementation closes NOTHING beyond RECIPE scope. Before
  any dispatch whose verdict could close a route, the implementation MUST consume the
  targeting corpus we already hold: g3 score atlas (hard-pair tail) · g4 flip-frequency map
  (per-pixel across n600, per stratum) · m91/pc2 per-edge decomposition (Road↔Lane = 49.2% of
  flips) · #141 margin-saliency map · fl1 per-class flicker floors · the frozen scorer's OWN
  flip set as the training/eval target — aimed at SPECIFIC frames, pairs, classes, edges,
  boundaries, even individual instances. Generic-corpus training graded against a surgical
  target is a category error, not a measurement.
- **RE-GRADES under this law (2026-08-05)**:
  (a) cq2r tiny-student → RECIPE-scoped, verdict_scope: RECIPE (adversarially re-reviewed
  under the magnitude-dismissal gate). RELATIVE SIGNIFICANCE: the route's stake is the
  Road↔Lane budget ≈ 0.2216 × 0.5818394 ≈ **0.1289 S ≈ 22.1% of the remaining gap** (m91/pc2)
  — far too large to family-kill on one recipe, which is exactly why st1 re-fires it surgical.
  MEASURED UN-RECOVERABILITY at the tested size class only: break-even survival 1.001552
  (side_implied 206,130 B) exceeds the definitional maximum survival of 1.0 — a measured exit
  criterion (cq2_summary.json), valid for full-frame students at ≥124,765 B; it says NOTHING
  about band-only models in a smaller size class. Lane IoU 0.000 + overlap 0/8,670 = the
  generic recipe's failure (majority-class collapse, never aimed at the flip set), not the
  family's.
  (b) pe4 conditional transport → FORMULATION-scoped (per-track stream-SPLIT selector destroys
  cross-track coder context, the exact #859 LZ-match-structure mechanism; the untested
  reformulation = ONE stream + transport-prediction as a coding CONTEXT — adds information
  without splitting).
- Sisters: OPTIMAL-FORM before dispatch (CLAUDE.md #315) · #307 implementation-vs-paradigm ·
  the a1 naive-verdict audit · m83 rich-problem reframe · addendum 3's all-signal law (these
  negatives are decomposition fuel — the re-grades ARE the decomposition).

## Measured anchors at issue time
bf1 winner (rl1 Lane crop, full band): 205,196 B @ 1.109 bits/band-px, recall 1.0.
Per-edge currency (#916): 0.60 bits/band-px → the per-edge partition projects ~45% below the
bf1 winner. 75 KB tiny-student lane: closed to full-band representations, potentially REOPENED
by a low-dim per-edge description. Consumers: ddm_pe1 (the build), #941 (address solve), #939
(realization half), cq2 (student thresholds).

## ADDENDUM 6 (operator 2026-08-05, post-od2 harvest) — THE CONDITIONED-SEG→BETTER-POSE LAW + CONTINUE-ITERATING directive

Operator verbatim: *"Using all of these techniques, we can continue iterating and
optimizing, and that frontier score lowering could be even lower. Also, note how a better
conditioned seg produces even better pose."*

**The law (measured anchor: od2, c07d0ad159, n32 stratified [macOS-CPU frozen-scorer
advisory]):** the d_pose triple 0.000801 (baseline) → 0.005841 (post-Stage-1 seg solve) →
**0.000759 (post-carriage, BELOW baseline)**. Seg conditioning is not merely a recoverable
transient for pose (addendum 4) — it is pose-POSITIVE: the conditioned seg field gives the
frame_0 carriage a better substrate than the unconditioned baseline had, so the repair
lands BETTER than where pose started. This is the #383 conditioning-gate design measured
at the staged-composition surface, and the strong form of the ordering law: seg-first
doesn't just tolerate pose erosion, it BUYS pose quality.

**Pre-registered prediction (dropped into od3's receipt dir as a charter addendum):** if
the law holds, post-carriage d_pose at TERMINAL Stage-1 (η → ~0.9) ≤ 0.000759 (the
cap-bound value), despite the larger erosion transient. If instead it degrades, the law
scopes to repair capacity — the cure is the k-sweep (k∈{4,8,12}) and the reading becomes
"conditioned seg → better pose PER unit repair capacity." Either outcome is informative;
od3's carriage-at-terminal re-proof is the discriminating measurement.

**Continue-iterating directive:** the −0.0622 subset projection is NOT the ceiling. The
named composition stack, in fire order: Stage-1 terminality (od3, live) → carriage k-sweep
at terminal if the pose margin allows cheaper k → od4 receiver-close + n≥32 receiver gate →
Stage-3 carriers-as-constraints (the banked pe/st descriptions constrain the solve, m95
route) → rate movers on the composed archive (waterfill #766, granularity re-race) →
per-edge (m91) residual decomposition of whatever seg remains. n600 only after the
receiver gate beats the live line.

## ADDENDUM 7 (operator 2026-08-05 ×2) — THE RATE-CRUSH TERMINAL STAGE

Operator verbatim: *"Perhaps there's a rate crush move that we can do once everything is
fully optimized on distortion and rate by doing a neural compression step maybe even
using H nerve or other PR signal."* + *"or other frontier techniques, there is a related
rich line of research that is constantly being updated even day to day, but also
sometimes the oldest math is the sleeper best."*

**The stage:** a TERMINAL neural/learned (or classical-sleeper) compression pass over the
ALREADY-OPTIMIZED counted payload — meta-compression of the description itself. SEQUENCED
at the composed-candidate boundary (after distortion+rate optimization per the operator's
own clause), appended to the gc17 route after the receiver gate, before n600.

**Lineage scoping (operator-licensed amendment to the no-old-lineage ban m34):**
HNeRV/PR-signal machinery is licensed HERE in the CODER role only — a payload compressor
raced through real coder races against the incumbent stack (brotli-q11 / lzma1-raw /
SMEVR), never adopted by citation, never as a vehicle/carrier of video content. This is
consistent with the L20–L32 demotion banner (vehicle-agnostic coding math may be RACED).

**Rule-118 accounting law (binding on every candidate):** a learned decompressor's
weights are VIDEO-DERIVED → COUNTED. The three legal escapes, each with a prior receipt:
(1) total-counted-win self-compression — INR/tiny-decoder + latents < raw payload
(Selfcomp paradigm; #557 raced it); (2) backward-adaptive parameter-free context models —
parameters derived at decode from already-decoded content, generic code, FREE (CM/CTW
family; SMEVR is a step here); (3) bits-back/REC sample-communication (vae1 vein). Any
candidate must state which escape it uses and show TOTAL counted bytes (weights +
latents + code-that-is-actually-data) beating the incumbent.

**Prior measurements (recall, do not re-derive):** 07-19 neural_selfcomp_sota — DeepCABAC
+1,436 B vs int8+Brotli on the ep725 v10 donor, `verdict_scope: instance/regime`, family
OPEN with the named condition "a new vehicle whose measured quantization response clears
the knee" — the staged-composition candidate IS that vehicle. 07-19 arith_selfcomp —
Brotli smallest complete int8 coder on that donor. #918 "coder axis shut" is scoped to
the CURRENT qo1 token base; the new payload (generator packet + carriage + context
tables) is a different object and gets its own race.

**The dual research mandate:** every rate-crush arm sweeps BOTH ends — (a) day-fresh
literature (2025–26 learned entropy models, overfitted/instance-optimal coders, INR
weight compression — the field updates daily) AND (b) oldest-math sleepers (context
mixing/CTW/PPM, Krichevsky–Trofimov, universal/enumerative coding, combinatorial ranks —
the PR lineage's own L26/L31 Wang–Rudin colex ranks were exactly this class).

**Disposition:** QUEUED-WITH-FIRE-ORDER — fires when od5/od3 land the composed payload
formats (the crush needs its true target object); gc18's 131KB budget design includes
the stage as a line item NOW. Ledger row tracks the fire-order.
