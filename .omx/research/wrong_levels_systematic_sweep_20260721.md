# Wrong-levels systematic sweep — where else we operate/represent/act at the wrong level (2026-07-21)

## THE BINDING PRINCIPLE (operator 2026-07-21, verbatim-grounded) — supersedes the "levels" framing
> "Meet everything precisely where it is, and model and represent and carry the THING ITSELF in the
> smallest representation and basis and everything possible, across n600 temporal."

Three clauses, all binding on every representation/description decision going forward:
1. **MEET IT WHERE IT IS** — represent each object/stratum in ITS OWN natural coordinates, level, and basis
   (ground-frame lanes as static polynomials; ego-motion as an SE(3) ξ curve; topology as sparse Morse
   events; the argmax as its rank-4 subspace). Never in the ambient/pixel/weight embedding.
2. **CARRY THE THING ITSELF, SMALLEST BASIS** — code the intrinsic object (not a proxy, not an ambient
   embedding, not an arbitrary gauge member) in the minimal matched basis — the Kolmogorov-minimal program
   in a grammar whose primitives ARE the object's operations. Smallest representation = the thing itself
   expressed once in its own alphabet.
3. **TEMPORALLY UNIFIED ACROSS n600** — carry the ONE worldsheet object across the FULL 600-pair temporal
   extent as a single temporally-coherent representation (static geometry + ξ-curve + events), NOT 600
   independent slices. The temporal dimension is spanned coherently by the basis, not paid per-frame.

This is the unified statement of the RATE axis (the only open axis — seg+pose solved). Every byte paid for
an embedding, a proxy, a gauge member, a per-frame slice, or an ambient coordinate is waste against this
principle. The sweep below is the catalog of where we still violate it.


**Source:** operator 2026-07-21 "what have we overlooked … based on the deep math / geometry / worldsheet
model / dynamical system" + "the inverse-chart solve is inherently smaller if we design our own
syntax/grammar/ops representation" + "where else is our wrong levels? apply to that we have not applied it
to yet." Sister of [[act_at_the_highest_causal_level_pixels_are_last_resort_20260721]] +
[[master-thesis-invert-frozen-space-hybrid-formulation-realization-completeness]] +
`feedback_seg_pose_distortion_already_solved_rate_is_only_open_axis_20260720`.

## The one-sentence root cause
The object lives on a LOW-DIM CHART — {static ground-frame lane polynomials} × {SE(3) ξ(t) trajectory
curve} × {sparse Morse events}, read through a frozen rank-4 SegNet head. We keep describing / training /
quantizing / coding / storing in the AMBIENT representation (image-frame pixels, full 512×384 scorer plane,
600 independent per-frame/per-pair slices, raw weight space) and PAY FOR THE EMBEDDING. Rate is the only
open axis (seg+pose solved: d_seg 1.52e-4 / d_pose 1.02e-4). Every byte we pay for the embedding is waste.

## PART A — overlooked STRUCTURE (from the deep-math / worldsheet / dynamical-system view)
1. **Worldsheet is a SURFACE, likely DEVELOPABLE.** The boundary sweeps a codim-1 surface in (x,y,t);
   under rigid ξ it is ruled. If Gaussian curvature K≈0 ($0 measurable), description collapses 2D→1D:
   one directrix curve + ξ ruling (solved) + bending. #574 (ξ-keyed delta) is the WEAK form; the strong
   form codes the surface's intrinsic 1D chart. UNMEASURED.
2. **Wrong COORDINATE FRAME (likely biggest).** Image-frame boundary = messy homographic warp of static
   ground geometry. In the ground/BEV frame (we HAVE the homography, #325/#327) lanes are STATIC
   polynomials and ego-motion is a rigid 2D screw. Road/lane mass (~61% of the necessity band) = static
   ground coeffs stored ONCE + ξ(t) (solved) → image boundary DERIVED FREE through the known homography.
   We used the lane prior for training/survival, NEVER as the describe-line substrate.
3. **Seed codes the AMBIENT plane, not the rank-4 subspace.** SegNet head is EXACT rank-4 over a 144-dim
   quotient; only margin DOF near boundaries flip argmax. Code the seed's rank-4-relevant projection;
   U2's Kolmogorov bound against THAT intrinsic dim, not the 78,969-B full plane.
4. **Code the GENERATOR, not the STATE.** The level-set velocity field is DERIVED from the FROZEN scorer
   margin → the flow generator is nearly FREE (tiny IC + known frozen scorer). We pay to store trajectory
   states. Describe (IC + field + time), integrate at decode.
5. **Events are SPARSE (Morse).** Between births/deaths the worldsheet is smooth rigid transport (free
   from ξ). Code smooth-transport-segments + a sparse event list, never per-frame. #597 is the live home.

## PART B — where else the WRONG-LEVEL principle applies (NOT yet applied)
- **B1 · TRAINING GRADIENT level.** We backprop PIXEL losses into WEIGHT space. Higher level: flow the
  training signal to the CHART (lane coeffs / boundary knots / event params) so optimization lives in the
  low-dim chart space, not pixel/weight space → faster convergence AND chart-native descriptions by
  construction. UNAPPLIED.
- **B2 · QUANTIZATION level (generalize g2g).** g2g proved chart-symbols survive uint8 where sparse pixels
  die. Generalize: NEVER quantize at the pixel level for description — quantize CHART coefficients and
  rasterize (coherent survivable pixels). Apply specifically to the SCORER-PLANE SEED (currently
  pixel-quantized): quantize its rank-4-relevant coefficients instead. UNAPPLIED to the seed.
- **B3 · POSE STORAGE level.** Store ξ as a few SE(3) B-spline KNOTS (the trajectory curve), not 600×6
  per-pair vectors — ego-motion is smooth; per-pair is the wrong level. #574 IN FLIGHT — confirm it is
  curve-level not delta-per-pair.
- **B4 · RATE-ACCOUNTING / CODER level.** Code each semantic STRATUM (road / lane / movable / events) in
  its OWN matched-context grammar, not one aggregate arithmetic stream. Rate is accounted at the
  byte-stream level; the causal level is per-stratum. This IS the ops-grammar (below) applied to coding.
- **B5 · GAUGE level applied to the SEED.** Render has 80.67% nullity + ~52% gauge (scorer-invisible,
  #519/#553/#580). mdl freed the camera MEMBER; but the SEED itself may still carry gauge/blind-subspace
  DOF. Describe the seed's gauge-fixed, range(A)-restricted representative — code only scorer-VISIBLE DOF.
- **Partially-applied (in flight, named for completeness):** CURRICULUM = continuation-path level vs
  discrete stages (#302/#344); OPTIMIZER = parameter-manifold vs Euclidean weight space (#556 pending);
  EVENTS = topological-event detection vs per-frame diff (#597); DISPATCH = decisive-dimension vs
  single-rung ([[spec-decisive-dim-arm-not-ladder-single-to-joint]], meta-level).

## PART C — the CUSTOM OPS-GRAMMAR (the operator's lead; Kolmogorov-optimal by construction)
Generic coders (zlib/#557) code the ambient rep and pay for the embedding. A grammar whose PRIMITIVES are
the solve's own ops — `LANE(ground_coeffs)`, `TRANSPORT-by-ξ(knots)`, `SPAWN-event(t,type,loc)`,
`HOMOGRAPHY-to-image(params)`, `RASTER` — codes each intrinsic DOF ONCE; one op-token expands to megabytes
in the FREE inflate.py interpreter. This is Kolmogorov-optimal by primitive-matching (shortest program in a
language matched to the object). The archive payload IS this program ("the inverse chart as a program, not
data"). U5 should be this ops-grammar, not a generic coder on the plane.

## THE DECISIVE $0 PROBE (ties A2+A3 and unlocks C)
Measure, on the n64 → n600 road/lane stratum: (i) BEV-staticity — transform boundaries to the ground frame,
are road/lane boundaries (near-)static across frames? (ii) worldsheet developability — is K≈0 under ξ? If
BOTH hold, the describe line for the majority mass collapses to `(static ground coeffs + ξ-curve knots +
sparse events)` expanded through a free homography, and the ops-grammar (C) is literally that payload.
Prioritize this before more seed-coder tuning.

## Honesty / scope
A1/A2/B-family are HYPOTHESES gated on $0 measurement. Movable objects (cars) break developability and
ground-staticity; real lanes have dashes/forks/curvature changes that aren't perfectly static polynomials —
so scope the collapse to the static road/lane stratum (the majority necessity mass), measure per-stratum,
never claim family-wide off one stratum. Triality: DAG leg landed (this memo + FEED); DSL leg pending
(these become Levers only once a trainer/receiver flag exists); equations leg FORMALIZATION_PENDING (register
the developability-collapse + grammar-Kolmogorov laws only AFTER the $0 probe measures them). Pointer 0.19108
[contest-CPU] UNMOVED — all of the above is means, no score claim.
