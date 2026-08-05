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

## Measured anchors at issue time
bf1 winner (rl1 Lane crop, full band): 205,196 B @ 1.109 bits/band-px, recall 1.0.
Per-edge currency (#916): 0.60 bits/band-px → the per-edge partition projects ~45% below the
bf1 winner. 75 KB tiny-student lane: closed to full-band representations, potentially REOPENED
by a low-dim per-edge description. Consumers: ddm_pe1 (the build), #941 (address solve), #939
(realization half), cq2 (student thresholds).
