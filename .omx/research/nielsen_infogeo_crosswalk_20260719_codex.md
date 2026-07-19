# Nielsen information geometry x the flattened KKT campaign (#550)

**Date:** 2026-07-19

**Lane:** `nielsen_infogeo_crosswalk_20260719`

**Posture:** `research_only=true`; research and routing only

**Verdict scope:** corpus-to-campaign mathematical crosswalk, not an implementation,
receiver, archive, score, launch, promotion, or family verdict

## Outcome first

**Pointer delta: exactly zero.** The pointer remains
`0.1910828242 [contest-CPU Linux x86_64]`. No archive was built, no scorer was
run, and no launch or paid dispatch was authorized.

**One-line verdict:** Nielsen's corpus makes a smaller #539 channel-target
description algebraically credible and supplies useful prioritizers for #536/#541,
but it does not close the nonlinear spatial pullback, receiver arithmetic, or the
empirical bytes-versus-`d_seg` secant; therefore none of the literature is
promotion-authorizing.

The highest-value concrete result is a **DERIVED** description audit of the
existing 338-byte `PDW1` target. Its exact layout is

```
12 header + 10 class ids + 80 sites + 20 weights + 36 edges
+ 144 tie normals + 36 tie offsets = 338 bytes.
```

The 180 tie-locus bytes are deterministic functions of the sites, weights, and
edges. A new generator-only float32 packet retaining the current header, class
ids, sites, weights, and nine observed edges is therefore a **DERIVED 158-byte
construction** before compression. Encoding the five class scores relative to a
fixed reference-class affine gauge uses `(K-1)(d+1)=20` float32 scalars rather
than 25 site/weight scalars, giving a **DERIVED 138-byte margin-preserving
construction** with the same header/ids/edges. If only the argmax partition is
required, fixing one additional positive common scale leaves 19 continuous
degrees of freedom and a **DERIVED 134-byte partition-only construction**.

These are format constructions, not measured packet bytes, lower bounds, or
receiver savings. They do not repair the missing spatial/RGB inverse. The current
338-byte `PDW1` remains the only **MEASURED** byte-close target.

## Evidence language and authority boundary

- **MEASURED** means a cited local artifact/result already on disk. No new
  empirical measurement was made in this research-only arm.
- **DERIVED** means algebra from the pinned score law, local packet layout, or a
  cited mathematical identity.
- **INFERRED** means a campaign hypothesis transferred from the literature and
  requiring the named $0 gate.
- **UNKNOWN/PENDING** means the required receiver, held-out result, or sibling
  measurement is not yet available.

The contest-law break-even is independent of the pending sibling secant. Holding
Pose fixed, a `1e-6` reduction in `d_seg` is worth

```
100 * 1e-6 = 1e-4 score units,
B_break_even = 1e-4 * 37,545,489 / 25 = 150.181956 bytes.
```

Thus **150.181956 B per `1e-6 d_seg` is DERIVED** from the canonical score
formula as a continuous indifference point. Because an archive changes by
integer bytes, exactly `1e-6` of Seg improvement can pay for at most 150 added
bytes and still improve the score; 151 bytes requires a larger Seg gain. What
remains **UNKNOWN/PENDING** from sibling
`seg_secant_rd_curve_20260719` is the achievable receiver-closed secant
`Delta bytes / Delta d_seg`. A Bregman radius has neither unit and must not be
substituted for that measurement.

This memo follows [`docs/operating_manual_craft_handoff.md`](../../docs/operating_manual_craft_handoff.md):
artifact caveats travel with numbers, negatives retain their narrow scope, and
the pointer is reported before the means.

## Corpus sweep and triage

The allowed corpus surfaces were swept, including the homepage, reverse
chronological publications, information-geometry tutorials, Bregman portal,
software, and visual/poster material.

| corpus surface | material harvested | campaign disposition |
|---|---|---|
| [Homepage](https://franknielsen.github.io/) and [publications](https://franknielsen.github.io/npublications.html) | Bregman Voronoi, information projections, Chernoff information, centroids, enclosing balls, Fisher-Rao, recent curved/generalized-Legendre work | Primary index; individual claims below cite the paper, not the index |
| [Bregman-divergence portal](https://franknielsen.github.io/BregmanDivergence/) | first/second-type geometry, duality, Voronoi and clustering references | Adopt mathematical identities only |
| [Information-geometry hub](https://franknielsen.github.io/IG/) and [elementary tutorial](https://arxiv.org/abs/1808.08271) | natural/expectation coordinates, Hessian metric, dually flat connections, Bregman projections | Directly relevant to the dual-metric interpretation |
| [Selected software](https://franknielsen.github.io/software.html), [pyBregMan site](https://franknielsen.github.io/pyBregMan/index.html), and [paper](https://arxiv.org/abs/2408.04175) | reference implementations for manifolds, geodesics, bisectors, centroids, balls, and Chernoff points | Reference oracle/visualization only; no production or inflate dependency |
| [Slides, lectures, and videos](https://franknielsen.github.io/SlidesVideo/index.html) | diagrams and the [smallest-enclosing-Bregman-ball poster](https://franknielsen.github.io/SlidesVideo/BCBregmanBalls-ECML-2005.pdf) | Diagnostic/farthest-first hypothesis only |
| [Curved representational Bregman divergences](https://arxiv.org/abs/2504.05654) | constrained representation as projection of a full-space object | Screened; lower priority than the existing hard-oracle trust-region gate |
| [Generalized Legendre transforms](https://arxiv.org/abs/2507.20577) | affine-deformed convex conjugacy and dual Hessian structure | Screened; the paper is not a V9 runtime gauge-covariance receipt and does not close the local implementation-custody gap |

## Settled local facts that constrain adoption

| local fact | label and consequence |
|---|---|
| Frozen Seg head is a 5-class affine map over a 4-dimensional quotient; singular values `3.128, 2.154, 2.025, 1.796, 0` | **MEASURED.** Power-diagram geometry is exact in the channel quotient, not in image space. |
| The n600 `PDW1` target is 338 raw bytes and byte-close; nine of ten class pairs are observed as four-neighbor adjacencies | **MEASURED.** The class graph is almost complete, so class-edge sparsification has little room. |
| Real-prefix fitted target stopped at frame 195 on receiver-arithmetic disagreement; frames 0..194 had fitted feature-pullback `d_seg=0.023459097...`; fitted target was 314 raw/306 Brotli bytes | **MEASURED at the stated prefix/feature-pullback scope.** A tiny target does not imply a spatial witness. |
| Positive-band n24 curves: repairs `9,9,41,40`; all 96 admitted rows had `d_seg=0`; the linear Pose proposer was active 4/96 while hard Pose was inactive 96/96 | **MEASURED.** There is no Seg secant in these four rows; hard repair traffic grows while the admitted endpoint remains flat. |
| Lowest positive-band range-coordinate point was 1,474,579.92 Brotli bytes/pair, 37.6073% below its same-scale tiny-`tau` calibration | **MEASURED range-coordinate residual only.** It is not archive bytes. |
| Same gradient pair: Euclidean cosine `-0.00105`, Fisher cosine `+0.0435`; relative norm `0.00186` Euclidean / `5.9e-5` Fisher | **MEASURED.** The sign flips, while the force is sub-dominant under both readings. |
| The current continuous solver analytically projects each pixel's single RGB half-space and uses Dykstra only between the combined Seg/box set and optional rank-6 Pose ellipsoid | **VERIFIED BY SOURCE INSPECTION.** The reported hard-oracle `repairs` are not Dykstra iterations. |

## Ranked adoption crosswalk

### 1. Gauge-fixed, generator-only power packet for #539 — `ADOPT_AS_$0_FORMAT_PROBE`

**Literature basis.** Nielsen, Boissonnat, and Nock's
[Bregman Voronoi paper](https://arxiv.org/abs/0709.2196) distinguishes
first-type affine cells from second-type curved cells, relates them through
Legendre duality, and identifies weighted Bregman diagrams with affine/power
diagrams in the appropriate coordinates. Our frozen quadratic/head instance is
the simple first-type case.

For affine scores in `d` dimensions,

```
l_i(z) = a_i^T z + b_i,
s_i = a_i / 2,
omega_i = b_i + ||s_i||^2,
l_i(z) = -[||z-s_i||^2 - omega_i] + ||z||^2.
```

The argmax therefore equals a weighted-site power partition. Adding the same
affine function `u^T z + v` to every score changes neither pairwise score
differences nor cells. This removes `d+1` continuous degrees of freedom. For
`K=5,d=4`, margin-preserving description complexity is at most
`(K-1)(d+1)=20` scalars. If only cell labels matter, a positive common scale is
also immaterial, leaving 19 partition degrees of freedom for non-degenerate
instances. These are **DERIVED upper constructions**, not a proof of a universal
information-theoretic minimum.

**Named consumer.** Task #539 and
`src/tac/boundary_math/power_diagram_witness.py`, as an additive candidate
packet version; the existing `PDW1` remains unchanged.

**$0 falsifiable gate.** Build only a new local packet fixture that:

1. encodes relative affine coefficients in a fixed reference-class gauge;
2. derives tie normals/offsets after parse-back using explicitly declared
   float32 arithmetic rather than shipping their 180 redundant bytes;
3. reproduces every existing ordinary parity fixture and the exact frame-195
   near-tie diagnostic under the declared receiver arithmetic;
4. is encode/decode/re-encode byte-identical and rejects every malformed/trailing
   byte case already covered by `PDW1`; and
5. remains `TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT` until a legal spatial/RGB
   receiver is parsed back and measured through `R`.

**Adoption threshold.** The raw candidate must be at most 138 bytes without
lossy coefficient quantization and must preserve the full margin-bearing score
differences. The 134-byte scale-normalized variant is eligible only for
partition-only consumers. Float16 arithmetic is not admitted merely because it
would halve the coefficient bytes. Even a 200-byte target saving (338 to 138)
is only rate-equivalent to about `1.3317e-6 d_seg` at the fixed score law, and
that equivalence becomes meaningful only when the target is inside a complete
receiver-closed archive.

**Stop rule.** If the frame-195 case cannot be reproduced without a video-
specific exception, or the spatial receiver remains absent, do not optimize the
packet further. The open rate axis is the feature-field/RGB pullback, not the
last few certificate bytes.

### 2. Dual-coordinate calibration for #536 — `ADOPT_AS_MEASUREMENT_SCHEMA`

**Literature basis.** In Nielsen's
[information-geometry tutorial](https://arxiv.org/abs/1808.08271), a regular
exponential family has natural coordinate `theta`, expectation coordinate
`eta=nabla F(theta)`, Fisher/Hessian metric `H=nabla^2 F`, and Legendre-dual
metric `H^{-1}`. Locally,

```
deta = H dtheta,
dtheta^T H dtheta = deta^T H^{-1} deta.
```

Natural and expectation coordinates are two coordinate descriptions of the
same local statistical geometry; they are not two independent contestants for
the physical byte price.

**Campaign answer.** Neither Euclidean nor Fisher geometry prices bytes by
itself. Euclidean geometry describes the optimizer's update coordinates;
Fisher geometry describes local scorer decision effect; actual receiver-closed
`Delta bytes / Delta score` is the waterfill price. The measured cosine sign
flip is therefore expected to be informative rather than contradictory. Keep
Euclidean cosine, Fisher cosine, and relative norm together. The existing
`src/tac/witness_dsl/bregman_dual_metric_guard.py` is also binding: raw dual
Euclidean is an `H^2` geometry, not a no-solve substitute for Fisher-natural
`H^{-1}` cotangent geometry.

**Named consumers.** Task #536,
`tools/dual_metric_readback.py`, and the canonical metric
`argmax_native_vjp_fidelity_v1`.

**$0 falsifiable gate.** On the exact candidate moves used by the pending Seg
secant, freeze train/holdout pair IDs and compare four predictors of held-out
receiver-closed `Delta S / Delta B`: Euclidean-only, Fisher-only, both plus
relative norm, and a bytes-only baseline. Report sign accuracy, Spearman rank,
top-k overlap, and calibration error. Adopt the dual schema only if it improves
held-out ranking/calibration over both single-metric baselines without changing
the hard-oracle admission set. If only one metric exists, the other read-back is
explicitly owed.

**Stop rule.** Do not turn a better cosine into a byte-allocation law. If the
dual features do not predict the finite receiver-closed secant, retain them as
diagnostic telemetry only.

### 3. Separate Dykstra convergence from nonlinear repair traffic — `ADOPT_DIAGNOSTIC_FIRST`

**Literature basis.** Benamou et al.'s
[iterative Bregman projections paper](https://arxiv.org/abs/1412.5154) uses
closed-form projections onto simple constraint sets and Bregman-Dykstra
iterations when inequality constraints are present. This supports the solver
family, not a claim that one order will reduce our hard-oracle retries.

**Local correction.** `project_pixelwise_seg_relaxation` already solves each
independent three-coordinate box/half-space KKT problem by breakpoints.
`solve_constructive_projection` invokes cyclic Dykstra only when intersecting
that combined set with the Pose ellipsoid. The linear Pose proposer was active
in 4/96 rows while hard Pose was inactive 96/96. Reordering the two projections
therefore remains a low-EV explanation for repairs `9 -> 41`, although the four
linear-active rows should be checked rather than assumed irrelevant. Those
counts are hard-oracle proposal retries after nonlinear native-float32
evaluation, not failure of the inner convex projection to converge.

**Named consumers.** The band-law measurement path
`tools/measure_joint_seg_pose_rate.py`, its solver
`src/tac/optimization/joint_seg_pose_rate.py`, and the production proposal
surface `src/tac/optimization/v10_constructive_solver.py`.

**$0 falsifiable gate.** Before changing an algorithm, append per proposal:
continuous projection iteration count, active-set size, minimum linear slack,
hard repair level, hard mismatch count, candidate hash, and whether Pose
projection changed any byte. Reuse the fixed n24 inputs and four existing
operating points. Then compare:

- current analytic/cyclic baseline;
- same converged quadratic problem with reversed set order; and
- narrow-to-wide continuation initialized from the prior hard-admitted point,
  clearly labeled as a changed anchor/objective if it is one.

The gate passes only if the final hard-admitted bytes and constraints are
identical while hard retries or wall time fall. A different admitted candidate
is not an ordering speedup and must enter the secant as a new treatment.

**Further INFERRED route.** A nonquadratic Bregman generator aligned to a
validated separable rate surrogate could improve proposals, but actual
Brotli/zstd bytes are discrete and are not automatically a Bregman divergence.
No such generator should be adopted until its held-out rank correlation with
actual bytes beats the current weighted-quadratic proxy.

### 4. Chernoff point as an ambiguity prioritizer — `PROBE_NOT_ALLOCATE`

**Literature basis.** Nielsen's
[Chernoff-information paper](https://arxiv.org/abs/2207.03745) gives Chernoff
information as a minmax KL symmetrization and locates the unique Chernoff point
using an exponential geodesic and the corresponding mixture/Bregman bisector.
For categorical logits with `F=logsumexp`, the oriented Bregman divergence is
the corresponding oriented KL divergence. The skew Jensen quantity

```
J_F^alpha(theta_0,theta_1)
  = (1-alpha) F(theta_0) + alpha F(theta_1)
    - F((1-alpha) theta_0 + alpha theta_1)
```

and its maximizing `alpha` define a two-endpoint Chernoff radius. This is a
closed one-dimensional search, not a byte allocator.

**Named consumers.** #539 active facet/tie diagnostics and #541's active
constraint triage, never direct byte admission.

**$0 falsifiable gate.** For each winner/rival cell in fixed held-out pairs,
compute a two-class Chernoff/Jensen ambiguity statistic without loading anything
at decode time. Compare its precision/recall and top-k overlap for actual
native-float32 flips or repair-causing cells against the existing cached margin,
closed-form feature flip distance, and full-K categorical Fisher statistic.
Adopt only if it gives held-out lift and the chosen orientation is recorded.

**Stop rule.** Chernoff information is a robust pairwise ambiguity radius, not
`d_seg`, archive bytes, or a substitute for full-K triple-saddle behavior. No
conversion to the 150.181956-B break-even is valid without an empirical map.

### 5. Smallest-enclosing Bregman ball as a support-set proposal — `LOWER_PRIORITY_PROBE`

**Literature basis.** Nock and Nielsen's
[smallest-enclosing-Bregman-ball material](https://franknielsen.github.io/SlidesVideo/BCBregmanBalls-ECML-2005.pdf)
uses a farthest-point update in dual coordinates and exposes support points of a
minimax center. Its displayed update has the form

```
c_(t+1) = nabla F^-1((t/(t+1)) nabla F(c_t)
                     + (1/(t+1)) nabla F(s_farthest)).
```

It is attractive as a coreset heuristic. The material does not
establish that our overlapping receptive-field half-spaces and discrete repair
oracle are a point-cloud ball, and the arbitrary-Bregman approximation guarantee
is not a free transfer.

**Named consumers.** #536 group allocation and #541 active-set proposal
compression.

**$0 falsifiable gate.** Select a support subset by farthest-first Bregman
radius on a frozen training split, solve using that subset, then recheck every
omitted constraint and the full hard oracle on held-out pairs. Compare against
margin-top-k and uniform subsets at the same support count. Adoption requires
the same admitted result, zero omitted violations, and reduced proposal work;
the full recheck remains mandatory forever.

**Stop rule.** Any missed hard constraint rejects the coreset at that scope; it
does not reject Bregman balls as a family.

### 6. Centroids/Jensen bounds and pyBregMan — `REFERENCE_ONLY`

Nielsen and Nock's
[sided/symmetrized Bregman centroid paper](https://arxiv.org/abs/0711.3242)
and pyBregMan can supply independent fixtures for sided centroids, geodesics,
bisectors, Chernoff points, and enclosing balls. A centroid summarizes average
divergence; a Chernoff or enclosing-ball radius targets worst-case ambiguity.
Neither is an archive-cost law.

**Named consumer.** Unit-test/diagnostic fixtures adjacent to #536/#539 only.

**$0 falsifiable gate.** On seeded synthetic convex generators, require the
local NumPy reference to agree with pyBregMan for primal/dual transforms,
oriented divergences, bisectors, and centers within a declared tolerance. Do
not add pyBregMan to the production or inflate dependency set.

## What the corpus settles, and what it does not

| question | verdict |
|---|---|
| Is the rank-4 max-of-affine head a power diagram? | **DERIVED YES over real channel-quotient arithmetic.** |
| Are generators plus weights a complete description of those channel cells? | **DERIVED YES**, modulo gauge and declared finite arithmetic. Explicit tie coefficients are redundant. |
| Does that describe the spatial n600 witness? | **NO at implementation/custody scope.** The nonlinear feature-field/RGB pullback is absent. This is not a power-diagram-family negative. |
| Does Nielsen's corpus give a smaller target description? | **DERIVED construction YES (158/138/134 raw-byte variants), MEASURED packet NO.** |
| Does Dykstra theory explain repairs 9 -> 41? | **NO.** The measured counts are nonlinear hard-oracle retries. The linear Pose proposer was active in only 4/96 rows and hard Pose was inactive 96/96; per-row correlation is still owed. |
| Which metric prices bytes? | **Neither alone.** Geometry predicts decision effect; only receiver-closed finite score/byte secants price the waterfill. |
| Can Chernoff/Jensen/Bregman-ball radii be converted to 150.181956 B per `1e-6 d_seg`? | **NO without calibration.** The break-even is score-law arithmetic; the achievable secant remains pending. |
| Does generalized Legendre theory prove V9 CGauge runtime covariance? | **NO at implementation-custody scope.** A typed transform and pre/post action-equality receipt remain owed. |

## Exact primary sources

1. Frank Nielsen, Jean-Daniel Boissonnat, and Richard Nock,
   [“Bregman Voronoi Diagrams: Properties, Algorithms and Applications”](https://arxiv.org/abs/0709.2196).
2. Frank Nielsen,
   [“An elementary introduction to information geometry”](https://arxiv.org/abs/1808.08271).
3. Jean-David Benamou, Guillaume Carlier, Marco Cuturi, Luca Nenna, and Gabriel
   Peyre,
   [“Iterative Bregman Projections for Regularized Transportation Problems”](https://arxiv.org/abs/1412.5154).
4. Frank Nielsen,
   [“Revisiting Chernoff Information with Likelihood Ratio Exponential Families”](https://arxiv.org/abs/2207.03745).
5. Frank Nielsen and Richard Nock,
   [“On the Centroids of Symmetrized Bregman Divergences”](https://arxiv.org/abs/0711.3242).
6. Richard Nock and Frank Nielsen,
   [“Fitting the Smallest Enclosing Bregman Ball”](https://franknielsen.github.io/SlidesVideo/BCBregmanBalls-ECML-2005.pdf).
7. Frank Nielsen et al.,
   [“pyBregMan: A Python library for Bregman Manifolds”](https://arxiv.org/abs/2408.04175).
8. Frank Nielsen,
   [“Curved representational Bregman divergences and their applications”](https://arxiv.org/abs/2504.05654).
9. Frank Nielsen,
   [“A note on the Artstein-Avidan-Milman's generalized Legendre transforms”](https://arxiv.org/abs/2507.20577).

## Triality, six-hook disposition, and resumability

- **Equations leg:** affine-to-power conversion; 20-scalar margin gauge;
  19-scalar partition gauge; primal/dual local metric identity; exact
  150.181956-B score-law break-even.
- **DAG leg:** `frozen rank-4 head -> gauge-fixed generators -> derived ties ->
  declared receiver arithmetic -> spatial pullback -> R -> hard Seg/Pose ->
  exact archive secant`. This memo authorizes only the first routing edge.
- **DSL leg:** N/A at `research_only=true`. Any packet/solver treatment must be
  additive, legacy-compatible, typed through the canonical DSL/registry, and
  resumable before a run is considered.
- **Sensitivity-map hook:** Chernoff/ball/dual values may be proposal metadata
  only until held-out hard-oracle calibration.
- **Pareto hook:** the actual receiver-closed `(Delta bytes, Delta d_seg,
  Delta d_pose)` row is the only admissible constraint surface.
- **Bit-allocator hook:** #536 may consume a calibrated finite secant; it must
  reject null Seg curves and uncalibrated divergence radii.
- **Autopilot hook:** none. `ready_for_dispatch=false`; no launch is authorized.
- **Continual-learning hook:** this memo plus the research-only lane row is the
  durable posterior. There is no new empirical anchor to register.
- **Probe-disambiguators:** each ranked recommendation names its fixed-input $0
  gate; where two interpretations remain, both are compared against the same
  hard receiver outcome.

No job or scratch artifact was created, so no storage waterfall, cleanup, or
resume checkpoint was needed beyond the delegated lane checkpoints.

The lane is registered `research_only=true` at L1 with this memo as its sole
`impl_complete` evidence. Canonical whole-registry validation remains blocked by
110 historical missing-evidence paths outside this lane; no failing path belongs
to `nielsen_infogeo_crosswalk_20260719`.

## Stores consulted

1. `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and
   `docs/operating_manual_craft_handoff.md`.
2. `.omx/research/v10_flattened_lagrangian_kkt_derivation_20260719.md`.
3. `.omx/research/vjp_custody_positive_bands_20260719_codex.md` and
   `.omx/research/joint_seg_pose_inverse_solve_20260719_codex.md`.
4. `.omx/research/constructive_solver_541_20260719_codex.md` and its
   implementation spec.
5. `.omx/research/segnet_recursive_fractal_factorization_20260715.md`.
6. `.omx/research/power_diagram_witness_20260718.md`,
   `.omx/research/v10_power_diagram_byteclose_findings_20260718.md`, and
   `.omx/research/v10_power_diagram_byteclose_build_spec_20260718.md`.
7. `.omx/research/codex_premise_falsification_factor10_kkt_waterfill_20260718T121220Z_codex.md`
   and `.omx/research/factor10_kkt_waterfill_DAG_FEED_20260718.md`.
8. Project memory
   `dual_metric_readback_euclid_cosine_vs_fisher_both_informative_20260717.md`.
9. `src/tac/boundary_math/power_diagram_witness.py`,
   `src/tac/optimization/v10_constructive_solver.py`, and
   `src/tac/witness_dsl/bregman_dual_metric_guard.py` by source inspection.
10. Canonical frontier/lane/subagent/ledger surfaces and both delegated inboxes;
    no newer per-arm directive or completed sibling Seg-secant receipt was
    present before drafting.
11. The primary Nielsen/arXiv corpus listed above. No source outside
    `franknielsen.github.io` or `arxiv.org` was used for literature claims.

The sacred directory
`experiments/results/levelset_n600_witness_20260717T113932Z/` was not modified or
invoked.

## Self-review and MAIN handoff

Four bounded adversarial rounds were used, within the cap of five:

1. **Claim/unit attack:** separated the exact 150.181956-B score-law break-even
   from the pending achievable Seg secant; corrected 9 -> 41 from “Dykstra
   iterations” to hard-oracle repair levels; distinguished linear Pose active
   4/96 from hard Pose inactive 96/96; and marked 158/138/134 as derived
   constructions rather than measured/lower-bound bytes.
2. **Receiver/equivalence attack:** carried the frame-195 float32 confound into
   the #539 gate; retained the spatial-pullback blocker; kept dual Euclidean
   distinct from Fisher-natural `H^{-1}` geometry; and rejected direct
   divergence-radius-to-byte conversion.
3. **Registry/seam attack:** rechecked every cited local path, allowed-domain
   URL, packet sum, score-law unit, sacred-tree status, and
   implementation/measurement boundary. Whole-registry validation exposed 110
   pre-existing missing-evidence paths outside this lane; that blocker is now
   carried explicitly rather than hidden.
4. **Clean pass:** repeated the local diff, path, lane-row, unit, and custody
   checks after the registry disclosure. It found no further claim or formatting
   defect.

**MAIN landing review is required.** Before merge or any adoption, MAIN should
independently review: (a) the 20/19-degree gauge count and 338 -> 158 -> 138 ->
134 byte arithmetic; (b) the claim that tie loci may be recomputed with exact
declared float32 semantics; (c) the Dykstra-versus-hard-repair separation; (d)
the metric/byte-pricing boundary; and (e) the absence of a completed sibling
Seg-secant receipt. This branch grants no promotion, launch, score, or pointer
authority.
