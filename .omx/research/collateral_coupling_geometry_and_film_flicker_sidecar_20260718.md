# Channel-space collateral coupling and a FiLM flicker sidecar

Date: 2026-07-18
Lane: lane_collateral_coupling_film_flicker_sidecar_20260718 (L0, research_only=true)
Authority: deep-math analysis and design only; no training, paid dispatch, archive promotion, or pointer move
Evidence axes: exact frozen-weight algebra plus labeled [macOS-CPU advisory] measurements; no contest score claim

## Verdict first

**[DERIVED — exact at the frozen SegNet head] The coupling is solvable in the correct local coordinates.**
The five affine class logits form an upper envelope in a four-dimensional decision quotient.  The ten
pair normals are only rank four, and a feasible target winner cell is reached by an active-set metric
projection onto an intersection of four half-spaces (closed form conditional on the active set).  This
replaces independent pixel crossings with one joint polyhedral solve.

**[UNMEASURED — renderer/full-chain] Whether that solve is coherently exploitable is unknown.**
The exact head supplies the target and the four control coordinates, but the camera-to-SegNet map is
spatially extended and nonlinear before the head.  The published real-weight evidence does not contain an
n600 stacked spatial FiLM-actuator/Jacobian Gram matrix or an n600 coherent correction A/B.  The local
head Gram cannot substitute for cross-pixel collateral.  Therefore a nonzero collateral-free Road-to-Lane
reduction and its reachable amount are both **unknown**.  One n24 diagnostic reported an approximately
0.0011 d_seg counterfactual magnitude under collateral-free harvesting; it is neither an n600 bound nor an
achieved result and is not promoted here as an n600 finding.

**[INFERRED — design] The right v10 section 14.2 form is a train-side, channel-aligned, spatially gated
residual FiLM sidecar plus an MS pullback seed, not a frozen-output edit.**  Pair4 is the mandatory
lowest-rank ablation; deployed rank r must come from the stacked n600 actuator spectrum and exact byte
marginal.  The same counted seed supplies pair-labeled spatial gates/critical precision and quantized
coefficients, then co-adapts with the renderer through the real R operator.  Pair4 costs about 2.4 KB raw
before header/basis; no compressed byte or d_seg benefit is claimed until typed receiver parse-back and
the currently NOT_LAUNCHABLE n600 design ticket land.

**[MEASURED — process] Pointer 0.1910828242 [contest-CPU Linux x86_64] is unchanged.**  The sacred
experiments/results/levelset_n600_witness_20260717T113932Z run was not read from or modified by this work.

## 1. Evidence custody and exclusions

### 1.1 Primary artifacts reused, not re-derived

- **[MEASURED]** The frozen scorer chain and shared resize are from
  .omx/research/frozen_scorer_exact_factorization_20260715.md.
- **[MEASURED]** The actual head weights, singular values, pair-normal norms, ERF, and stride-2 skip
  intervention are from .omx/research/segnet_recursive_fractal_factorization_20260715.md and the registered
  segnet_head_rank4_linear_flipdist_v1 equation.
- **[MEASURED]** Full-video edge counts, margins, saddle census, and inverse-preimage strata are from
  .omx/research/necessity_solver_inverse_factorization_20260715.md (n600 where stated there).
- **[MEASURED]** The composed Road-Lane gain and skip-channel factorization are from
  .omx/research/lane_channel_deep_refactorization_20260716.md.
- **[MEASURED]** The ker(A) dimensions and energy fractions are from
  .omx/research/null_subspace_rate_measure_20260717.md (8-pair/16-frame decoded-energy leg and 2-pair
  output-layer Jacobian-effect leg; its separate scorer-delta leg used n32).
- **[MEASURED]** The #425 byte-only carrier row is from
  .omx/research/p0_425_phase_carrier_byte_close_row_20260716.md.
- **[DIAGNOSTIC ONLY]** .omx/research/phase_stack_efficacy_probe_v10_gate_20260718.md is n24 and is
  used only to name the mis-test and its sample-local counterfactual magnitude.  Its efficacy sign and
  magnitude are not n600 findings or bounds.

### 1.2 Fresh artifact custody in this pass

- **[MEASURED]** The operator-provided banked checkpoint exists at
  experiments/results/banks/v9c2_defensive_bank_20260718/levelset_witness_ema_BEST.npz in the canonical
  repository and hashes to
  b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef.
- **[MEASURED]** Read-only NPZ inspection gives epoch 725 architecture tensors including code shape
  (1200,32), film.weight shape (768,32), film.bias shape (768,), four hidden 96x96 matrices, and a
  five-row out_sdf head.  This is a real receiver attachment point; it is not a proposed standalone
  side module.
- **[UNMEASURED]** The gitignored raw stage_a/stage_b/lane-chain JSON artifacts are absent from this
  isolated worktree.  Consequently, the committed angle summary is custody, but a full numeric 10x10
  pair-normal Gram table is not independently reproduced here.
- **[EXCLUDED]** The historical minus-48-percent directional-basis proxy is not used.  It is
  unreproduced through the real n600 receiver; the settled warm-start comparison has OFF slightly better
  than the directional arms.

## 2. The first-class object: a rank-4 tropical hyperplane arrangement

Let f in R^144 be the frozen SegNet penultimate 16-channel x 3x3 patch and let

    z_c(f) = w_c^T f + b_c,              c in {Road, Lane, Undrivable, Movable, MyCar}.

Define the oriented pair normal and margin

    n_cc' = w_c - w_c',
    m_cc'(f) = n_cc'^T f + b_c - b_c'.

### 2.1 What is exact

- **[MEASURED]** The centered head singular values are (3.128, 2.154, 2.025, 1.796, 0), with rank-4
  reconstruction error 5.96e-8.  Common logit shift is a gauge, so the decision quotient has exactly
  four measured nonzero directions.
- **[DERIVED]** The max-plus polynomial max_c z_c(f) has at most five convex winner cells

      C_y(mu) = {f : (w_y-w_k)^T f + b_y-b_k >= mu for every k != y}.

  (with the observed labels establishing nonempty cells on this video's support).  At mu=0 its walls are
  the active upper-envelope facets.  The same object is an affine/power
  (Laguerre) diagram, a tropical hypersurface at ties, and a polyhedral upper envelope.  These are three
  descriptions of one frozen-head object, not three new levers.
- **[DERIVED]** There are ten formal pair hyperplanes, but only hyperplanes that support an
  upper-envelope facet at a given f can be crossed as a direct class adjacency.  Observed image-space
  adjacency validates active facets for this video; absence in this video does not prove a frozen-head
  facet is globally empty.
- **[DERIVED]** Zaslavsky's generic affine bound for ten hyperplanes in four dimensions is
  sum_{i=0}^4 choose(10,i)=386 chambers.  It is only a loose comparison bound here: difference normals
  obey cycle identities such as n_ab+n_bc=n_ac, pairwise signs must describe a transitive ordering of
  five logits (at most 5!=120 strict order chambers), and the scored upper envelope retains at most five
  convex winner cells.  No 386-cell or 120-cell rate claim is licensed.
- **[DERIVED]** The target cell label is a cheap combinatorial training target only when generated from
  legal compressed geometry.  The archive must never contain a per-pixel GT argmax table or scorer
  weights.  A world-dash/curve seed may generate memberships at decode; a copied frozen-scorer label map
  would violate the witness contract.

### 2.2 Exact pair geometry by class

Class order is Road=0, Lane=1, Undrivable=2, Movable=3, MyCar=4.

| unordered pair | head normal norm | real n600 edge geometry | control reading |
|---|---:|---|---|
| Road-Lane | **3.953 MEASURED** | 1356.8 cracks/frame; 21.42 components/frame; median margin 0.39; median feature flip distance 0.0987 **MEASURED** | largest spatial debt; 16-channel stride-2 skip owns 77% of its skip-ablation flips **MEASURED** |
| Road-Undrivable | 2.602 **MEASURED** | 483.6 cracks/frame; 4.15 components/frame; margin 0.35; feature distance 0.135 **MEASURED** | lower head gain than Lane pairs; major active facet **DERIVED from measured adjacency** |
| Road-Movable | 2.942 **MEASURED** | 150.5 cracks/frame; 3.64 components/frame; margin 0.23; feature distance 0.078 **MEASURED** | fragile major facet; object motion prevents a ground-only phase model **INFERRED** |
| Road-MyCar | 2.705 **MEASURED** | 529.5 cracks/frame; 1.94 components/frame; margin 0.42; feature distance 0.155 **MEASURED** | nearly static hood boundary is a one-seed control, not a per-pair flicker sidecar **INFERRED** |
| Undrivable-Movable | 2.946 **MEASURED** | 165.9 cracks/frame; 3.50 components/frame; margin 0.19; feature distance 0.065 **MEASURED** | lowest listed feature distance; non-ground object motion still matters **INFERRED** |
| Lane-Undrivable | 3.748 **MEASURED** | included in the published rare-pair group, <=8.5 cracks/frame **MEASURED group bound** | Lane-amplified but sparse on this video **DERIVED** |
| Lane-Movable | **4.007 MEASURED** | included in the rare-pair group, <=8.5 cracks/frame **MEASURED group bound** | largest head normal, but not a dominant n600 spatial debt **DERIVED** |
| Lane-MyCar | 3.862 **MEASURED** | included in the rare-pair group, <=8.5 cracks/frame **MEASURED group bound** | Lane-amplified but sparse on this video **DERIVED** |
| Movable-MyCar | 2.910 **MEASURED** | included in the rare-pair group, <=8.5 cracks/frame **MEASURED group bound** | no ground-homography assumption allowed **DERIVED** |
| Undrivable-MyCar | 2.869 **MEASURED** | no individual n600 count in the consulted published table **UNMEASURED** | do not infer zero adjacency |

- **[MEASURED]** All four Lane-involving normal norms (3.748-4.007) exceed every non-Lane norm
  (2.602-2.946).
- **[MEASURED, summary custody]** The ten pair-normal angles span 25.8 degrees to 90 degrees, with
  median 62 degrees.
- **[DERIVED]** This is not an orthogonal ten-control system.  Ten normals in a four-dimensional
  quotient have at least six dependencies, and the 25.8-degree minimum indicates a strongly shared
  direction for at least one pair of pair normals.  The median also rules out treating the typical
  cross-effect as zero by fiat.
- **[UNMEASURED]** Which specific pair-pair entries are positive, negative, or close to zero in the
  Fisher metric remains owed because the full oriented Gram matrix is not in committed custody.

### 2.3 Spatial pullback: the conditional Morse-Smale dual

Let the decoded/scored feature map be

    F_theta(x) = SegNetPenultimate(R(render_theta))(x).

The exact spatial pullback of a frozen-head winner cell is

    Omega_y^score = F_theta^-1(C_y),
    Sigma_yk^score = {x : m_yk(F_theta(x))=0 and z_y=z_k>=z_j for every j}.

- **[DERIVED — exact]** The co-maximum clause matters: the pullback of a formal equality hyperplane is
  a scored class boundary only where those two logits dominate all others.  This is the spatial
  1-skeleton of the scorer decision complex; two independent co-maximum equalities generically give an
  isolated triple-class 0-stratum in a two-dimensional image.
- **[DERIVED — conditional]** If grad_x(m_yk composed with F_theta) is nonzero on an edge, the implicit
  function theorem makes that edge locally a smooth curve.  A channel perturbation displaces it by

      delta_s_yk(x) = -[n_yk^T B_x delta_u] / ||grad_x(m_yk composed with F_theta)||,

  to first order, where B_x=dF_theta(x)/du.  Thus the FiLM channel move and the spatial separatrix move
  are one actuator viewed before and after pullback.
- **[CORRECTION]** The witness-native tie set

      Sigma_yk^phi = {x : phi_y(x)=phi_k(x)>=phi_j(x)}

  is an actuator scaffold, not automatically Sigma_yk^score.  They coincide only where render, R, and
  the frozen scorer preserve the intended cell.  Their discrete disagreement contributes exactly the
  residual d_seg this design must reduce; assuming equality would erase the problem being solved.
- **[CORRECTION]** This pullback is a stratified/Morse-Smale-like cell complex, not automatically a
  classical Morse-Smale complex of one scalar potential.  Triple-logit ties are codimension-two
  junctions; they are classical Morse critical points only with a specified potential, zero-gradient
  condition, nondegenerate Hessian, and transversality certificate.  Until those are measured, this memo
  uses “MS pullback complex” as a combinatorial carrier description, not a theorem of gradient flow.

The existing apparatus occupies the correct strata without collapsing those distinctions:

- **[MEASURED n600]** The necessity solver found 97.8% scorer-cell interiors, pair-labeled edge curves,
  and 6,703 junction vertices (11.2/frame; only one four-way event).  Lane participates in three of the
  four largest triple-class groups.
- **[BUILT, no score claim]** TieLocusDisplacement (#360) actuates subpixel edge-normal placement;
  the per-class critical-nucleus guard (#315) delays CE-to-tau until every present class has positive,
  formed mass.  The former is an edge force and the latter a birth admissibility sensor, not stored
  topology bytes.
- **[MEASURED negative, other pose formulation]** The #365 MS-stratified low-DOF parallax carrier did
  not close pose: advisory medians were d_pose 1.685 (0 DOF), 1.486 (6 DOF), and 1.223 (12 DOF/oracle
  mask), with only about 0.5% off-plane parallax mass.  That formulation negative forbids claiming the
  cell complex supplies a solved pose carrier; the joint pose head remains separately routed.

### 2.4 One representation: MS seed plus channel actuator

The legal counted object is not a per-pixel cell table.  It is an irreducible seed

    s_MS = (cell/RAG metadata,
            pair-labeled edge-generator seeds and birth/death events,
            junction precision annotations,
            quantized low-rank channel coefficients a),

consumed by one deterministic receiver: generate the witness-native cells/curves, transport them by xi,
apply the boundary-gated channel residual, render through R, and verify the scorer pullback cells.  Generic
graph/curve rasterization and projection code are free under rule 118; every video-derived coordinate is
counted.  Cell labels may be generated from legal class/curve seeds, but no GT argmax table, scorer weight,
or hidden per-pixel membership map may ship.

- **[DERIVED]** Cells, edges, and junctions are not separate payloads when one implies another.  The
  necessity solver estimates cell palettes near 15 B/video given edges and junction bytes near zero given
  intersecting curves; junctions instead demand tighter precision on the incident edge seeds.
- **[DESIGN]** The same edge seed generates g_p(x), while a_p controls displacement through the measured
  channel basis.  Export must eliminate information duplicated by the base code, xi, or edge generator.
  This is the precise sense in which the MS seed and FiLM control are one representation.
- **[NOT PROVEN]** A true critical-point/adjacency trajectory codec has not been built or byte-closed.
  The settled proxies say a standalone full-complex seed is presently larger, not smaller, than the
  vehicle.  Therefore the live design is a residual/irreducible-seed composition on v9c2, not a standalone
  MS replacement.

Measured/derived compactness custody:

| representation | counted/custody bytes | comparison with v9c2 BEST |
|---|---:|---|
| v9c2 BEST packed carrier | 0.bin 64,376 B; archive.zip 63,659 B **MEASURED** | reference; realized d_seg for this bank still owed |
| v9c2 BEST training checkpoint | 460,448 B NPZ **MEASURED** | not contest-counted and not a valid rate comparator |
| #180 full polygon partition, eps=0.5 | 444,000 B extrapolated; partition d_seg 5.57e-4 **MEASURED sample-40** | about 7.0x the packed archive; standalone rate-dominated |
| FEED-fh full / rare-only arcs | 513 KB / 221 KB **MEASURED n48 proxy, extrapolated to 600** | about 8.1x / 3.5x the packed archive; standalone deferred |
| necessity K-ladder edge seeds, eps=1 px | 143,552 B **MEASURED bytes, spatial tolerance only** | about 2.3x the packed archive; no through-R d_seg calibration |

The byte answer is therefore **[MEASURED/UNMEASURED split]**: old full/rare MS proxies are not compact
against the current carrier; the proposed true cells+adjacency+critical trajectory seed is unmeasured.
Its reactivation gate is an exact parse-back seed whose full n600 packed-byte verdict beats v9c2 after
cell, pair-boundary, junction-incidence, Seg, Pose, and byte comparisons.  The #369 raster edit sidecar
also stays formulation-negative at 1.254 B/corrected flip versus the through-R admission bar 0.65; it is
not silently renamed as this seed.

## 3. The coupling law: head Gram, actuator Gram, and spatial spill

Choose a positive feature metric M.  Euclidean M=I reproduces the registered single-pair flip distance;
the local Fisher/margin metric is the decision-aware choice when its custody is present.  Stack the
oriented pair normals as rows of N.

### 3.1 Head-only cross-coupling

For a desired change q in one active margin t, the metric-minimum single-facet move is

    delta_f_t = M^-1 n_t q / (n_t^T M^-1 n_t).

Its effect on any other oriented margin s is

    delta_m_s = q * (n_s^T M^-1 n_t) / (n_t^T M^-1 n_t).

Define the head Gram G_head = N M^-1 N^T.

- **[DERIVED — exact]** Off-diagonal G_head entries are the irreducible channel-space pair coupling.
  Zero means first-order margin decoupling in metric M; sign predicts help versus harm for the chosen
  orientation; magnitude predicts how much another margin moves per target-margin unit.
- **[DERIVED — exact]** Moving along n_t crosses only the intended upper-envelope facet **if and only
  if** every other target-cell inequality remains satisfied along the line segment.  A pair normal is
  not globally collateral-free merely because it is the normal of one hyperplane.
- **[DERIVED — exact]** When more than one inequality is active, the correct object is the joint cell
  projection

      minimize_delta  0.5 delta^T M delta
      subject to      N_y (f+delta) + beta_y >= mu,

  where N_y contains the four winner-versus-rival normals.  For a known equality active set A,

      delta* = M^-1 N_A^T (N_A M^-1 N_A^T)^dagger r_A.

  This is solve-don't-train at the frozen head.  Training is only the realization step from renderer
  controls to this solved target.

### 3.2 Renderer reachability, not head algebra, decides efficacy

Let u be a low-dimensional FiLM control and let B_p = d f_p / d u be the real renderer -> R -> frozen
SegNet-penultimate response at pixel p.  Then

    delta_m_p = N B_p u,
    G_act,p = (N B_p) H_u^-1 (N B_p)^T.

For cross-pixel collateral, stack locations as well as class-pair margins:

    J_spatial[(j,s),(i,k)] = d m_s(j) / d x_k(i),
    J_u = stack_p(N_p B_p),
    G_act,stack = J_u H_u^-1 J_u^T.

- **[DERIVED]** A requested active-margin residual r is reachable at p only if it lies in
  range(N_A B_p); the least-control solution is the corresponding weighted pseudoinverse.  Residual
  outside that range is the honest actuator blocker.
- **[DERIVED]** G_act, not G_head alone, is the FiLM exploitability measurement.  It includes the
  renderer, R, frozen nonlinear trunk, spatial gate, uint8 straight-through training convention, and
  actual channel consumption.
- **[DERIVED]** The off-diagonal blocks of G_act,stack—not G_head—measure whether changing pair s at
  location j moves another pair/location.  G_head supplies only the same-patch decision geometry.
- **[UNMEASURED]** No n600 stacked J_spatial or G_act exists today.  Therefore neither low-rank coherent
  exploitability nor a nonzero collateral-free reduction may be called measured.

### 3.3 Why pixel corrections spill even though each local head is rank four

- **[MEASURED, n96 advisory]** Margin-gradient ERF has r50 about 50-160 scorer pixels (median about 85),
  r90 about 206-424 (roughly 300), and 46-74 percent of gradient mass beyond 65 pixels.
- **[DERIVED]** The global image Jacobian is a block-spatial operator: each local output has at most four
  decision directions, but those blocks overlap across a large ERF.  Rank four per patch does **not**
  imply rank four for the whole frame.  The global spatial coupling can be high rank.
- **[MEASURED, n600]** Road-Lane has 21.42 components/frame and about half of all cracks.  Four global
  channel coefficients cannot independently specify every dash phase unless a legal spatial generator
  and coherent gate supply the component locations.
- **[DIAGNOSTIC ONLY, n24]** Independent min-norm camera edits showed regional overshoot dominated by
  the opposite Lane-to-Road orientation.  The geometry explains the sign: n_LR=-n_RL and overlapping
  ERFs superpose normal displacements.  The magnitude/sign is not elevated to an n600 efficacy verdict.

### 3.4 All requested factors and dimensions

| factor | evidence | consequence for the sidecar |
|---|---|---|
| class-pair channel normal | norms and rank are **MEASURED**; Gram entries mostly **UNMEASURED** | solve all winner-cell inequalities, never one independent pixel crossing |
| spatial ERF | broad r50/r90 **MEASURED n96** | accept only with full-neighborhood and full-frame collateral accounting |
| boundary curvature/components | cracks, turns, components **MEASURED n600** | use connected-component/world-dash gates; a global scalar alone is insufficient |
| stride-2 skip | 16 channels at 192x256; Road-Lane 77% of induced flips **MEASURED n16** | add an auxiliary skip-response target during joint training; scorer is absent at decode |
| luma/chroma | Road-Lane full-chain gradient about 90% luma; skip channel 10 is 99.6% chroma-tuned **MEASURED small sample** | retain both geometric/luma phase and fine-chroma control; do not declare a single private Lane channel |
| temporal xi coherence | Lane dash churn and xi transport need are **MEASURED n600**; exact sidecar win **UNMEASURED** | gate in the world/dash chart and generate per-pair control from xi where possible |
| uint8 realization | saddle/edge sub-LSB fractions are **MEASURED on labeled subsets** | quantization/parse-back belongs inside the training loop and verdict path |
| along-tangent versus normal | head normal is exact channel decision direction **DERIVED**; image tangent modes are geometry-dependent | channel projection chooses the class move; the dash/curve generator chooses where along the boundary it acts |
| Fisher/margin | head margin is exact distance after metric normalization **DERIVED**; Pearson 0.978 is inherited measured context | weight the QP and acceptance by decision margin, not RGB norm |

## 4. ker(A), range(A^T), and the honest collateral-free manifold

The shared camera-to-scorer resize is A=A_seg=A_pose.

- **[MEASURED]** Its camera-space kernel dimension is 80.674 percent; 22.6969 percent of camera pixels
  lie on exact zero-weight rows/columns.  On 8 pairs/16 decoded frames, about 52 percent of frame energy
  lay in ker(A); on a separate 2-pair output-layer Jacobian leg, about 50-53 percent of linearized effect
  lay there.  These are stable-looking advisory subsamples, not n32 measurements.
- **[DERIVED]** ker(A) cannot fix d_seg because it is invisible.  It is gauge freedom **after** a visible
  solution, not target-moving control.  Calling the large kernel a large exploitable correction space
  would be a category error.
- **[DERIVED]** The minimum-camera-energy lift of a scorer-space correction lies in range(A^T).  The
  complete preimage is x_visible* + ker(A).  Set the kernel component to zero unless another legal
  objective needs it; this avoids spending renderer capacity on scorer-invisible energy.
- **[UNMEASURED]** The nonlinear renderer has no exact weight-space ker(A) projection.  The prior
  50-53-percent weight-effect result is first-order and cannot be turned into a byte-saving claim for a
  grammar with no camera-resolution payload.
- **[DERIVED]** The useful collateral-free set is therefore not ker(A).  It is the intersection of
  (i) the GT winner-cell cone, (ii) the range of the FiLM actuator through R and the frozen trunk,
  (iii) unchanged/slack-preserved inequalities for already-correct pixels, (iv) the uint8 lattice, and
  (v) the pose tube.  Whether this intersection contains a useful Road-Lane trajectory is the n600 test.

### Exploitability answer in numbers

- **[MEASURED n600 geometry]** Road-Lane is the largest spatial stratum and most fragmented active
  boundary.
- **[DERIVED from exact head]** Four channel coordinates are sufficient to specify any feasible local
  winner-cell correction at the frozen head.
- **[DIAGNOSTIC MAGNITUDE, n24 only]** Approximately 0.0011 d_seg was attributed to Road-to-Lane flips
  in that sample if all could be harvested without collateral.
- **[UNMEASURED]** The full-n600 Road-to-Lane error count needed even for a numerical error-mass ceiling
  is not in the consulted custody.  The portion reachable by a coherent FiLM actuator is also unknown;
  no positive lower bound or n600 upper bound is asserted.  Do not transfer 0.0011 or a percentage from
  n24.

## 5. FiLM flicker sidecar: the geometry-derived design

### 5.1 Stored sufficient statistic

Lowest-rank full-local-quotient arm:

    a_p in R^4, p=0,...,599, for the scored frame of each pair.

- **[DERIVED]** Four spans the decision quotient at one penultimate patch; it is not a maximum global
  control rank.  Across pixels, the pullbacks B_p differ and their stacked spatial actuator can have rank
  greater than four.  `pair4` is therefore the smallest dimension capable of spanning every local head
  coordinate; actual rank(NB U)=4 must be measured rather than assumed, and it does not prove that four
  coefficients realize the whole frame.
- **[INFERRED]** a_p is not a copied GT label or a post-hoc camera delta.  It parameterizes a solved
  winner-cell displacement in the channel quotient, then becomes a jointly trained latent with the
  renderer and exact rate/quantization pressure.
- **[INFERRED]** Initialize a_p from the active-set cell projection, aggregated over the pair's legal
  Road-Lane gate.  In the first causal arm, joint descent changes a_p and the renderer while the solved
  U and banked final FiLM map remain frozen; receiver-weight adaptation is a separate rank-protected arm.
  Every update prices pixel collateral, pose, and exact sidecar bytes.
- **[REQUIRED BEFORE BUILD]** Stack the real banked conditioning-to-margin Jacobians across active n600
  pixels and measure their Fisher-weighted spectrum.  Select receiver rank r by the exact packed-byte
  marginal, retaining pair4 as the mandatory rank ablation.  If r>4, use a_p in R^r and U in R^(32xr);
  if spatial residual remains after rank expansion, add the world-dash/xi generator rather than hiding a
  missing spatial chart in channel width.

Three predeclared coding modes avoid an arbitrary choice:

1. **[DESIGN] pair4-int8:** 600 x 4 signed int8 coefficients plus four scales and a small parse-back
   header.  This is the lowest-rank full-local-quotient test.
2. **[DESIGN] pair-r-int8:** 600 x r signed int8 coefficients, with r selected from the stacked n600
   actuator spectrum and exact rate marginal.  This tests spatially varying channel reachability.
3. **[DERIVED DESIGN ESTIMATE] dash-seed:** the prior 0.9-1.8 KB world-anchor/xi estimate assumes
   25-30 m/s and excludes spline/registration overhead.  It may generate a_p and the spatial gate with
   only entropy-coded residual coefficients, but it has no byte-close authority.  This is the follow-on
   to test if pair-r works but its per-pair table is redundant.

### 5.2 Byte budget

- **[DERIVED raw accounting]** pair4-int8 coefficient bytes = 600*4*1 = 2400 B.  The coefficient-only
  score rate term is 25*2400/37545489 = about 0.00160.
- **[DERIVED raw accounting]** pair4-fp16 is 4800 B and rate term about 0.00320.  It is a diagnostic
  precision arm, not the default shipping form.
- **[DERIVED raw accounting]** Pair-r int8 costs 600r coefficient bytes and rate term about
  0.0003995r before basis/header.  This is an accounting identity, not permission to increase r.
- **[DESIGN BUDGET]** A shared 32xr int8 U costs 32r B (128 B at r=4); scales/header/checksum should keep
  the pair4 section near 2.5-2.7 KB if the receiver grammar stays compact.  Because U is bank/scorer-
  derived, it is counted whether stored as a sidecar matrix or folded into receiver weights.  This is a
  budget, not a measured archive size.
- **[UNMEASURED]** Brotli/entropy savings, exact archive framing, score gain per byte, and parse-back
  survival are all owed.  No 1-5 KB success claim exists before the exact packer row.

### 5.3 Receiver attachment and injection

**[MEASURED]** The banked v9c2 receiver already maps 32-dimensional per-frame code through film.weight
into four hidden-layer gamma/beta blocks.  The new branch must land inside this vehicle, not as a detached
helper.

Use an additive, identity-preserving scored-frame residual:

    q_p = U a_p,                       U in R^(32xr),
    (delta_gamma_p, delta_beta_p) = FilmFinalLinear(q_p) - FilmFinalLinear(0),
    h'_L(x) = h_L(x) + g_p(x) * [ delta_gamma_p elementwise h_L(x) + delta_beta_p ],

where r=4 in the pair4 arm, the base code c is unchanged, the flicker residual is exactly zero at a_p=0,
and g_p is a smooth renderer-native gate.  FilmFinalLinear reuses or derives from the final 192-row
gamma/beta slice of the in-vehicle film map; it does not send q_p through every earlier layer.

- **[DESIGN]** Inject after the final shared hidden block and before out_sdf/out_tex.  This is the Seg
  analogue of the safe pose-FiLM-v2 lesson: head-local, residual-contained, identity at init.  Earlier
  per-layer FiLM may remain as the base vehicle; the flicker-specific residual stays final/local so its
  collateral is attributable.
- **[DESIGN]** Build g_p from the renderer's own Road-Lane tie/annulus field, world-dash anchors, and xi
  advection.  One connected dash/boundary component shares one modulation.  There are no independent
  per-pixel stored controls.
- **[DERIVED]** Spatial coherence is structural because the stored object is r global coefficients
  and the support is generated by a smooth connected-component/world-coordinate gate.  This removes the
  independent-pixel formulation that caused superposition overshoot.
- **[NOT PROVEN]** Structural coherence is not structural collateral-freedom.  The gated residual still
  changes pixels throughout an ERF and shared pair normals still move other margins.  Full-cell inequalities
  and the joint loss must certify the result.
- **[REQUIRED]** The sidecar must carry unique receiver information: an a_p ablation must worsen the
  packed-byte verdict, and export must conditionally code or project away any a_p component already
  losslessly recoverable from the base code.  A second copy of existing per-pair data is not admissible.

Rank preservation is a hard design contract, not implied by the word FiLM:

- **[MEASURED warning]** Existing FiLM modulation participation rank collapsed from 3.34 in CE to 1.19
  at l7 (with 91.8% of variation in one axis).  That collapse co-occurred with other changes and is not
  itself a d_seg cause, but it falsifies nominal-width-as-deployed-rank.
- **[DESIGN, first arm]** Solve U from H_cond, orthonormalize it so U^T U=I_r, persist its hash, and keep
  U plus FilmFinalLinear frozen.  This makes the intended channel frame auditable and prevents a learned
  basis from rotating/collapsing during the side-information test.
- **[REQUIRED telemetry]** At every preserved stage report singular values and participation rank of
  a, q=Ua, and deployed (delta_gamma,delta_beta), plus the selected r.  Export drops/prices any dead
  column; nominal r without deployed participation is no custody.
- **[SEPARATE FORMULATION]** If frozen receiver directions cannot realize the target, a trainable receiver
  arm must use an explicit Stiefel/rank-floor contract and the existing DM1 telemetry.  Plain FiLMFix is
  insufficient because its default rank_floor_weight is zero.  StiefelW/CodeSpectralEntropy remain an
  isolated treatment, not silently folded into the causal first arm and not assumed score-positive.

### 5.4 Alignment to the real scorer channels

The renderer hidden h_L is not the frozen SegNet penultimate f.  The scorer is absent at inflate time.
Therefore no direct identification between a renderer channel and w_c-w_c' is allowed.

- **[DESIGN]** On the banked real n600 states, compute the streamed conditioning-to-feature Jacobian
  B_{p,x}=d f_{p,x}/d q_p through final residual -> render -> real R -> frozen SegNet penultimate.
  Define the margin Jacobian J_{p,x}=N_y B_{p,x}, and derive the
  conditioning basis U from the packed-byte-admissible leading eigenspace of

      K_y    = N_y M^-1 N_y^T,
      H_cond = sum_{p,x in active gates} J_{p,x}^T K_y^dagger J_{p,x}
             = sum_{p,x in active gates} B_{p,x}^T N_y^T K_y^dagger N_y B_{p,x}.

  Here K_y is the four-margin head Gram for target cell y.  The pseudoinverse whitens the coupled
  margin coordinates without pretending the pair normals are orthogonal.  Sum over all active spatial
  locations before selecting r; the resulting 32x32 H_cond can have rank greater than four even though
  every local head quotient is rank four.  This is a solve against the real bank, not a random basis and
  not a learned surrogate.
- **[DESIGN]** Train with the exact winner-cell hinge for all four rivals, a trust region on already-correct
  margins, and the existing score objective.  Quantize/dequantize a_p in-loop so the receiver lattice is
  the trained object.
- **[DESIGN / UNBUILT]** A phase-2 arm may add a frozen-SegNet stride-2-skip auxiliary target on the
  16-channel 192x256 feature map, weighted on Road-Lane gates.  The current trainer has no differentiable
  skip16 target/DSL consumer, so this is not part of the launchable flicker branch.  It requires its own
  CPU-torch intermediate-feature parity surface; no SegNet feature or weight may ship.
- **[MEASURED premise]** The skip target is justified by 6205/8072 Road-Lane flips under stride-2-detail
  ablation and by the 1.344 Road-Lane skip-detach gain ratio.
- **[NOT PROVEN]** There is no private Lane skip channel.  The measured 16-channel profiles are shared
  with Road-Undrivable.  The auxiliary must retain the full cell/trust-region loss.

### 5.5 Objective

For each real pair, optimize the existing witness action plus

    L_cell   = sum_x max_{k != y_x} relu(mu - (z_y_x-z_k)),
    L_keep   = sum_{x already correct} trust_region(all winner margins),
    L_skip   = RoadLaneGate * distance(skip16(render), skip16(GT)),  # phase-2 only; UNBUILT
    L_phase  = existing TieLocusDisplacement + PhaseAdvectionConsistency,
    L_rate   = exact quantized sidecar byte proxy, checked by real packer at stage boundaries.

- **[DERIVED]** L_cell is the soft constrained form of the exact polyhedral projection.
- **[INFERRED]** L_keep and the global d_seg term price spatial collateral that a single-pair solve cannot
  see.
- **[MEASURED/Built]** TieLocusDisplacement and PhaseAdvectionConsistency already exist as typed DSL
  levers and reuse the shared phase primitives.
- **[DESIGN]** Loss weights change only at stage boundaries and must be derived from matched gradient
  share/provenance; this memo intentionally supplies no bare numeric weight.

## 6. Composition with pose FiLM and comparison with #425

### 6.1 Pose composition

- **[MEASURED, other vehicle]** Quantizr/HNeRV pose-FiLM demonstrates the structural pattern “stored
  low-dimensional side information -> receiver feature modulation,” but its low pose distortion is bound
  to a photometric neural decoder.  It is not a v9 witness score transfer.
- **[MEASURED, other vehicle]** Shared-stem pose FiLM changed d_seg badly in the old small-n probe; the
  later residual pose-FiLM-v2 routes only to the SegNet-invisible frame-0 head and makes frame-1
  bit-invariant to pose input in that HNeRV vehicle.
- **[DESIGN]** Keep two actuator heads even if one packet stores both conditions:
  - pose condition -> frame-0/head-local residual only;
  - flicker condition -> scored-frame boundary-gated residual only.
  The conditioning container may concatenate metadata, but its routing matrix must be block-separated.
- **[NOT PROVEN]** The v9 witness does not inherit the ancestor 3.4e-5 pose number.  The paired n600 test
  must report d_pose and reject any Seg gain whose pose contribution makes net score worse.

### 6.2 #425 is complementary, not dominated

- **[MEASURED]** #425 stores 13,222 ground-class residuals in a 10,682 B zlib9 section, q=1/64,
  RMSE 0.06568 px, bit-identical parse-back.  Its recovered d_seg is explicitly owed.
- **[DERIVED]** #425 is a high-spatial-fidelity curve/phase representation.  The FiLM sidecar is a
  low-channel-rank receiver-conditioning representation.  One can seed the FiLM/gate from #425's xi
  predictor without applying its decoded values as frozen camera edits.
- **[INFERRED]** FiLM is better aligned with joint collateral pricing and has a smaller raw target budget;
  #425 may be better when per-dash spatial variation exceeds the selected low-rank receiver's capacity.
- **[NEGATIVE, formulation scope]** Frozen-output, independent-pixel STORE/APPLY is closed by the labeled
  diagnostic.  Coherent curve-domain #425 consumption and train-side FiLM remain open; this memo does not
  kill the carrier family.

## 7. Correct n600 train-side design — v10 section 14.2 build ticket

Status: **NOT_LAUNCHABLE / PRE-REGISTERED DESIGN, NOT FIRED.**  The proposed receiver and skip target
do not yet have typed consumers or a compiled config receipt.  Operator-GO alone is insufficient until
the gates below make the ticket executable.

### 7.1 Build gates before any run

1. **[REQUIRED — UNBUILT]** Add one typed DSL Lever/LawRef/parser/receiver receipt for the in-vehicle
   channel-aligned flicker branch.  Existing FiLMFix configures the current FiLM path; it does not create
   this final residual branch.  This memo mints no ad-hoc argv.  TieLocusDisplacement and
   PhaseAdvectionConsistency remain composed through the DSL only.
2. **[REQUIRED — SEPARATE UNBUILT CONSUMER]** FrS additionally needs a differentiable CPU-torch
   stride-2-skip feature target with its own typed training consumer/parity receipt.  No such skip16 loss
   exists in the current trainer/DSL.  Defer FrS to build phase 2 unless this second surface lands cleanly;
   one flicker Lever cannot pretend to cover both mechanisms.
3. **[REQUIRED]** Before any launch label, emit a compiler receipt containing the exact parent-bank hash,
   recorded deterministic seed, selected r and U hash, all LawRefs, real parser/consumer reachability,
   immutable per-arm output directories, sequential order, and exact compiled argv.  This memo is not
   that receipt.
4. **[REQUIRED]** Register every additive U/a/gate state in the canonical resume registry; old banked
   checkpoints restore a_p=0 and the identity branch.  Save EMA shadow, optimizer state, exact stage/update
   position, quantizer state, and preserved per-stage checkpoints atomically.
5. **[REQUIRED]** Receiver pack/unpack consumes the exact a_p bytes and proves ON changes output, OFF and
   zero-code are byte-identical, and parse-back reconstructs the trained quantized values.
6. **[REQUIRED]** The scorer-feature/cell target is training-only.  Receiver closure must prove no scorer
   weights, GT argmax table, or per-pixel oracle data enters the archive.
7. **[REQUIRED]** Reuse the storage waterfall and success-only scratch cleanup.  Any rendered raw bulk goes
   to the SSD tier with a reproducibility manifest; no durable evidence path points at /tmp.

### 7.2 Matched arms

All arms use the exact bank hash above, weights-only warm start with fresh optimizer, the same
compiler-recorded seed, pair order, update count, R implementation, stage transitions, checkpoint
cadence, and frozen scorer.

| arm | phase/cell losses | flicker receiver | purpose |
|---|---|---|---|
| C: joint control | existing TieLocusDisplacement + PhaseAdvectionConsistency; matched base action | identity/off | measures ordinary joint phase descent from the same bank |
| N: pair-invariant clip-global control | identical to FiLM arm | same frozen U/final map/gate, but one learned coefficient vector tied across all pairs; deterministic counted padding matches the tested FiLM archive bytes | measures branch/gate capacity plus clip-global conditioning, while withholding pair-indexed information |
| F4: local-quotient FiLM | identical to C | pair4-int8 final residual + Road-Lane coherent gate | tests the smallest full local decision-coordinate arm |
| Fr: stacked-rank FiLM | identical to C/F4 | pair-r-int8 using the preregistered stacked-J rank r; collapses to F4 when r=4 | tests whether spatial actuator rank beyond four is load-bearing |
| FrS: stacked-rank FiLM + skip target | identical to Fr | identical bytes/receiver to Fr; adds the separately built training-only skip16 auxiliary | phase-2 test of whether the measured bottleneck improves realization rather than target geometry |

- **[DERIVED]** N versus C measures branch/gate capacity plus learned clip-global conditioning, not a
  no-video-information control.  F4/Fr versus N isolates the incremental value of pair-indexed
  side information beyond that clip-global control.  Fr versus F4 measures stacked spatial rank beyond four.
  FrS versus Fr measures skip supervision at zero receiver-byte change.
- **[REQUIRED READBACK ABLATIONS]** For every trained FiLM arm, score the same weights and same packed
  sidecar with (a) real codes, (b) code consumption forced to zero, and (c) a generic fixed cyclic
  pair-index permutation.  Real codes must beat both.  Reordering/ignoring happens in the diagnostic
  receiver so the counted bytes stay identical; no video-derived permutation ships.
- **[REQUIRED]** If the stacked-J selection returns r=4, F4 and Fr are the same arm and must be executed
  once, not duplicated.  Any r>4 is selected before training from the banked Jacobian spectrum plus exact
  byte marginal; it is not tuned from the verdict.
- **[REQUIRED]** The window length is derived at compile time as the maximum of the EMA-settling interval,
  three full-verdict opportunities, and the configured phase-stage completion event.  Do not insert an
  arbitrary epoch literal.  Stop all arms at matched optimizer updates and preserve every stage end.

### 7.3 Real verdict harness and observability

- **[REQUIRED]** Training uses the existing differentiable real-R path; verdicts render exact uint8 camera
  frames and run the frozen CPU-torch SegNet with the pinned preprocess.  Full n600, batch geometry 32,
  is the advisory verdict surface; smaller samples are liveness only.
- **[REQUIRED]** Reuse the decode/score path exercised by
  tools/probe_phase_stack_efficacy_road_lane.py and tools/levelset_byte_close_and_eval.py, but consume the
  new branch through the actual receiver.  --skip-parity is forbidden for the efficacy verdict.
- **[REQUIRED]** Persist for every stage: total d_seg; directed 5x5 flip matrix; Road-to-Lane and
  Lane-to-Road counts; per-class d_seg; d_pose; exact section/archive bytes; pose/rate/seg score terms;
  active-set cell violations; G_head plus stacked spatial G_act summaries; a/q/modulation spectra and
  participation ranks; cell IoU/components; pair-boundary displacement/F1; triple-junction incidence;
  skip16 response when that separate consumer exists; uint8 round-away fraction; runtime/RSS;
  checkpoint/archive hashes.
- **[REQUIRED]** Score the exact packed sidecar bytes, not live fp32 a_p.  Report marginal Seg delta,
  nonlinear pose-term delta, byte/rate delta, and value per byte.

### 7.4 Memory, storage, and c2 containment

- **[MEASURED prior]** The earlier per-frame CPU diagnostic peaked around 3.1 GiB RSS; the current live c2
  process is a large-memory training run.  The former does not prove a second MLX trainer is safe.
- **[DECISION]** All joint-training arms are **QUEUED behind c2 by default** and run sequentially,
  single-flight.  They are not declared co-run-safe, and nothing in this design may read/write the sacred
  c2 run directory.
- **[REQUIRED]** Claim each arm separately through the governed lane-dispatch helper and write to an
  immutable arm-specific directory of the form
  `experiments/results/v10_channel_film_<arm>_<utc>/`; never reuse a run directory between C, N, F4, Fr,
  or FrS.  The compiled receipt fixes the exact resolved paths before the first arm starts.
- **[REQUIRED]** At fire time, the governed launcher performs the actual memory/storage admission check.
  Preserve the repository's >=10 GiB fail-closed memory floor and SSD-first waterfall; a governor REFUSE is
  the result, not permission to bypass it.
- **[DESIGN]** Stream n600 render/score batches and stage outputs rather than retaining a 3.66 GB raw video
  locally.  Any unavoidable raw material is certified and externalized to the SSD tier automatically.

### 7.5 Pre-registered verdict

**PROCEED to the longer v10 stage only if all conditions pass:**

1. **[MEASURED REQUIRED]** At least one of F4/Fr beats the architecture/byte-matched N arm on
   deterministic full-n600 packed score at matched update count, and the sign persists on at least two
   preserved post-settle checkpoints.  C versus N separately attributes any branch-only effect.
2. **[MEASURED REQUIRED]** The Road-Lane reduction remains after subtracting all new flips; no other
   class-pair or already-correct-cell regression reverses the total gain.
3. **[MEASURED REQUIRED]** Exact receiver parse-back retains the sign and

       100*Delta_dseg + Delta_sqrt_10_dpose + 25*Delta_bytes/37545489 < 0.

4. **[MEASURED REQUIRED]** Every admitted FiLM arm beats its same-weight forced-zero and fixed-permuted
   readbacks on the identical counted bytes; zero condition is output-byte-identical to the OFF branch;
   resume round-trip and every stage checkpoint are complete.
5. **[MEASURED REQUIRED]** Fr must beat F4 after exact extra-byte pricing to justify r>4.  FrS may replace
   Fr only if it improves the same full-n600 packed-byte verdict; a lower skip-feature loss alone is not
   admission evidence.

**REVISE, do not family-kill, on any failure:**

- **[FORMULATION scope]** If head-cell loss improves but packed through-R d_seg does not, revise the
  renderer injection/gate using the measured range(NB) residual.
- **[FORMULATION scope]** If target flips improve but other pair margins cross, use the full active-set
  QP/trust region and measured G_act signs rather than a single normal.
- **[FORMULATION scope]** If pair-r succeeds but bytes are redundant, project to the dash-seed/xi mode.
- **[FORMULATION scope]** If pose worsens, strengthen frame/head separation and fine-chroma routing; do not
  transfer the HNeRV pose result.
- **[INSTANCE scope]** A loss at this bank/short window does not kill channel FiLM, coherent curve carriers,
  or the hyperplane-cell formulation.

## 8. Candidate canonical law and triality

### 8.1 Coupling/exploitability law

Candidate equation id: channel_cell_projection_actuator_gram_v1

    C_y(mu) = intersection_{k != y} {f : n_yk^T f + beta_yk >= mu}
    delta_f* = M^-1 N_A^T (N_A M^-1 N_A^T)^dagger r_A
    G_head       = N M^-1 N^T
    G_act,p      = (N_p B_p) H_u^-1 (N_p B_p)^T
    J_stack      = stack_{p,x}(N_{p,x} B_{p,x})
    G_act,stack  = J_stack H_u^-1 J_stack^T
    globally reachable iff r_stack is in range(J_stack)
      and all inactive cell inequalities retain slack at every stacked location.

- **[DERIVED]** The first two lines are exact corollaries of the registered rank-4 linear-head law.
- **[DERIVED]** G_head gives same-patch directed pair coupling; the off-diagonal spatial blocks of
  G_act,stack give global pair/location coupling.  B inserts actual receiver reachability and separates
  target geometry from realization.
- **[FORMALIZATION_PENDING]** Do not register a new empirical canonical equation yet.  The exact head
  portion already lives under segnet_head_rank4_linear_flipdist_v1; stacked spatial G_act has no n600
  anchor.  Register
  this multi-constraint/actuator extension only after the preregistered n600 test persists its Gram and
  reachability receipt.

### 8.2 DAG FEED

FEED-channel-film-flicker (2026-07-18):

    consumes:
      FEED-segnetfractal -> exact rank-4 head, ERF, skip16
      FEED-lane-gain     -> head-amplified / input-attenuated Road-Lane chain
      FEED-necessity     -> n600 cells, edges, saddles, range/ker split
      FEED-cg/fh #180    -> standalone MS rate negative; residual-only survival
      #315/#360          -> class-birth admissibility + edge-normal placement
      #365/#369          -> scoped pose-warp and raster-edit formulation negatives
      FEED-424gate       -> diagnostic only; post-hoc independent-pixel mis-test
      FEED-phase-carrier-build / #425 -> coherent xi residual codec and byte row
    derives:
      head cells -> scorer pullback complex -> MS residual seed/gate
      pair normals -> multi-constraint GT-cell projection -> stacked spatial actuator Gram
      code32 -> selected-r orthonormal U -> gated final residual FiLM -> exact packed sidecar
    emits:
      NOT_LAUNCHABLE v10 section 14.2 ticket C/N/F4/Fr/(phase-2 FrS), queued behind c2
      equation candidate channel_cell_projection_actuator_gram_v1
      duty-to-measure: n600 stacked G_act, MS seed bytes, packed d_seg, pose, directed collateral
    pointer_delta: 0

### 8.3 Six-hook wire-in status

1. **Sensitivity map — DESIGN:** n600 stacked spatial G_act and per-pair active-set slack become the canonical
   channel-control sensitivity receipt.
2. **Pareto — DESIGN:** admission uses exact Seg, nonlinear Pose, and exact-byte deltas on the same
   packed artifact.
3. **Bit allocator — DESIGN:** pair4-int8 versus selected pair-r versus fp16 diagnostic versus dash-seed
   is selected by measured score-unit value per byte.
4. **Cathedral/autopilot — QUEUED:** no dispatch hook is enabled; the governed ticket remains behind c2 and
   operator-GO.
5. **Continual learning — OWED ON RESULT:** append the n600 Gram/reachability and proceed/revise row to the
   probe outcome/costate consumers in the same landing as the measurement.
6. **Probe disambiguator — PRE-REGISTERED:** C/N/F4/Fr/(phase-2 FrS) separates branch capacity,
   pair-specific information, local head rank, stacked spatial actuator rank, and skip16 supervision;
   real/zero/permuted readback proves consumption; pair-r versus dash-seed separates channel rank from
   missing spatial grammar.

DSL leg: existing FiLMFix, TieLocusDisplacement, and PhaseAdvectionConsistency are real typed levers.
The flicker branch is design-only and needs one new typed Lever/LawRef/parser/receiver receipt.  The
skip16 auxiliary is a distinct unbuilt training consumer and needs a second typed receipt or remains
deferred; no raw flag is authorized by this memo.  Equations leg: exact parent equation reused; actuator
extension pending n600.  DAG leg: the FEED above is the durable bridge in this artifact.

## 9. Round-1 adversarial self-review

### Attack 1: “Rank four proves low-rank global collateral.”

**Finding: false.**  **[DERIVED]** Rank four is local decision rank per penultimate patch.  Overlapping ERFs
make the frame Jacobian spatially extended and potentially high rank.  The memo now calls only the head
quotient rank four and makes n600 stacked spatial G_act the global evidence gate.

### Attack 2: “Moving along one pair normal is collateral-free by construction.”

**Finding: false without conditions.**  **[DERIVED]** Pair normals have a non-diagonal Gram and exact cycle
dependencies.  A single-normal projection is valid only when the projected segment crosses one active
upper-envelope facet while every other winner inequality retains slack.  The design therefore targets the
whole GT cell with an active-set QP and a keep-region trust term.

### Attack 3: “FiLM cannot per-pixel overshoot.”

**Finding: overclaim.**  **[DERIVED]** A global coefficient plus coherent gate removes independent stored
pixels, but a shared modulation still changes many rendered pixels and those changes overlap inside the ERF.
FiLM is collateral-priced, not collateral-free, until the packed n600 verdict proves it.

### Attack 4: “Four stored scalars are sufficient because the head rank is four.”

**Finding: only patch-local channel-sufficient.**  **[MEASURED/DERIVED]** Road-Lane has about 21
components/frame, and stacked B_p can have rank greater than four.  The preregistered Jacobian spectrum,
F4/Fr ablation, and pair-r/dash-seed disambiguator prevent local rank from becoming global architecture.

### Attack 5: “The 0.0011 Road-Lane amount is an n600 ceiling.”

**Finding: false transfer.**  It is retained only as a sample-local n24 counterfactual magnitude.  The
n600 Road-to-Lane error count, numerical ceiling, and reachable amount are all unknown in consulted
custody, and the memo reports them that way.

### Attack 6: “The large ker(A) is the exploitable correction manifold.”

**Finding: false.**  ker(A) is scorer-invisible and cannot move the target.  It is solution gauge freedom;
the useful visible lift lies in range(A^T) and must also satisfy cell, actuator, lattice, and pose constraints.

### Attack 7: “Quantizr proves this sidecar will work.”

**Finding: false transfer.**  Quantizr supplies the receiver-conditioning pattern, not a v9 result.  Its
pose behavior is decoder/vehicle-specific; the flicker branch needs its own exact receiver, n600 Seg, Pose,
and byte evidence.

### Attack 8: “The n24 independent-pixel probe killed phase.”

**Finding: false at family scope.**  Its post-hoc application is a formulation negative and its
phase-addressability diagnosis motivates the correct test.  It is not an n600 efficacy verdict and is not
used here to rank or kill joint FiLM/coherent-curve forms.

### Attack 9: “The scorer hyperplane pullback equals the witness phi tie set.”

**Finding: false unless calibrated at that locus.**  The scorer boundary is the co-maximum pullback
through render, R, and frozen SegNet.  The native phi tie set is the level-set actuator scaffold.  Their
disagreement is residual d_seg, so the design measures both instead of identifying them.

### Attack 10: “Triple ties are automatically Morse critical points and the MS seed is compact.”

**Finding: two overclaims.**  Triple ties are codimension-two stratified junctions; a classical Morse
critical label needs a potential and nondegeneracy/transversality proof.  Existing full/rare arc proxies
cost about 8.1x/3.5x the current packed archive.  A true critical-trajectory codec is unmeasured, so the
representation remains residual/seed composition rather than a standalone replacement.

### Attack 11: “Nominal FiLM width is deployed rank.”

**Finding: contradicted by prior telemetry.**  Existing modulation participation fell 3.34 to 1.19.
The first arm freezes an orthonormal solved U and banked final map, measures a/q/modulation spectra, and
prunes dead exported columns.  Trainable receiver adaptation is a separate Stiefel/rank-floor treatment.

### Attack 12: “F versus the legacy control proves side-information value and the ticket can fire.”

**Finding: false.**  N is an architecture/byte-matched pair-invariant clip-global control, not a
no-video-information control; F versus N isolates only the incremental pair-indexed information.  The
real/zero/permuted same-weight readbacks are also required to prove consumption.  The ticket is
NOT_LAUNCHABLE until the flicker receiver has a typed
compiler/parser/consumer receipt; FrS additionally needs a distinct unbuilt skip16 training consumer.

## 10. Stores consulted and final custody

STORES CONSULTED: docs/operating_manual_craft_handoff.md; CLAUDE.md; AGENTS.md; PROGRAM.md;
SPEC_v75_optimal_single_trunk_20260708.md section 8; SPEC_v8_perclass_decomposition_20260708.md;
graph-memory recall for collateral/channel/rank4/FiLM/#424/#425/#524; reports/latest.md;
lane_registry.json; subagent_progress.jsonl; canonical_task_status.jsonl;
canonical_equations_registry.jsonl; both live inboxes through operator directives
2026-07-18T04:02:45Z, 2026-07-18T04:08:50Z, and 2026-07-18T04:15:38Z;
frozen_scorer_exact_factorization_20260715.md;
segnet_recursive_fractal_factorization_20260715.md; lane_channel_deep_refactorization_20260716.md;
necessity_solver_inverse_factorization_20260715.md; null_subspace_rate_measure_20260717.md;
deepmath_lens_tropical_ot_powerdiagram_20260704.md; tropnnc_311_20260712T010936Z.md;
morse_smale_partition_codec_feasibility_20260626.md;
dynamical_partition_optimal_form_morse_smale_nca_FEED-fh_20260627.md;
projection_unification_and_eight_lenses_20260715.md; dmtz_taskaware_rate_lever_design_20260709.md;
pose_taskspace_native_morse_smale_depth_warp_design_20260708.md and canonical measured #365 equation;
p0_forces_derivation_20260708.md plus #315/#360 DSL consumers;
phase_stack_efficacy_probe_v10_gate_20260718.md (diagnostic scope only);
p0_425_phase_carrier_byte_close_row_20260716.md; p0_444_v9_stage_byte_close_banking_20260716.md;
pose_sidecar_reuse_assessment_20260708.md; pose_film_cpu_disambiguator_20260612.md plus its result JSON;
lever3_optimal_quantizr_pose_solve_port_trackA_20260613.md; current typed DSL and trainer parser/receiver
consumers; banked v9c2 EMA bytes and tensor manifest.

**Review custody:** two read-only fresh-eyes passes were completed.  The final surgical recheck was
**CLEAN** after correcting conditioning-Jacobian dimensions, the pair-invariant clip-global control's
causal label, and local-versus-stacked actuator-Gram scope.

**Final state: [MEANS] pointer unchanged; no training/dispatch; no sacred-run mutation; ticket remains
NOT_LAUNCHABLE; lane remains L0 research-only.  MAIN landing review is required before this design is
merged or converted into a build.**
