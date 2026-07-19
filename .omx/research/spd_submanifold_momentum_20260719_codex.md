# SPD-submanifold momentum crosswalk (#552)

**Date:** 2026-07-19

**Lane:** `spd_submanifold_momentum_20260719` (L0, `research_only=true`)

**Authority:** operator drop, [arXiv:2302.09738](https://arxiv.org/abs/2302.09738)

**Author implementation inspected:** [yorkerlin/StructuredNGD-DL](https://github.com/yorkerlin/StructuredNGD-DL), commit `288b20cae250dd1b0f885133130139ec17023326`

**Score authority:** none; no launch, archive, score, or pointer mutation

**Pointer delta:** `0.1910828242 [contest-CPU]` **UNMOVED — MEANS**

**Landing authority:** this memo and its lane-registry row require independent MAIN review before landing or implementation.

## Executive verdict

**ADOPT-AS-A-TYPED, DEFAULT-OFF #509 CONVERGENCE SPEC; REJECT AS A #496 RATE LEVER AND AS A REPLACEMENT FOR THE CURRENT RANK-4 HEAD SOLVER.** The paper gives a credible inverse-free, matrix-multiply-only way to maintain dense or structured SPD preconditioners in moving normal coordinates. It does not make optimizer factors part of the shipped witness, does not establish low-precision MLX behavior, and does not outperform the existing analytic categorical-Fisher head inverse on paper. The clean future composition is SPD geometry for an SPD factor and spectral/tangent Muon for an orthogonal factor; applying both blindly to the same weight-space momentum is not geometrically identified.

## Evidence labels and scope

- **MEASURED** means a value exists in a repository artifact or was read from the exact source snapshot named above.
- **DERIVED** means an algebraic or complexity consequence of those sources.
- **INFERRED** means an implementation prediction that still needs a custodied probe.
- Every negative below is formulation-scoped. None kills structured natural-gradient, Muon, or polar-factor families.

## Ranked crosswalk

| Rank | Proposal | Verdict | Consumer | Falsifiable gate |
|---:|---|---|---|---|
| 1 | Kronecker GNC preconditioner as an optimizer-only, default-OFF #509 treatment | **SPECIFY, DO NOT FIRE** | typed witness DSL + optimizer owner; use the existing #509 wall-clock/convergence lane, not a new fifth #496 arm | NumPy-fp32 step/resume authority; MLX-fp32 parity; no solve/eigh/CPU fallback; sealed single-difference n24 screen; positive n600 confirmation before adoption |
| 2 | Dense 4x4 GNC factor for the SegNet quotient head | **DO NOT A/B NOW** | current `HeadNaturalGradient` owner | Reopen only if measured nonstationary aggregate covariance shows the analytic per-pixel inverse is inadequate or unstable |
| 3 | SPD momentum on the positive factor of a polar chart, with Muon/SPEL on the orthogonal factor | **DEFER, FAMILY OPEN** | MuonH/SPEL owner after the existing Q-only treatment has authority | exact reconstruction and split-resume parity; separate Q/P telemetry; sealed n24 negative filter, then n600 confirmation without rate/Pose regression |
| 4 | Diagonal/block-diagonal controls, then Kronecker only where measured covariance supports it | **MEASURE STRUCTURE FIRST** | covariance producer + bit/convergence allocator | held-out covariance approximation and preconditioned-gradient fidelity beat diagonal under equal state/step budgets |
| 5 | Generic low-rank-plus-diagonal or hierarchical SPD factor | **NO IMPLEMENTATION RECOMMENDATION** | no current consumer | a fixed connected matrix subgroup, closure/retraction proof, and measured spectrum/covariance custody must exist first |

`verdict_scope` for ranks 2 and 5: **CURRENT FORMULATION + CURRENT DATA CUSTODY ONLY**. A failed dense-head replacement or absent hierarchical proof is not a negative on the broader SPD-submanifold family.

## 1. Exact paper mechanism and the #496 low-precision question

### 1.1 Full-SPD moving normal coordinate

For an SPD quantity `tau`, the paper uses an inverse factor

```text
tau = A A^T,
A(eta) = A_cur exp(eta / 2),    eta = eta^T.
```

At the local origin, the affine-invariant metric is Euclidean in `eta`. The local gradient and momentum step are

```text
g_eta = A_cur^T g_tau A_cur,
m     <- alpha m + beta g_eta,
A_new <- A_cur exp(-m / 2).
```

**DERIVED:** under the paper's approximated Euclidean transport and moving-coordinate transform, symmetry makes the required Jacobian-vector transform the identity for this coordinate (the appendix also shows its retained higher-order correction vanishes in this case). Momentum can therefore remain in the local symmetric coordinate without an inverse, eigendecomposition, or linear solve. This is the paper's practical GNC momentum update, not a claim that exact Riemannian parallel transport is globally the identity.

The paper's guaranteed truncation is

```text
h2(N) = I + N + 1/2 N^2,
```

which is nonsingular for real symmetric `N`. The released deep-learning code instead uses the cheaper linear factor update `I + N`. **MEASURED SOURCE CUSTODY:** the authors report the linear approximation works in their experiments, but the released code does not supply a general nonsingularity certificate. A Pact spec should therefore make the quadratic update the correctness control and allow the linear update only as a measured treatment with a factor-condition refusal.

### 1.2 Released Kronecker deep-learning update

For a layer parameter matrix `mu in R^(d x p)`, the released structured inverse factor is `A = K tensor C`, with input covariance `Sigma_AA in R^(p x p)` and output-gradient covariance `Sigma_GG in R^(d x d)`. The paper's update is

```text
H_K = K^T Sigma_AA K
H_C = C^T Sigma_GG C
kappa^2 = lambda Tr(K^T K)
c^2     = lambda Tr(C^T C)

m_K <- alpha1 m_K + beta1/(2d) [Tr(H_C) H_K + c^2 K^T K - d I_p]
m_C <- alpha1 m_C + beta1/(2p) [Tr(H_K) H_C + kappa^2 C^T C - p I_d]

K <- K exp(-m_K)    (released code uses K(I - m_K))
C <- C exp(-m_C)    (released code uses C(I - m_C))

M_mu <- alpha2 M_mu + C C^T grad_mu K K^T + gamma mu
mu   <- mu - beta2 M_mu.
```

**MEASURED SOURCE CUSTODY:** the public implementation is PyTorch-specific and its optimizer path uses matrix multiplies, reductions/traces, identity construction, and additions. It contains no mixed-precision/autocast policy and hard-codes CUDA identity placement in the inspected path. It is evidence for the mathematics and the PyTorch treatment, not evidence of MLX or low-precision parity.

### 1.3 Operations, precision, and residency

- **DERIVED operations:** dense factor maintenance costs cubic matrix products in the factor widths, `O(p^3 + d^3)` per factor refresh; applying the two-sided preconditioner costs `O(dp(p+d))`. Refresh period `T` amortizes only factor maintenance, not the two-sided parameter update. There is no solve, inverse, eigendecomposition, QR, or SVD in the released treatment.
- **INFERRED MLX feasibility:** all required primitives have native MLX equivalents. This is materially more MLX-feasible than the repository's generic Helmert quotient solver, whose MLX path currently assembles and solves in CPU float64 because the required Metal solve/parity surface is absent.
- **Required factor residency:** `K`, `C`, `m_K`, `m_C`, covariance accumulators, and parameter momentum are optimizer/checkpoint state. Keep factor state and covariance reductions in fp32 for the first authority implementation. Eligible large products may be A/B-tested in bf16 only against a full-fp32 reference. No factor belongs in `archive.zip` or `inflate.py`.
- **Resume contract:** checkpoint every named state above plus refresh counter, optimizer step, dtype policy, structure ID, damping, and both momenta atomically at every stage boundary. Split-run output and optimizer state must match the uninterrupted NumPy-fp32 authority within the declared deterministic tolerance.

### 1.4 Rate verdict

**MEASURED local prior (#496):** int8+brotli is already at 52.6 KB / 6.54 bits per parameter, equal to the measured iid entropy; int5 QAT/LSQ improved `d_seg` only 9.5% while leaving hard walls near 0.0042; sensitivity waterfilling had net `Delta S = +0.114`, and every tested int8-to-int5 tensor move increased `d_seg`.

**VERDICT:** this paper does **not** beat post-hoc quantization as a direct rate lever. Optimizer factors are unshipped state, so changing their precision changes memory traffic and possibly wall clock, not archive bytes. A better basin could indirectly improve the exact post-quantization witness, but that is a convergence hypothesis and must be measured through the same quantize/package/parse-back/R/scorer chain.

`verdict_scope`: **THE CURRENT INT8 WITNESS AND DIRECT ARCHIVE-RATE CLAIM ONLY.** Low-precision wall-clock work (#509), a future below-int8 vehicle, int4 plus outliers, and basis-changing representations remain open.

### 1.5 Consumer and gate

Consumer: the existing #509 convergence/wall-clock owner, through a typed, default-OFF DSL treatment. Do not add a fifth arm to the settled #496 four-arm matrix. An owner may replace or sequence a held arm only after a new compiled design receipt.

Falsifiable staged gate:

1. A deterministic NumPy-fp32 reference produces byte-identical serialized optimizer state on repeat and split-resume runs.
2. MLX-fp32 matches every factor, momentum, and parameter update to the repository parity threshold (`>= 0.9997`) and invokes no CPU solve/eigh fallback.
3. A compiler-emitted n24 treatment differs from the sealed control only in this optimizer policy. Use identical seed, pair order, stage boundaries, checkpoint cadence, and downstream quantization.
4. Report median seconds/step with uncertainty, peak resident bytes, factor condition/refusal counts, per-class `d_seg`, total `d_seg`, `d_pose`, exact quantized archive bytes, and score-unit delta per byte. n24 is a negative filter only.
5. Any positive n24 result remains non-authorizing until a resumable n600 run reproduces the advantage through exact parse-back/R and preserves Pose/rate facets.

## 2. Rank-4 head versus the current natural-gradient implementation

### 2.1 Current repository geometry

**MEASURED:** the frozen five-class SegNet head has an exact centered rank-4 quotient. Its four nonzero singular values are `3.128, 2.154, 2.025, 1.796` (condition `1.74`), and its ten pair normals span angles from `25.8` to `90` degrees. The current `HeadNaturalGradient` uses the categorical Fisher

```text
G(p) = diag(p) - p p^T
```

and, for a zero-sum cotangent `v`, applies the exact quotient pseudoinverse in `O(K)`:

```text
G(p)^+ v = v/p - mean(v/p) 1,
```

with explicit damping through `p + eps`. It pushes this preconditioned logit cotangent back to all witness parameters. It holds constant state and performs no matrix solve.

### 2.2 Costed comparison

| Surface | Current analytic head NG | Dense rank-4 GNC factor |
|---|---:|---:|
| Per-pixel Fisher action | `O(K)` divisions/reductions | at least `O(4^2)` dense action after covariance construction |
| Persistent geometry state | constant | factor + factor momentum + covariance, all `O(4^2)` |
| Matrix factor refresh | none | `O(4^3)` dense products/truncated exponential |
| Captures exact categorical per-pixel metric | yes | only an aggregate/learned SPD approximation unless rebuilt per pixel |
| MLX solve | none | none for GNC |

**DERIVED VERDICT:** the paper loses on paper as a replacement for the existing head solver. Therefore the requested n24 head A/B is **not owed now**. This is a rejection of redundant dense factor maintenance, not of natural gradient.

Consumer/reopening gate: the `HeadNaturalGradient` owner may reopen a 4x4 GNC control only after a custodied trajectory shows one of: unstable analytic damping, a nonstationary aggregate covariance objective not represented by the pointwise Fisher, or a measured wall-clock bottleneck in the current `O(K)` action. Then the A/B must compare exact downstream metrics and state cost, not proxy loss alone.

`verdict_scope`: **DENSE GNC AS A DROP-IN REPLACEMENT FOR THE PRESENT FIVE-CLASS ANALYTIC ACTION.**

## 3. Relationship to Muon and legal composition

### 3.1 They are not the same optimizer

**MEASURED:** current Pact Muon applies an NS5 approximation to the matrix sign/polar factor of weight-space momentum for six trunk matrices. It normalizes singular values and retains the polar orientation. The MuonH/SPEL treatment charts `W = Q H0`, keeps `H0` frozen, maps the weight gradient to `Q`, projects to a Stiefel tangent, applies spectral momentum, retracts by QR, and folds back `Q H0`.

The paper instead updates an SPD inverse-preconditioner factor by the affine-invariant metric in a moving symmetric normal coordinate. Its momentum lives in the symmetric Lie algebra, not the Stiefel tangent and not the raw weight-momentum space.

**DERIVED:** “Muon is natural gradient” remains a false equivalence here. Both use matrix functions and geometry, but they act on different manifolds with different metrics and invariants.

### 3.2 Composition options

1. **Clean product-manifold composition:** write a full-rank weight chart as `W = Q H`, with `Q` orthogonal/Stiefel and `H` SPD. Apply Muon/SPEL only to `Q`; apply GNC momentum only to `H` (or its SPD preconditioner factor). Maintain separate momenta and retractions. This is geometrically legible.
2. **Same-gradient serial composition:** first two-side-precondition `grad_W` with SPD factors and then apply NS5 Muon. This is not equivalent to either source method: NS5 discards singular magnitudes while the SPD action can still rotate singular vectors. The net treatment may erase or distort the scale information GNC introduced. It needs an independent equation and A/B, not an assumed composition claim.
3. **Current repository constraint:** because existing SPEL freezes `H0`, immediately adding an `H` update would change the treatment's function-preserving premise and create a new arm. It must not be silently folded into the existing Q-only control.

Recommendation: do not compose on the same matrix now. After a Q-only treatment earns authority, specify a separate `FilmPolarSPDNormalMomentum` treatment on the positive factor. Consumer: MuonH/SPEL owner. Gate: exact `QH` reconstruction at initialization, independent Q/H gradient and momentum telemetry, stage-boundary split-resume parity, bounded factor condition, and a compiler-emitted same-budget Q-only versus Q+H n24 negative filter through quantization/R. A positive n24 result still requires n600 confirmation with no Pose/rate regression.

`verdict_scope`: **UNIDENTIFIED SAME-MOMENTUM SERIAL COMPOSITION AND THE CURRENT FROZEN-H0 ARM ONLY.** The product-manifold family remains open.

## 4. Structured SPD menu versus current geometry

The paper's actual demonstrated menu is narrower than the phrase “arbitrary structured covariance” suggests:

- full symmetric SPD;
- diagonal or triangular connected subgroups;
- the paper-named “rank-one SPD submanifold,” whose SPD matrix is diagonal plus a structured rank-one block (not a generic rank-deficient covariance);
- a Heisenberg structured subgroup;
- Kronecker factors with a blockwise approximate metric, whose factors may themselves use valid sparse subgroups.

The released deep-learning code implements the Kronecker case. It does not provide a generic low-rank-plus-diagonal or hierarchical MLX optimizer. Any hierarchical proposal must first prove that its fixed sparsity/parameterization is a connected matrix group (or supply a correct retraction) and is closed under the required products/update.

### 4.1 Geometry-matched ordering

| Candidate structure | Current evidence | Recommendation |
|---|---|---|
| Dense 4x4 head quotient | **MEASURED exact rank 4**; categorical off-diagonals `-p_i p_j`; nonparallel class normals | Geometry is real, but retain the cheaper exact analytic per-pixel inverse |
| Per-class/diagonal | **DERIVED mismatch** to categorical cross-class coupling; useful as a cost control | First control, not presumed optimum |
| Module/block diagonal | Natural fit to current module boundaries; covariance not yet custodied | First trunk approximation after covariance capture |
| Kronecker input x output-gradient | Paper/code-supported; no Pact covariance-factorization receipt | Highest-priority structured treatment once measured |
| Generic low-rank + diagonal | No paper implementation and no retained spectrum custody | Defer |
| Hierarchical/tree SPD | No subgroup/closure proof or measured hierarchy custody | Defer |

**NO-VERDICT_DATA_CUSTODY:** the existing dual-metric artifacts retain scalar inner products/cosines, not full activation/output-gradient covariance spectra or Kronecker residuals. They cannot choose rank, blocks, or hierarchy.

Consumer: a read-only covariance producer feeding the #509 optimizer selector. Falsifiable selection gate: on held-out custodied steps, compare diagonal, block, and Kronecker approximations under equal factor-state bytes and equal update-time budgets; retain a structure only if it improves both covariance approximation/preconditioned-gradient fidelity and the sealed downstream n24 screen. Measure rather than infer per-class blocks.

`verdict_scope`: **STRUCTURE RANKING WITHOUT FULL COVARIANCE CUSTODY.** This is not a negative on any structure after its evidence exists.

## 5. Dual-metric sign flip and the Nielsen sibling

Let `u` and `v` be two parameter-space directions and let the categorical-Fisher pullback be

```text
H = J^T G(p) J = Q diag(lambda_i) Q^T,    lambda_i >= 0.
```

Writing `a_i = q_i^T u` and `b_i = q_i^T v`, the two unnormalized signs are

```text
Euclidean: sign(sum_i a_i b_i)
Fisher:    sign(sum_i lambda_i a_i b_i).
```

**DERIVED exact condition:** the cosine changes sign iff these two sums have opposite signs. This requires modal products `a_i b_i` of both signs and curvature weights `lambda_i` large enough to reorder the balance. Equivalently, Fisher cosine is Euclidean cosine after the normal-coordinate whitening `u_tilde = H^(1/2)u`, `v_tilde = H^(1/2)v` on the supported quotient. For an SPD tangent pair `U,V`, the paper's affine-invariant metric makes the analogous whitening explicit:

```text
g_P(U,V) = Tr(P^-1 U P^-1 V)
         = <P^-1/2 U P^-1/2, P^-1/2 V P^-1/2>_F.
```

Thus the paper predicts the **mechanism and algebraic condition**, not which Pact pair/epoch will flip. That requires custodied eigenspectrum/modal products.

**MEASURED local observations:**

- phase-advection versus Seg at ep725 stayed negative in both metrics: Euclidean `-0.1494`, Fisher `-0.1178`;
- margin-satisfice at ep725 flipped from Euclidean `+0.1875` to Fisher `-0.3208`;
- subpixel at ep800 flipped from Euclidean `+0.2156` to Fisher `-0.1301`;
- the identical-state ep725/726 repeat bounds observed Fisher-cosine noise at `|Delta cos| <= 0.036`, below both reported flip magnitudes.

The phase-advection row therefore does not validate a flip, while the two trajectory rows are consistent with curvature reweighting. They do not identify an SPD structure.

### 5.1 Convergence with the Nielsen Bregman sibling

The paper notes that generalized normal coordinates in expectation and natural parameters are Bregman-dual and agree to first order when their factor coordinates share the required group structure. That is the precise overlap with the Nielsen/Bregman lane: both describe a metric as Euclidean only after the correct local/dual coordinate map.

This memo does **not** duplicate the sibling by claiming that the observed cosine flip is an expectation-versus-natural-parameter duality result. The categorical pullback `J^T G J`, the affine-invariant SPD metric, and the sibling's finite Bregman/dual-Euclidean claims are different authority surfaces. Consumer: the dual-metric readback should ingest only the spectral sign condition and the coordinate/metric ID. Gate: persist `lambda_i` and signed modal products for a repeated state; predict the sign before reading the Fisher cosine; refuse interpretation when the support, Jacobian, or metric ID differs.

`verdict_scope`: **PREDICTION OF INDIVIDUAL PACT SIGN FLIPS WITHOUT SPECTRAL CUSTODY.** The coordinate-level mechanism is established; instance-level prediction is not.

## Proposed triality (spec only; no implementation in this lane)

### DSL leg

Propose one default-OFF typed lever, with no invented numeric defaults:

```text
SPDNormalCoordinatePreconditioner(
    structure = "kronecker",
    factor_dtype = "fp32",
    matmul_dtype = {"fp32", "bf16"},
    exp_truncation = {"quadratic_control", "linear_treatment"},
    factor_lr = Required[LawRef],
    factor_momentum = Required[LawRef],
    parameter_momentum = Required[LawRef],
    damping = Required[LawRef],
    update_period = Required[LawRef],
)
```

The compiler must refuse absent provenance, a non-fp32 factor state in the authority arm, linear truncation without a condition/refusal receipt, or missing complete resume-state declarations. This lane proposes the stub only and does not edit the trainer/DSL.

### DAG leg

```text
paper equation + source snapshot
    -> NumPy-fp32 reference + full optimizer-state schema
    -> deterministic repeat/split-resume proof
    -> MLX fp32 parity and no-fallback receipt
    -> covariance structure probe (diagonal/block/Kronecker)
    -> compiler-emitted sealed single-difference n24 negative filter
    -> resumable n600 treatment
    -> exact quantize/package/parse-back/R Seg+Pose+bytes facets
    -> adoption or formulation-scoped rejection
```

### Equation leg

Canonicalize two separate equation IDs rather than conflating them:

1. `spd_gnc_inverse_factor_momentum_v1`: the symmetric normal-coordinate `A exp(-m/2)` update and its quadratic control.
2. `kronecker_inverse_free_ngd_v1`: the `K,C,m_K,m_C,M_mu` layer update above.

Neither equation ID may alias `muon_ns5_weight_momentum_v1` or `categorical_head_pseudoinverse_v1`.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, and `docs/operating_manual_craft_handoff.md`.
- `reports/latest.md`; `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`; `.omx/state/master_gradient_anchors.jsonl`; `.omx/state/modal_call_id_ledger.jsonl`; `.omx/state/cost_band_posterior.jsonl`; `.omx/state/continual_learning_posterior.jsonl`.
- Latest applicable `codex_findings_*_codex.md`, `codex_session_summary_*_codex.md`, `council_t3_*.md`, design, directive, and operator-authorization surfaces.
- `.omx/research/segnet_recursive_fractal_factorization_20260715.md`.
- `.omx/research/fisher_actuation_arm_a_build_20260717.md` and the dual-metric trajectory/readback receipts.
- `.omx/research/p0_recovery_rate_probes_20260715.md` and the #496/#509 A/B ownership artifacts.
- `.omx/research/muonh_manifold_muon_dig_20260713.md`, `.omx/research/muon_round2_wire_fireable_20260713.md`, `.omx/research/optimizer_dynamics_followup_20260715.md`, and the #469 audit.
- `src/tac/information_geometry/fisher_natural_solver.py`, `src/tac/information_geometry/fisher_natural_solver_mlx.py`, `src/tac/optimization/md_decoupling.py`, `src/tac/witness_dsl/gauge.py`, and the current head-natural-gradient trainer implementation.
- [arXiv abstract/source for 2302.09738](https://arxiv.org/abs/2302.09738) and the [authors' repository](https://github.com/yorkerlin/StructuredNGD-DL).

## Final authority statement

No experiment was launched; no paid provider, live run, sacred result directory, trainer, archive, score, or frontier pointer was touched. The only repository mutations are this dated research memo and the L0 research-only lane registration. **MAIN must independently review the equations, ownership routing, state-file diff, and formulation-scoped verdicts before landing.**
