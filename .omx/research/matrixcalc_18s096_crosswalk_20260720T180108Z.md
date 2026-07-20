# MIT 18.S096 matrix-calculus crosswalk into Pact

**UTC:** 2026-07-20T18:01:08Z  
**Lane:** `lane_matrixcalc_18s096_crosswalk_20260720` (`L0`, `research_only=true`)  
**Authority:** read/analyze/write only; `$0`; no training, scorer actuation, archive mutation, or dispatch  
**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**  
**Landing:** isolated worktree; independent **MAIN landing review is required**

## Verdict first

This is a **confirmatory crosswalk with three narrow adoptions, not a new solver family**. Pact already has the
important mechanisms: the exact rank-4 SegNet head law, exact resize adjoint and uint8 feasibility layer,
first-order-plus-secant-plus-cluster-QP correction, real VJP custody, a measured sparse/low-rank VJP negative,
an exact-factorized costate organ, and a much deeper SPD/Muon treatment than this course.

The course sharpens three boundaries:

1. **ADOPT for Task #580/#391:** name the existing camera projector by its actual space:
   `P_visible = P_range(A*) = A+ A = P_ker(A)^perp`, not `P_range(A)`. Because both 1-D resize factors have
   full row rank, `range(A)` is the whole *real scorer plane*; the restrictions that remain are uint8/rational
   realization and native-fp32 decision cells, not real-linear range. The current QR implementation is correct
   and is computationally better than materializing a Kronecker matrix.
2. **ADOPT for `r1b2_mdl_xi0_compile`:** keep the Fréchet/Jacobian prediction and the realized uint8 secant as
   different typed objects. A local derivative is a tangent model with a normed little-o remainder; a finite
   through-rounding secant is endpoint evidence at one step. The QP may consume both, but must record the base
   point, step, norm, remainder, crossed quantization cells, and hard-oracle endpoint. Never relabel the secant as
   a derivative or use the derivative as uint8-realization evidence.
3. **ADOPT for pending #556:** for `W=QH`, require the explicit product differential and pullbacks, plus a metric
   declaration. Independent `Q`/`H` momentum is a chosen product geometry; it is not automatically the
   Frobenius geometry induced from `W`. The course supplies the orthogonal tangent constraint, but not the SPD
   affine-invariant update, polar differential, or Muon mathematics; Task #552 already has the stronger material.

No course result replaces `segnet_head_rank4_linear_flipdist_v1`. The course does **not** derive the SVD; it only
mentions singular values as an example and explicitly leaves further matrix-factorization derivatives to other
literature. The current canonical Task #461 surface is a lossless cross-tensor storage chart, not a differentiable
spectral-rate operator, so eigen/SVD derivatives are **N/A** to that artifact.

## Source custody and selected reading

I read the real notes, not only the syllabus or table of contents:

- [MIT OCW 18.S096 lecture-notes index](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/pages/lecture-notes/)
- Bright, Edelman, and Johnson, [*Matrix Calculus (for Machine Learning and Beyond)*, arXiv:2501.14787v1](https://arxiv.org/abs/2501.14787)
- downloaded primary PDF: `1,884,010 B`, 101 pages, SHA-256
  `ee3d28554bf92495d84e5f672e6c3820733e37daee95ba212bfaca6332ee77db`

| Course unit | Exact content used | Pact question |
|---|---|---|
| Ch. 2 §2.1–2.2, pp. 9–10 | derivative as a linear operator plus `o(||delta||)` | local winner/rival Jacobian scope |
| OCW Lecture 2 / Ch. 3 §3.3.1–3.3.3, pp. 25–28 ([PDF](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec03.pdf)) | `vec(B C A^T)=(A tensor B)vec(C)`; do not materialize dense Kronecker matrices | separable resize, projector, pseudoinverse |
| OCW Lecture 3 / Ch. 4 §4.1–4.7, pp. 29–32 ([PDF](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec04.pdf)) | finite differences are checks with truncation/roundoff error | measured secant versus derivative |
| Ch. 5 §5.2, pp. 37–38 | normed/Banach-space Fréchet remainder | trust-region and remainder custody |
| OCW Lecture 4 / Ch. 6 §6.3–6.3.2, pp. 42–44 ([PDF](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec06.pdf)) | one transposed adjoint solve; implicit differentiation avoids differentiating converged iterations | #486/#516, VJP extension |
| OCW Lecture 5 / Ch. 8 §8.4, pp. 61–62 ([PDF](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec08.pdf)) | forward cost scales with inputs; reverse with outputs; reverse tape overhead | batch-16 and 24-to-600 economics |
| OCW Lecture 8 / Ch. 13 §13.1–13.2.1, pp. 96–99 ([PDF](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec13.pdf)) | sphere/orthogonal tangents; simple symmetric eigenderivatives and gap denominators | Q/H chart and spectral guards |
| Ch. 14, pp. 100–101 ([PDF](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec14.pdf)) | repeated eigenvalues lose ordinary differentiability; further factorizations not covered | no fake SVD transfer |

## Crosswalk table

| Pact surface | Course transfer | Verdict | Exact consumer / consequence |
|---|---|---|---|
| `r1b2_mdl_xi0_compile`, batch-16 first-order rank-4 winner/rival Jacobians plus realized secants in one QP | Fréchet remainder + finite-difference scale/roundoff distinction | **ADOPT-TYPING; ALREADY-HAVE-MECHANISM** | `inner_jacobian_secant_qp` manifest must separate local `J` from realized endpoint secants and hard-oracle acceptance |
| Separable resize `A(X)=D_r X D_c^T`; #580 projector; #391 adjoint | vec/Kronecker identity, transpose, structured operator cost | **ADOPT-NOTATION/PSEUDOINVERSE; ALREADY-HAVE-BETTER-CODE** | clarify `range(A*)` camera projector; expose closed-form minimum-norm preimage only as a real-linear reference, never a uint8 authority |
| #486/#516 and sibling `probe_exec_vjp_extend` 24-to-600 | forward/reverse scaling; implicit adjoint solve | **ALREADY-HAVE-BETTER; ADOPT-REUSE-GUARD** | batch scalar VJP is appropriate; frozen weights do not make pair-dependent activation Jacobians reusable |
| #552 SPD momentum, #469 Muon, pending #556 `W=QH` | `Q^T dQ` skew and tangent projection | **ADOPT-CHART-DIFFERENTIAL; ALREADY-HAVE-BETTER-SPD/MUON** | #556 must declare pullbacks, metric, retraction, full-rank domain, and split-resume state; no automatic same-momentum composition |
| `segnet_head_rank4_linear_flipdist_v1`; Task #461 | simple symmetric eigenderivatives; repeated-root caveat | **N/A for head flip and canonical #461; CONDITIONAL spectral-gap guard** | head hyperplane flip is exact and does not differentiate its SVD; canonical #461 is bijective storage coding, not a spectral gradient arm |

## 1. Fréchet derivatives versus realized uint8 secants

For a smooth pre-round margin map `m`, the course's definition is

```text
m(x + delta) - m(x) = J_m(x) delta + r_x(delta),
||r_x(delta)|| / ||delta|| -> 0 as ||delta|| -> 0.
```

That statement is norm-, base-point-, and scale-dependent. It sharpens the existing r1b2 design as a custody
contract, not a new predictor. For every candidate family or batch-16 QP block, retain:

```text
base_id, base_hash, delta_hash, ||delta||,
J_delta, realized_delta_margin,
remainder = realized_delta_margin - J_delta,
remainder_ratio, quantization_cells_crossed,
native_f32_hard_oracle_verdict.
```

The distinction becomes load-bearing through rounding. For a scalar quantizer
`R_q(x)=q floor(x/q+1/2)`, `D R_q=0` in the interior of every cell and is undefined at cell walls. Therefore the
Fréchet derivative of the composite through-rounding map is zero almost everywhere and cannot predict a cell
crossing. A finite realized secant

```text
s_R(x;delta) = [m(R(x+delta)) - m(R(x))]
```

is the correct endpoint measurement, but it is not a local derivative and does not transfer to another step or
base cell without validation. The course's §4.5–4.6 truncation-versus-roundoff discussion supports the separation;
Pact's integer lattice and hard oracle are stricter than the course.

**QP consequence.** Keep `J delta >= margin_debt + epsilon` as the local continuous model inside an explicit
trust region. Use realized secants to calibrate or reject that model at their exact stored displacements; do not
treat a finite secant row as a globally valid Jacobian row. Admission still requires the integer/uint8 solve and
native-fp32 hard oracle. This exactly preserves the registered `flip_margin_step_law_v1` hierarchy:

```text
exact rank-4 head in feature space
-> local input Jacobian
-> finite secant/remainder calibration
-> overlapping-footprint cluster QP
-> exact integer realization
-> native hard oracle.
```

**Verdict scope:** this does not claim the pending r1b2 artifact is wrong. It prevents a specific category error in
its evidence schema. The course confirms the already-routed corrected first-order+secant+QP mechanism; it does
not supply a new measured lift above the historical `0.594` naive verification.

## 2. Separable resize: the exact Kronecker, kernel, pseudoinverse, and projector map

Let `D_r in R^(384 x 874)` and `D_c in R^(512 x 1164)` be the exact impulse-probed bilinear factors, and use
column-major vectorization. From §3.3.1,

```text
A(X) = D_r X D_c^T,
vec(A(X)) = K vec(X),
K = D_c tensor D_r.
```

The adjoint is

```text
A*(Y) = D_r^T Y D_c.
```

The current implementation correctly avoids constructing `K`; this is exactly the structured-computation warning
in §3.3.3. With full-row-rank factors, the Kronecker pseudoinverse identity gives

```text
K+ = D_c+ tensor D_r+,
A+(Y) = D_r+ Y (D_c+)^T.
```

This is the minimum-Frobenius-norm **real** camera preimage. It is not the minimum-rate preimage and is not
necessarily uint8-feasible; Pact already learned that minimum-norm filling is rate-dead. The two projectors are

```text
camera visible-space:  P_in = A+ A = P_range(A*) = P_ker(A)^perp,
scorer output-space:   P_out = A A+ = P_range(A) = I_(384 x 512),
```

where the last equality follows because both factors are full row rank. Consequently, the current
`range_a_projection.apply_projection` formula

```text
P_in(X) = Q_r (Q_r^T X Q_c) Q_c^T
```

is correct, but its `P_range(A)` label means the wrong side of the operator in standard linear-algebra language.
The honest label is `P_range(A.T)` / `P_visible` / `P_ker(A)^perp`.

The full nullspace is larger than the set of individually blind pixel coordinates:

```text
ker(D_c tensor D_r)
  = (R^1164 tensor ker D_r) + (ker D_c tensor R^874),
intersection = ker D_c tensor ker D_r.
```

Thus zero columns of the resize matrix are a useful sparse subset, while the QR projector captures the complete
dense nullspace. This matches the existing distinction between the certified blind-coordinate lever and the larger
measured scorer-invisible energy.

**Task route:** #580 should clarify the type/name and may expose `A+` as a reference/certificate; #391 keeps the
exact adjoint and dead-zone law. Do not replace the QR/separable implementation with an explicit Kronecker
matrix, and do not infer uint8 realization from real-linear surjectivity.

## 3. VJP economics and what a frozen operator does not amortize

For `f:R^n->R^m`, the course's §8.4 result is structural:

- forward/JVP work scales with input directions;
- reverse/VJP work scales with output cotangents;
- reverse mode carries a tape/storage cost.

For batch-16 independent pairs scalarized into one loss, one reverse sweep returns a gradient block for every
batch input, so the current VJP direction is correct. Going from 24 to 600 pairs should be treated as approximately
linear in the number of evaluated base states, modulated by measured batch geometry and memory. It is not a
reason to form a full Jacobian.

What can be amortized:

- exact linear maps and their transposes: resize `A/A*`, frozen head pair normals, incidence matrices;
- compiled graph structure, frozen weights, allocation buffers, and genuinely identical factorizations;
- a tape only while its exact base activations remain valid.

What cannot be assumed reusable:

- `J_F(x)` across different frames/states merely because the scorer weights are frozen;
- a sparse output mask as an exact input VJP (Task #486 measured global SE/receptive-field spread);
- a low-rank cohort basis (Task #486 measured `r95=68/120`, `r99=100/120` and retained the scoped negative);
- a factorization of a pair-dependent implicit Jacobian without same-matrix custody.

For a genuinely implicit converged state `h(p,x)=0` and scalar `g(p)=f(x(p))`, §6.3.1 gives

```text
h_x^T lambda = grad_x f,
grad_p g = -h_p^T lambda.
```

For a fixed point `x=F(x,p)`, this becomes `(I-F_x)^T lambda=grad_x f`. This can avoid differentiating every
solver iteration, but only after convergence error is bounded. It does not apply to an ordinary explicit scorer
forward simply because that scorer is frozen.

**Verdict:** #486/#516 already contain stronger Pact-specific structure and measurements. For sibling
`probe_exec_vjp_extend`, adopt only the fail-closed reuse declaration: each reused tape/factorization must name
the base-state hash and operator hash; otherwise recompute. Preserve per-pair resumability when extending 24 to
600. No course result reopens the masked/low-rank VJP negatives.

## 4. Orthogonal/SPD product chart versus Muon

The course proves the orthogonal tangent constraint

```text
Q^T Q = I  =>  Q^T dQ + (Q^T dQ)^T = 0.
```

For a square full-rank polar chart `W=QH`, `Q` orthogonal and `H` SPD, the exact product differential is

```text
dW = dQ H + Q dH,
Omega = Q^T dQ is skew,  dH is symmetric.
```

If `G=grad_W L` under the Frobenius pairing, the ambient pullbacks are

```text
C_Q = G H,
C_H = sym(Q^T G),
grad_Q^tangent = Q skew(Q^T C_Q)        [square orthogonal case].
```

For the affine-invariant SPD metric used by Task #552, the Riemannian gradient is `H C_H H`; that is from the
SPD source audited in #552, **not** from 18.S096. Likewise, Muon's Newton–Schulz matrix-sign/polar-momentum
operation is not derived by this course.

The metric choice cannot be left implicit. The Frobenius metric induced from `W` is

```text
||dW||_F^2 = ||Omega H + dH||_F^2,
```

which generally has a nonzero cross term `2 <Omega H, dH>`. Declaring a direct-product metric with independent
Q/H momenta is legal, but it is a new chosen geometry rather than an automatic pullback of the weight-space
Frobenius metric. Pending #556 should therefore persist a `metric_id`, both momenta, both retractions, factor
condition, and the full-rank/polar uniqueness gate. At rank loss or clustered zero singular values, the chart must
refuse or change strata.

**Verdict:** the course supports the tangent algebra and the need for constrained directions. It does not justify
replacing the measured #469 Muon schedule or the #552 SPD-normal-coordinate spec. The only adoption is the
explicit chart/pullback/metric contract before a Q+H arm is compiled.

## 5. Eigen/SVD derivatives versus the rank-4 head and Task #461

For a symmetric eigendecomposition `S=Q Lambda Q^T` with simple eigenvalues, §13.2.1 gives

```text
d lambda_i = q_i^T (dS) q_i,
(Q^T dQ)_ij = (q_i^T dS q_j) / (lambda_j-lambda_i),  i != j.
```

Eigenvector sensitivity therefore grows like the inverse spectral gap. The course explicitly warns in Ch. 14
that ordinary eigenvalue/eigenvector differentiability fails at repeated eigenvalues; generalized/degenerate
treatments or a reformulation are then required.

This does not sharpen `segnet_head_rank4_linear_flipdist_v1`: the SVD there is an offline certificate that the
centered frozen head has rank four. The live flip law is the exact affine hyperplane distance

```text
|margin_(c,c')| / ||w_c-w_c'||,
```

so no singular-vector derivative appears. Replacing it by spectral perturbation calculus would be weaker.

The course contains no SVD derivative derivation. Moreover, the locally registered #461 equation is
`witness_lossless_cross_tensor_storage_law_v1`: bijective axis permutations and modulo-256 temporal deltas on a
fixed quantized state. Its score invariance follows from exact decoded-state equality, not differentiable spectral
geometry. Hence the course transfer to canonical #461 is **N/A**. If a different live arm uses “#461” for a
spectral-gradient lever, it must supply an exact canonical object identity before this memo can route it. Any such
future differentiable spectral arm should refuse individual-vector gradients at clustered singular/eigenvalues and
prefer invariant subspace/projector objectives.

## Independent identity checks

These are **MEASURED local NumPy algebra checks**, not scorer measurements or score evidence. Five short
scripts used deterministic seeds and no repository data.

| Check | Output |
|---|---:|
| `vec(D_r X D_c^T)=(D_c tensor D_r)vec(X)` max error | `1.6653345369377348e-16` |
| `(D_c tensor D_r)+ = D_c+ tensor D_r+` max error | `1.1102230246251565e-15` |
| matrix-form minimum-norm preimage max error | `1.1102230246251565e-16` |
| `A(P_visible X)-A(X)` max error | `2.220446049250313e-16` |
| tiny rounding probe at `x=.49`, `delta=.02` | local derivative probe `0.0`; realized secant `50.0`; quantized delta `1.0` |
| orthogonal tangent residual | `2.220446049250313e-16` |
| `QH` product finite-difference error at `epsilon=1e-7` | `2.6841260769572273e-7` |
| eigenvector slope, gap `.5` / `.01` | `1.40000014` / `70.00035`; theory `1.4` / `70.0` |
| implicit-adjoint gradient versus finite difference | absolute error `6.744809988301625e-10` |

The checks validate the algebra and expose scale/gap behavior. They do not establish a new empirical anchor and
should not be registered as a contest or training measurement.

## Ranked transfers

1. **`r1b2_mdl_xi0_compile`: derivative/secant/oracle type contract.** Highest immediate value because the live
   QP explicitly mixes these evidence classes; add the typed remainder/step/cell fields before treating its output
   as fireable.
2. **Task #580: correct the projector side and expose the separable pseudoinverse certificate.** Preserve the QR
   implementation; clarify that real scorer-plane range is full and uint8 feasibility is a separate layer.
3. **Pending #556: exact `QH` pullbacks plus explicit metric ID.** Prevent a geometrically unidentified same-
   momentum Muon/SPD composition.
4. **`probe_exec_vjp_extend`: base-state-bound reuse declaration.** Reuse fixed linear factors, not nonlinear
   activation Jacobians; preserve batch/resume custody through 24-to-600.
5. **Conditional spectral-gap guard only.** Add it if and when a real differentiable spectral objective is named;
   it is N/A to the rank-4 flip law and canonical #461 today.

## Caveats, integration boundary, and triality

- This course is pedagogical and broad. It does not provide specialized pseudoinverse, polar/SPD, SVD, Muon,
  uint8-lattice, or evaluator-cell machinery. Those conclusions above are either **DERIVED** from its identities or
  compared against stronger local artifacts; they are not falsely attributed to the authors.
- No course formula supplies a measured `Delta S`, byte saving, VJP speedup, or improved r1b2 flip rate.
- Focused regression validation was `14 passed, 1 skipped, 1 failed`. The failure is a pre-existing committed
  test/builder mismatch: `test_resize_exploit_flip_fix_frontier.py` expects two empirical anchors while the
  committed builder defines three. This lane changed neither file; the mismatch does not falsify the resize
  identities checked above and requires separate owner review.
- `lane_maturity validate` remains blocked by 110 pre-existing evidence paths absent from this isolated worktree;
  the new research lane itself is a valid L0 row.
- **DSL:** N/A; research-only crosswalk. The three adoptions are typed spec/manifest requirements, not trainer
  flags.
- **DAG:** route to `r1b2_mdl_xi0_compile`, #580/#391, sibling `probe_exec_vjp_extend`, and pending #556 after
  MAIN review. No launch edge is authorized.
- **Equations:** no new canonical equation is registered. The resize identities clarify existing operators;
  r1b2 consumes `flip_margin_step_law_v1`; head work consumes
  `segnet_head_rank4_linear_flipdist_v1`; VJP/costate work retains its registered laws.
- **Sensitivity/Pareto/bit allocator:** only receiver-closed realized secants may price #536/KKT. The local
  derivative remains a proposer. This memo creates no new marginal or posterior row.
- **Autopilot/continual learning:** N/A until a typed implementation or empirical anchor exists. Missing
  integration is explicitly the r1b2 manifest guard, #580 naming/reference API, and #556 metric declaration.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`; v7.5 §8 and v8 specs;
Claude top-memory entries; latest Codex findings/session, T3 council, and design memo; `reports/latest.md`;
lane/task/subagent/equation state; `tools/graph_memory_recall.py`; the registered rank-4, resize, VJP/costate,
manifold, waterfill, and cross-tensor laws; #391, #486, #516, #536, #549, #552, and #461 memos/code; the
per-arm inbox through `2026-07-20T17:52:10Z`; broadcast inbox through `2026-07-19T19:48:01Z`; and the primary
MIT/arXiv corpus cited above.

## Pointer and review

Pointer delta is exactly zero. There is no score, promotion, launch, or provider claim. **MAIN must independently
review the projector-side correction, the r1b2 typing recommendation, the #556 metric caveat, the isolated lane
state, and the serializer commit before landing.**
