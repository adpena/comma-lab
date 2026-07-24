# DDM MS1 minimum-description lattice equations

Date: 2026-07-24
Lane: `lane_ddm_ms1_min_description_lattice_solve_20260723`
Authority: `[macOS-CPU frozen-scorer advisory]`
`research_only=true`, `score_claim=false`, pointer
`0.1910828242 [contest-CPU]` unchanged; MAIN landing review required.

## 1. Campaign objective

Let `p` be an own-lineage stored inverse problem and `E(p)` its deterministic
receiver expansion. Let `e` be the solve-mandated exception stream and
`D(E(p),e)` the reconstructed uint8 witness. The counted length is

```text
L_total = bytes(p) + bytes(e | E(p)).
```

The campaign decision row is

```text
(L_total, d_seg(D(E(p),e)), d_pose(D(E(p),e))).
```

It exists only when `p` and `e` have exact bytes/SHA custody, `p` is own
lineage, the expansion is receiver-closed, the Pose6 tube is active in the
solve, and the final witness is accepted through uint8, the pinned resize, and
both frozen scorers. Donor-archive or PR-lineage conditioning makes the row
inadmissible rather than merely non-promotable.

`m7` supplies only the existence observation that enough conditioning permits
a zero-training joint solve. Its donor spine is not an MS1 expansion or prior.

## 2. Gauge-fixed lattice

Let `A` be the integer-numerator form of the pinned resize, `K=ker(A)`, and
`Q=I-P_K` the visible/range projector. For a feasible camera witness `x` and
deterministic expansion `b=E(p)`, the continuous gauge condition is

```text
P_K(x-b) = 0,
x-b = Q(x-b).
```

Uint8 projection generally prevents exact satisfaction. The integer problem is
therefore a closest-vector problem:

```text
minimize   CoderBytes(x-b)
subject to x in Z^n intersect [0,255]^n
           x in C_seg
           x in T_pose
           x minimizes ||P_K(x-b)|| under the preceding constraints.
```

The last line is a gauge tie-break, not free payload. After integer projection,
the implementation must measure residual gauge leakage and recheck the real
scorer constraints. Catalog #532 forbids inferring realized acceptance from a
continuous range projection.

## 3. Exact local kernel used by the v1 diagnostic

For each disjoint 2x2 resize block, let `a in Z^4` be the integer coefficient
row. A unimodular completion `U` satisfies

```text
a U = (gcd(a),0,0,0).
```

The last three columns form the saturated integer kernel

```text
K_Z = ker_Z(a),  a K_Z = 0,  rank(K_Z)=3.
```

Exact size reduction uses only unimodular row subtraction and permutation. It
changes search conditioning, not the represented lattice. This is the adopted
Catalog #586 row; no special-q/LAS analogy or global shortest-vector claim is
made.

The measured v1 proposal fixes the visible target `A x0` and approximately
solves

```text
x_hat = argmin_{x in (x0 + K_Z) intersect [0,255]^n} ||x-b||_2,
```

then admits `x_hat` only if the real modular-residual zlib code is strictly
smaller. It is a saturated local-CVP proposal, not the full minimum-description
cell/tube optimum.

## 4. Conditional coder diagnostic

For a uint8 conditioning expansion `b`, define the bijective residual

```text
r_b(x) = (x-b) mod 256,
x      = (b+r_b(x)) mod 256.
```

The v1 diagnostic measures

```text
L_b^diag(x) = bytes_zlib9(r_b(x)).
```

This is a real coder and exact parse-back surface, but it is not a
receiver-closed archive or the campaign total. In particular,

```text
L_b^diag != bytes(p) + bytes(e | E(p))
```

until `b=E(p)` is SHA-bound to an own-lineage stored problem and the exception
stream is emitted by the receiver grammar. Therefore v1 may quantify
conditional-code strength or a proposal loss, but its total cannot headline.

## 5. Joint scorer constraints

For current Seg winner `c_i` and every rival `j`,

```text
m_ij(x) = logit_i,c_i(x) - logit_i,j(x),
C_seg   = {x : min_j m_ij(x) >= epsilon_seg,i for every i}.
```

The active Fisher coordinate is the top1-top2 margin. Any positive tolerance
step must use the corrected inner-Jacobian/secant law and is still proposal
only until realized acceptance.

For Pose6 target `q` and tube radius `epsilon_pose`,

```text
T_pose = {x : mean((Pose6(x)-q)^2) <= epsilon_pose}.
```

The successor solve must include this tube in the optimization; a post-hoc
Seg-only member does not satisfy the campaign contract. The v1 diagnostic uses
the stronger equality oracle against the unchanged member, but it has no
tolerance-knee authority.

## 6. Active set, degeneracy, and honest duals

For local selected block `z`, let `I(z)` be the active uint8 box facets. With
saturated basis `B in Z^{4x3}`, the continuous local face dimension is

```text
d_facet(z) = 3 - rank(B[I(z),:]).
```

This is not an integer-neighbour reach count. The bounded projection exposes
no KKT multipliers, so v1 shadow prices are typed unavailable; Fisher margins
are never relabeled as duals.

The geometry-only facet lookup has `24,576 x 16` entries. It is now cached once
per solver instance. Recomputing it for each of 600 pairs was measured at
`2.608 s/pair` on the local host and is retired as apparatus debt.

## 7. Factorize and representation arbitration

Let `M` be the pair-by-feature SENSE matrix. Standardize nonconstant columns
and compute

```text
M_bar = U Sigma V^T.
```

A numerical factor clears the real-coder floor only if its rate-coordinate
amplitude is at least one byte:

```text
abs(sigma_k V[k,rate] std(rate)) >= 1 byte.
```

It is not yet a distilled vocabulary column. For each stratum `s`, measure

```text
B_skeleton(k,s) = bytes of token/topology code,
B_fiber(k,s)    = bytes of coefficient/transform code.
```

Then:

```text
B_skeleton < B_fiber -> tag SKELETON
B_fiber < B_skeleton -> tag FIBER
tie or missing race  -> do not distill.
```

SKELETON routes to pf1/g1 token-like coders. FIBER routes to
transform/quantize/entropy plus stratum amplitude-law coders. No default role
is permitted.

## 8. Headline firewall

The executable row is
`ddm_min_description_headline.v1`. Its necessary conditions are

```text
own_lineage(p)
and not donor_conditioned
and receiver_closed(E(p))
and pose_tube_active
and realized_uint8_R_frozen_scorers
and sha_bound(p,e).
```

Only then:

```text
headline.total_counted_bytes = bytes(p)+bytes(e).
```

Otherwise `decision_triple` is null and exact blockers are emitted while
diagnostic distortions remain available.

## 9. Triality and directive consumption

- **Equations:** this note owns campaign total, gauge, local lattice, joint
  constraints, factor arbitration, and headline admission.
- **DAG:** the adjacent FEED owns the source-to-expansion-to-solve-to-headline
  graph and the measured-v1 diagnostic branch.
- **DSL:** N/A with rationale. MS1 adds no trainer, curriculum, submission, or
  launch lever; inventing a flag would add false executable state.

| Directive | Consumption |
|---|---|
| EV-ranked reverse waterfill | Real coder admission is strict; a losing local-CVP proposal is pruned without a blanket pixel fix. |
| Fisher/margin, corrected inner Jacobian, xi factorization | Active coordinates are Fisher margins; no naive flip claim or Fourier residual basis; xi remains a conditional coder candidate. |
| Stored problem plus solve exceptions headline | Executable firewall requires both exact byte homes and realized joint constraints. |
| No donor spine and terminology correction | Donor conditioning is inadmissible; new authority artifacts use stored problem, deterministic expansion, and conditional exceptions. |
| SKELETON x FIBER stratification | Distillation requires a measured per-stratum coder race and has no default role. |
| n600, resumability, findings as first rungs | All 600 v1 stages are atomic; the v1 result stays formulation-scoped and the missing headline edge is explicit. |
