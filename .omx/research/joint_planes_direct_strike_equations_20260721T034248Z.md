# Equations — task-space level-set constructive inverse spine

**Authority:** `[macOS-CPU advisory]`; no score or global-optimality claim.
**Receipt:** `.omx/research/joint_planes_direct_strike_20260721T034248Z.json`

## S0: true frozen-scorer targets

For source video frames `V_t`, evaluator preprocessing `R`, frozen SegNet `F_seg`, and frozen
PoseNet `F_pose`,

```text
Y_seg,t  = argmax_c F_seg(R(V_t))_c
Y_pose,k = first6 F_pose(R(V_2k), R(V_2k+1)).
```

The sha-pinned n600 cache contains these target objects plus camera frames and margin fields. The
cache is source-derived but this pass reuses it; it does not claim a fresh full scorer rebuild.

## S1: support-fill exactness is not source-fraction exactness

For a rounded uint8 scorer plane `Y` and the disjoint resize operator `A`, canonical support fill
sets every owned camera tap in an output block to that block's `Y` value. Because the fp32
bilinear weights sum to one,

```text
c_can(Y)|support(i) = Y_i
A_fp32 c_can(Y) = Y
||A_fp32 c_can(Y) - Y||_infinity = 0.
```

Measured n600 residual: `0 / 117,964,800` values, max absolute error `0`. This is exact relative
to rounded `Y`. It does not recover the unrounded source resize fractions. The distinct M2 direct
source-target realization attains advisory `d_seg=d_pose=0` but costs `1,717,172,741 B`.

## S2: task-space level-set witness

Let `phi_c(x,t)` be one signed chart per SegNet class. The decoded partition is

```text
class(x,t) = argmax_c phi_c(x,t),
Sigma_cc'(t) = {x : phi_c(x,t) = phi_c'(x,t)}.
```

For Lane, the polynomial centerline and range-dependent half width define the zero-set of an SDF;
the #283 AA-SDF receiver integrates its subpixel footprint. The G1 temporal form is

```text
phi_c(x,t+1) = phi_c(H_xi(t)^-1 x,t) + epsilon_c,t(x),
```

where `H_xi` is induced by the Chasles screw and `epsilon` is coded at residual/topology events.
Measured median residual is `0.279212 px` within-pair and `0.279849 px` cross-pair. The registered
`worldsheet_transport_residual_event_rate_v1` is scoped to one global ground-plane homography;
cross-pair motion uses a proxy, so failure of that instance cannot close the worldsheet family.

## Fisher/margin precision waterfill

Registered equations consumed by ID:

- `frozen_scorer_fisher_curvature_margin_colocation_v1` — measured Fisher curvature versus
  negative margin colocation (`Pearson=0.978` in the registered band).
- `fisher_curvature_equals_categorical_fisher_trace_caustic_v1` — in the two-class annulus,
  `tr F(m) = 0.5 sech^2(m/2)`.
- `segnet_head_rank4_linear_flipdist_v1` — centered frozen head rank is four, and in penultimate
  feature space `d_flip(c,c') = |m_cc'| / ||w_c-w_c'||_2`.
- `witness_measured_reverse_waterfill_v1` — bits are removed/added in marginal score-value order.
- `worldsheet_transport_residual_event_rate_v1` — `E_r = N^-1 sum_i 1[d_i>r]`.
- `argmax_cell_identity_ideal_bytes_v1` — `B_cell = (1/8) sum_i -log2 p(c_i|context_i)`.
- `shearlet_nterm_upper_bounds_task_rate_v1` — curvelet/shearlet charts are the boundary-residual
  basis; this landing does not consume a Fourier candidate basis.

With `b_j` description bytes assigned to cell/chart `j`, the contest byte price is
`lambda_B=25/37,545,489`. A byte is admissible only while its measured marginal non-rate gain
exceeds that price:

```text
-d[100 D_seg + sqrt(10 D_pose)]/db_j > lambda_B.
```

This is a specification for the missing complete S2 allocator, not evidence that the allocator
ran in the current Lane-only row.

## Range(A), blind coordinates, and integer realization

For the linear resize matrix `A`,

```text
P_range(A^T) = A^T (A A^T)^-1 A,
x = P x + (I-P)x,
A(I-P)x = 0.
```

The full linear nullity fraction is `0.8067423152`. The implemented integer-exact blind mask is
the smaller `0.2269692609` fraction (`230,904` pixels/frame). Its generic receiver fill is free;
video-derived retained values remain counted.

S3 is the bounded integer problem

```text
min_x in {0,...,255}^n L_task(R(x),Y) + lambda_B L_MDL(x)
subject to the selected support/cell constraints.
```

No sub-step write may be admitted. Candidate moves have fixed integer magnitude and are accepted
only after a fresh hard uint8/R oracle shows the intended decision improvement without forbidden
collateral debt. r1b7 supplies only that lesson; its inherited candidate bytes are excluded.

## Score equation and current authority

```text
S = 100 d_seg + sqrt(10 d_pose) + 25 archive_bytes / 37,545,489.
```

No S4 archive exists for this from-scratch composition, so `archive_bytes`, per-class `d_seg`,
`d_pose`, and `S` are undefined. The pointer remains unchanged.
