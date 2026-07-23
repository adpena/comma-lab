---
schema: ddm_m1_missed_optimum_equations.v1
date_utc: 2026-07-23
lane_id: lane_ddm_m1_missed_optimum_hunt_20260723
research_only: true
execution_allowed: false
score_claim: false
verdict_scope: "DERIVATION for two probe formulations; no empirical, family, score, or promotion verdict"
pointer_moved: false
main_landing_review_required: true
---

# DDM M1 missed-optimum equations

These equations are the derivation leg for a design-only audit. They are not registered canonical
laws and do not claim that either representation is feasible.

## 1. Exact objective and byte dual

For exact archive bytes `B`,

`S = 100 D_seg + sqrt(10 D_pose) + lambda_B B`,

where

`lambda_B = 25 / 37,545,489 = 6.658589531221714e-7 score units/byte`.

The n600 Seg target is the integer condition

`E_seg <= floor(0.00116 * 117,964,800) = 136,839`.

Every finite proposal is admitted only from exact differences in the same decoded master:

`Delta S = 100 Delta D_seg + sqrt(10 D_pose_after) - sqrt(10 D_pose_before) + lambda_B Delta B < 0`.

## 2. Exact frame incidence

From `SegNet.preprocess_input`,

`D_seg(I_0,I_1) = D_seg(I_1)`,

so for a frame-0-only decision variable `theta_0`,

`partial D_seg / partial theta_0 = 0`

structurally, provided frame 1 is byte-identical. Pose is different:

`D_pose(I_0,I_1) = ||pi_6 P(Y(R I_0), Y(R I_1)) - p*||_2^2`,

where `Y` is the official RGB-to-YUV6 map, `R` is evaluator resize/uint8 realization, and `P` is
the frozen PoseNet. No luma-only or chroma-null simplification is imported.

The conditional frame-0 description problem is

`min_theta0 L(theta_0) + lambda_B^{-1} sqrt(10 D_pose(G(I_1,xi,theta_0), I_1))`

subject to exact frame-1 identity and legal scorer-free decode. Equivalently, the constrained rate
point is

`R_0(tau | I_1,xi) = min L(theta_0) : D_pose <= tau, I_1' = I_1`.

The probe asks whether `R_0(0.00161 | I_1,xi) <= 7,195 B` on the current DDM vehicle.

## 3. Kinetic Laguerre cell complex

An anisotropic power cell is

`C_j(t) = {x : P_j(x,t) <= P_k(x,t) for all k}`,

`P_j(x,t) = (x-q_j(t))^T M_r(t) (x-q_j(t)) - w_j(t)`,

with one positive-definite metric `M_r(t)` shared by all sites in chart `r` and a class label
`c_j`. Sharing the chart metric preserves affine pairwise bisectors after a chart transform and
therefore a regular-triangulation dual. A distinct per-site metric would instead induce quadratic
bisectors and is outside this registered formulation. The partition label is

`l_theta(x,t) = c_argmin_j P_j(x,t)`.

The counted description is not a framewise label table. It is

`theta = {initial sites, class labels, spline coefficients of q/w/M_r, initial regular triangulation, sparse flip/birth/death events}`.

The decisive target-space rate is

`R_K(delta) = min_theta L_real_coder(theta)`

subject to

`sum_{t,x} 1[l_theta(x,t) != l*(x,t)] <= delta * 117,964,800`.

The registered Stage-A gate is `delta<=0.001159998576` and `L<=100,099 B`. A Stage-A point has no
score authority until a scorer-free RGB pullback `G_rgb(theta)` satisfies the same target after
the exact `uint8 -> R -> SegNet` chain and also meets the Pose tube.

## 4. Why affine head bytes are insufficient

The frozen Seg head gives affine scores on a spatial quotient field:

`s_c(x,t) = <a_c, z(x,t)> + b_c`.

The packet `{a_c,b_c}` defines decision hyperplanes but not `z(x,t)`. For two finite fields
`z_1 != z_2`, the same packet can induce different spatial partitions. Therefore

`H(partition | {a_c,b_c}) > 0`

for the current source ensemble, and packet bytes alone cannot be called a spatial generator.
The kinetic proposal explicitly counts its spatial field representation.

## 5. Interaction-aware description

For exact score gain `g(A)=S(empty)-S(A)` of a set of decoded motifs, the pair interaction is the
Möbius coefficient

`mu(i,j) = g({i,j}) - g({i}) - g({j})`.

v19b measures positive `mu` for most admitted ordering steps. Thus a valid rate allocator cannot
assume

`g(A) = sum_i g({i})`.

Both proposed probes use exact common-master replay. Any future compound dictionary may encode a
hyperedge only when its complete joint byte cost and exact joint gain beat the best constituent
encoding.

## Scope

Failure of the finite site/degree/metric ladder closes only its named formulation. Failure of the
finite frame-0 basis/quantizer ladder closes only that conditional preimage formulation. Neither is
a generator-family, Pose-preimage-family, or direct-description-paradigm verdict.
