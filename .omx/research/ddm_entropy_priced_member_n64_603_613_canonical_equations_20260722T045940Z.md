---
title: Canonical equations - entropy-priced member solve
date_utc: 2026-07-22T04:59:40Z
task: 603
feeds_task: 613
lane_id: lane_ddm_mdl_member_solve_v3_entropy_603_613_20260722
research_only: true
---

# Exact entropy rate

Let `z=(z_0,...,z_5)` be the six receiver-semantic streams. For stream `s`, let `T_s` be the finite
set of exact invertible transforms and `C_s(t)` the eligible exact coder tournament. The compiler
selects

`(t_s*,c_s*) = argmin_(t,c) (len(frame_s(c(t(z_s)))), transform_id, coder_id)`

subject to exact `c^-1` and `t^-1` reconstruction. The deterministic stored ZIP archive is
`A_H(z)`, and the only admitted rate is

`B_H(z) = len(A_H(z)) = 22 + sum_s H_s(z_s)`,

where `H_s` is the unique final ZIP-home byte count of stream `s`. For the lossless n64 baseline:

`B_H(z)=22+338+531+12134+16162+15556+626=45369`.

The identical fixed-width semantic archive had `B_F(z)=274664`, hence

`Delta B_entropy = B_H(z)-B_F(z) = -229295` bytes.

# Exact constrained subset solve

Let `q_m(z)` collapse the residual streams selected by mask `m in {0,...,7}` to their maximal
integer-safe zero state. For stratum family `f` and key `k`, let

`E_f,k(q_m)=1-M_f,k(q_m)`

be exact frozen-SegNet same-cell escape, and let `C_pose(q_m)=1` denote all 384 Pose6 coordinates
present and exact. At tolerance `tau`, the admissible set is

`F_tau={m : E_f,k(q_m)<=tau for every (f,k), and C_pose(q_m)=1}`.

If `F_tau` is nonempty, select `m_tau=argmin_(m in F_tau)(B_H(q_m),m)`. If it is empty, publish the
baseline only as an infeasible diagnostic. Measured exact rates were

`B_H(q_0..q_7)=(45369,34657,31431,20719,31203,20491,17265,6553)`.

For every requested `tau in {0,0.000152,0.000300,0.000500,0.000800}`, `F_tau` was empty. Therefore
there is no Task #613 constrained knee in this candidate family. Aggregate `M_overall` cannot replace
the universal stratum constraint.

# Pricing ladder and contest separation

The settled pricing ladder is

`flat diagnostic proxy (#602) -> exact fixed-width final ZIP (v2) -> exact variable-length entropy ZIP (v3)`.

The measured archive marginal is `25/37,545,489 = 0.000000665858953122` score units per byte, but the
contest action is not evaluated here. Membership and Pose completeness are advisory constraints,
not `d_seg` or `d_pose`.

0.1910828242 [contest-CPU] — unchanged.
