---
title: Candidate equations - receiver-priced member tolerance wall
date_utc: 2026-07-22T04:08:36Z
task: 603
feeds_task: 613
lane_id: lane_ddm_mdl_member_solve_v2_priced_603_20260722
research_only: true
---

# Decision and exact rate

Let `z` be the six-stream receiver-consumable chart description and `A(z)` its deterministic final
ZIP_STORED archive. The only admitted rate is

`B(z) = len(A(z))`.

For residual stratum `s` and safe-zero proposal `q_s(z)`, the measured marginal rate is

`Delta B_s = B(q_s(z)) - B(z)`.

The reverse-waterfill admission order is exact rate first:

`admit(q_s | tau) only if Delta B_s < 0 and E_{f,k}(q_s) <= tau for every stratum family f and key k and C_pose(q_s)=1`.

Here `E_{f,k}=1-M_{f,k}` is the frozen-SegNet same-cell escape fraction and `C_pose` is the exact Pose6
coordinate completeness of the receiver payload. The rung curve is

`C(tau) = (tau, B(z_tau), M_overall(z_tau), C_pose(z_tau))`.

# Measured fixed-width identity

The current grammar requires a fixed number of fixed-size anchor, gradient, residual, and Pose
records for n64. Therefore the real encodes measured

`Delta B_low = Delta B_mid = Delta B_high = 0`

at every tolerance. The selected description remains the baseline for all rungs:

`B(z_tau)=274664`, `M_overall(z_tau)=0.493605613708`, and `C_pose(z_tau)=1`.

Every rung is infeasible because at least one `E_{f,k} > tau`; Road, Lane, MyCar, and Movable each
have `E=1`. The Task #613 knee is undefined on this flat rate curve. This identity is scoped to the
fixed-record grammar and does not apply to a future receiver-proven variable-length code.

# Contest action separation

The contest action `100*d_seg + sqrt(10*d_pose) + 25*B/N` is not evaluated here. Membership and Pose6
code completeness are advisory constraint statistics, not `d_seg` or `d_pose`. The exact marginal
rate price recorded for future admitted steps is `25 / 37,545,489`; no step crossed the strict byte
break-even gate.

0.1910828242 [contest-CPU] — unchanged.

