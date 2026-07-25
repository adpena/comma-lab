---
equation_id: ddm_rg4_g3_blocks_and_active_tube_laws_v1
date_utc: 2026-07-25
research_only: true
score_claim: false
---

# Source-local counted receiver

For exact candidate camera bytes `W`, admitted PC1 receiver `P`, and counted
packet coordinate `q`,

`C(q; W) = clip_uint8(W + int16(P(q; W)) - int16(P(0; W)))`.

Therefore `C(0; W) = W` exactly. The generic evaluator/receiver is free code;
the parent archive and serialized PC1 packet are counted. In this arm
`q_increment = +/-256`, which equals one physical quantum because each sealed
PC1 `xi_scale` is the physical quantum divided by 256.

# Joint action and admission

For exact archive bytes `B`,

`S = 100 d_seg + sqrt(10 d_pose) + 25 B / 37,545,489`.

A four-pair proposal is locally selectable only if it changes receiver bytes
and has both `delta d_pose < 0` and `delta S < 0`. Local selection is not the
verdict; source and final states are remeasured at `n=600`, batch 32.

# Full Pose active tube

For pair `i`, exact PoseNet output `p_i`, sealed center `c_i`, and low-rank
factor `L_i`,

`Q_i = ||L_i (p_i - c_i)||_2^2`.

Full membership is

`max_i Q_i <= rho^2`, with `rho = 0.05`.

The optional per-output diagnostic uses

`a_j = mean_i [L_i(p_i-c_i)]_j^2`,

`slack_j = rho^2 / 6 - a_j`.

Dimension `j` is diagnostic-active when `slack_j <= 0`. This is not six
independent constraints and cannot replace the full membership test.

# RG3 false-coverage law

Let `E_{p,b,f,m,s}` be the measured target-bucket event count for exact pair
`p`, bucket `b`, production family `f`, magnitude `m`, and sign `s`.

For each of the 25 terminal missing blocks,

`E_{p,b,f,m,s} = 0` for every production RG3 `f`, every admissible `m`, and
both `s in {-1,+1}`.

Thus the correct closure is a typed instance exclusion:

`coverage_proven = false`,

not a positive causal block. A new coordinate family is required before a new
positive-coverage attempt.
