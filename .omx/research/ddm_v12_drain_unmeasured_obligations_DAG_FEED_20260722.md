# FEED-603 V12 obligation drain — resolved formulation fork

**Producer:** `ddm_v12_drain_unmeasured_obligations`; tasks #603/#613, master #578. Evidence is
`[macOS-CPU frozen-scorer advisory]`, `score_claim=false`, `d_seg_claim=false`, and non-promotable.

The V11 measurement debt is resolved. All 4,096 bounded n600 atoms have terminal dispositions:
3,994 received exact canonical-batch Seg/Pose measurement, 66 were fail-closed receiver no-ops, and
36 were lower-predicted-EV address conflicts with earlier admitted values. Greedy joint-objective
waterfill admitted 44 bundles / 407 atoms. The archive moved 102,105→106,106 bytes; d_seg
0.034502249824→0.034003668891; d_pose 163.039648911962→163.034719422881. The exact 16/49/98/147
KiB requested rungs all select the same 4,001-byte correction state, so four consecutive upper
budget rungs are flat and the 200KB ceiling is non-binding.

**Resolved fork:** `ADVISORY_FORMULATION_PLATEAU_WITH_200KB_CEILING_NONBINDING_V6_SUCCESSOR_NAMED`.
Scope is post-solve correction of the bound v6 0.0345 predictor only. Movable ends at d_seg
0.989518086727 and Lane at 0.436911324151; they worsen while incidental Road/MyCar changes carry
the small aggregate gain. The v6 successor must predict Movable island worldsheet events natively
in PREDICT, not ask the correction layer to create the missing worldsheet. Chart/event/carrier
families and the describe-line paradigm remain open.

Anchor `ddm_describe_line_rate_distortion_bracket_v1` is amended append-only from an n600 derived
projection to a measured receiver-closed n600 row. Receipt SHA-256:
`eab2ef2478fb07f6a3242781887442c3fc49e9c34e10bd73a93f25d9a0262f0a`.

**Next node:** PREDICT-stage v6 successor design/build carrying native Movable island worldsheet
events, with the same scorer-custody and exact-objective admission contract. No paid dispatch and
no candidate promotion are authorized by this feed. `0.1910828242 [contest-CPU]` is unchanged.
