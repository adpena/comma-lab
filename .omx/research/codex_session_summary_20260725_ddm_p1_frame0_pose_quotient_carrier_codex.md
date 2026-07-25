# Codex session summary — DDM P1 frame-0 pose quotient carrier

Status: `AWAITING_MAIN_LANDING_REVIEW`

Research-only: `true`

Pointer: `0.1910828242 [contest-CPU]` (`UNMOVED`)

P1 independently derived and implemented an all-state-counted, legacy-compatible
PC1 frame-0 quotient carrier, canonical covariance-tail rank law, resumable
local runner, exact parse-back packet, six-rank reach curve, and seeded
matched control.

The preregistered endpoint did not pass. The best exact n600/batch32 treatment
row was rank 1, `d_pose=19.89493129583306` at 3,520 carrier bytes. All six ranks
were below the 30,000-byte bar, frame 1 and Seg cells were byte/digest
identical, and treatment-control `d_seg=0.0`.

Verdict:
`P1_SHARED_LOW_RANK_FRAME0_ACTUATOR_FORMULATION_BLOCKED`.

Named obstruction:
`SHARED_BASIS_TARGET_ACTUATOR_SPECTRAL_TAIL_PLUS_EXACT_UINT8_TRUST_REGION_CROSSING`.
This does not close nonlinear, pair-conditioned, higher-rank, or scorer-solved
frame-0 quotient generators.

Consumer: `FEED-603-p1` routes P3/G5 away from this carrier and retains
PC1-joint-descent (`j11/#366`) as the live pose path. MAIN must independently
review the branch, landing manifest, negative scope, and exact custody before
merge or landing disposition.
