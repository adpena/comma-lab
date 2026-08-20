The pz4 pre-proof passes and is fired to MAIN routing.

- Winner: `r6_b12_global`
- Banked-output MSE: `1.0985637375134246e-6` versus `<2.5e-6`
- Exact rate envelope: `168,005 B`, saving `19,221 B` versus LC2
- Encoded PGQ1 gauge: `3,837 B`
- Passing configurations: `19/330`
- Axis: `[macOS-CPU scorer-free banked-output MSE + exact real-coder rate envelope]`

All candidate payloads and checkpoints are retained in [preproof_v3](/Volumes/VertigoDataTier/pact/ddm_pz4p_20260811/preproof_v3). The resume audit verified 330 receipts; an independent audit verified 3,300 payload records and 2,640 checkpoints. Dedicated tests: `19 passed`; payload-retention findings: zero.

Delivered:

- [Durable receipt and handoff](/Users/adpena/Projects/pact/.omx/research/ddm_pz4p_pose_gauge_preproof_20260811.md)
- [Pre-proof runner](/Users/adpena/Projects/pact/experiments/ddm_pz4p_pose_gauge_preproof.py)
- [Tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_pz4p_pose_gauge_preproof.py)
- Commit: `749f4677f8`

No scorer ran. The unchanged receiver cannot parse PGQ1, so this is not a submission candidate, rendered `d_pose`, or exact score. Own-vehicle frontier remains **cp135 `S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`**; the pre-proof did not move it or reach sub-0.15.

## NEXT_IF_RESUMED

- **FIRED-TO-MAIN-ROUTING** — owner: `MAIN`, assigning `ddm_pz4_joint_target_conditioned_receiver`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/`; fire trigger: satisfied by `19,221 B` exact-envelope savings and `1.0985637375134246e-6` MSE. Build a resumable receiver that consumes PGQ1 and removes CPR1 before requesting a scorer lane.

## LIVE-HYPOTHESES

- A jointly trained PGQ1-conditioned receiver can realize much of the measured byte saving because PGQ1 preserves the shipped carrier’s six pose outputs below the registered surrogate gate.
- Refitting PGQ1 to CP135/PR135 may retain a similar byte scale because the encoded object remains `600×6`, but cross-lineage transfer is unmeasured.

## DEAD-ENDS

- Rank 2 is closed for this LC2/PGQ1 gate: best MSE was `2.1486e-4`, roughly 86 times the limit.
- Ranks 1–5 are closed for this instance and formulation; rank 5 bottoms at `5.202855334725472e-6`.
- Frozen post-hoc CPR1 gauges and low-rank-plus-exact-residual repacks should not be retried: prior measured savings were only `64 B`, while the residual form cost `+4,316 B`.
- Treating the retained envelope as a score row is forbidden: no receiver parses PGQ1 and no rendered output was scored.