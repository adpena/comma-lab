Completed and landed SG2B as commit `2ef37784dee47ebd4d8a069b276423c927f3cdad`.

| Row | Changed sites | Advisory archive |
|---|---:|---|
| p00 | 0 | 180,002 B, `cbb8d928…` |
| p01 | 1,084 | 179,833 B, `83660e34…` |
| p02 | 2,831 | 179,496 B, `440244d4…` |
| p03 | 9,723 | 178,014 B, `9a77787c…` |

The p00 control reproduced AFR1’s 113,411-byte RC64 stream exactly over 600/600 frames. All candidate fields, overlays, archives, receiver pins, and four canonical advisory dry-runs passed. No scorer or Modal job ran; MAIN retains the scorer lane.

Artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_sg2b_scmdl_distortion_leg_build_20260901.md)
- [Compose harness](/Users/adpena/Projects/pact/experiments/ddm_sg2b_scmdl_distortion_compose.py)
- [Tests](/Users/adpena/Projects/pact/tests/test_ddm_sg2b_scmdl_distortion_compose.py)
- [MAIN_FIRE_ORDER.json](/Volumes/APDataStore/pact/ddm_sg2b_scmdl_distortion_leg_build/MAIN_FIRE_ORDER.json)
- [BUILD_DONE.json](/Volumes/APDataStore/pact/ddm_sg2b_scmdl_distortion_leg_build/BUILD_DONE.json)

Validation: 9 tests passed, Ruff passed, payload-retention findings were zero, both Python review passes were recorded, and post-commit hashes matched. The repository-wide developer preflight’s observed failure was seven pre-existing lane-registration references in `ddm_lc3`/`ddm_ltg1`; no SG2B finding was reported.

Frontier remains `[contest-CUDA T4 n600]` AFR1: `S=0.14797617125559104 @ 180,002 B`, SHA `cbb8d928…d405bf25`.

## NEXT_IF_RESUMED

- **GATE_NULL_IDENTITY** — owner: `MAIN sole scorer-lane router`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sg2b_scmdl_distortion_leg_build/fire_main/p00`; fire trigger: free scorer slot, no duplicate lane, and p00 pin/dry-run revalidation; action: fire once and require exact AFR1 `d_seg=0.00020139`, `d_pose=6.37e-06`.
- **QUEUED_MAIN_ADVISORY** — owner: `MAIN sole scorer-lane router`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sg2b_scmdl_distortion_leg_build/fire_main/{p01,p02,p03}`; fire trigger: p00 identity passes and each preceding row is terminal; action: fire p01, p02, then p03 once each.
- **ADMIT_OR_REFUTE_GATE_2** — owner: `MAIN`; consumer store: `/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/HANDOFF.json` plus RXC1’s exact-delta store; fire trigger: advisories terminal and RXC1 publishes `GATE-1-PASSED`; action: join measured distortion with exact refit bytes and record the gate-2 verdict.

## LIVE-HYPOTHESES

- p01 may reduce `d_seg`: it concentrates realized-terminal disagreement into only 1,084 boundary sites, plausibly limiting spill.
- At least one proposal may achieve negative `delta d_seg` because this realized-cell signal was absent from the closed token-GT families.
- Some rate credit may survive RXC1 refitting because all three fixed-G vehicles became smaller, though G/M coupling prevents transferring those deltas directly.

## DEAD-ENDS

- Frames-direct scoring is closed: the canonical advisory accepts sealed runtime/archive inputs.
- Hand-composed rendering is closed: the shipped RX1M parser and riders must execute before `render_video`.
- Token-GT acceptance is closed by the prior broken realized-transfer result.
- Detached batching did not survive the managed shell; the four legal, individually bounded checkpointed runs completed instead.
- Fixed-G archive deltas are not rate admissions; only RXC1’s refitted exact-delta result may enter gate 2.
- The initial `runtimes/p00` lookup was wrong; finalize refused before firing, and the corrected `runtimes/p00_null` path is regression-tested.