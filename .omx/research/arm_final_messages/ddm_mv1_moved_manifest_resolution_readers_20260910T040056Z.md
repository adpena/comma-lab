Implemented and verified the [patch](/Users/adpena/Projects/pact/.omx/research/ddm_mv1_20260910/landing.patch): **171 tests passed**, two skipped; real rp1 raw verified. Two review passes completed.

**Not landed:** Git object writes were denied. The verified fallback bundle and receipts are in the [handoff](/Users/adpena/Projects/pact/.omx/research/ddm_mv1_20260910/FINAL_HANDOFF.json). Live source and payloads remain untouched.

## NEXT_IF_RESUMED

- **QUEUED — MAIN:** consume the handoff and install readers last, when rp1/tc3/vr6 stages are quiescent and Git is writable.
- **QUEUED — MAIN:** consume [census.json](/Users/adpena/Projects/pact/.omx/research/ddm_mv1_20260910/census.json) after installation; migrate 176 candidates, then strict-flip only at measured zero.
- **QUEUED — tc3 via MAIN:** validate the next public600 stage; consumer store `/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/`; trigger: fresh source binding and stage boundary.

**LIVE-HYPOTHESES:** The full tc3 stage should clear its former failure point—the exact `file_fact` operation passed on the real raw. Full replay remains untested.

**DEAD-ENDS:** Symlinks do not satisfy the ExFAT manifest contract. Rewriting logical paths unnecessarily changes receipt identity. Immediate strict activation is unsupported by the nonzero census. Moved custody never authorizes destination deletion.

OWN-VEHICLE FRONTIER UNMOVED: **S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600]**.