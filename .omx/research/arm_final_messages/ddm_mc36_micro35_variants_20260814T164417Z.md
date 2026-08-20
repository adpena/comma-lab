Variant C is the sole admitted successor and has a sealed, unfired T4 order:

| Variant | Seg flips | Rate | Δd_pose | Result |
|---|---:|---:|---:|---|
| A | 35 | +61 B | −3.239e−10 | Rate fail |
| B | 37 | +40 B | +5.441e−9 | Rate/Pose fail |
| C | 37 | +17 B | −1.463e−10 | All local gates pass |

C’s archive is 186,269 bytes, SHA-256 `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`. Parse-back and deterministic repeat passed. These are `[macOS-CPU advisory] NON-PROMOTABLE` results; no full-n600 T4 scorer, exact evaluator, Modal job, or frontier promotion occurred.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_mc36_micro35_variants_20260814.md)
- [Resumable builder](/Users/adpena/Projects/pact/experiments/ddm_mc35_micro35_union_build.py)
- [Focused tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_mc35_micro35_union_build.py)
- [Sealed fire order](/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/SEALED_FIRE_ORDER.json)

All payloads remain retained. Verification passed: 46 focused tests, Ruff, bytecode compilation, payload-retention guard, archive integrity/repeat checks, and two review passes. Committed as `2e4abc62108979131a073a9f255b2be0e7a10742`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatch/ddm_mc36_dual_axis_t4_r1`; fire trigger: no active full-n600 scorer lane, MAIN claims `ddm_mc36_dual_axis_t4_r1`, and the sealed archive/runtime SHAs match; action: fire the approximately $0.16 RE1T/JS1B dual-axis T4 worker and retain every required return.

## LIVE-HYPOTHESES

- C’s local improvement may survive T4 because receiver closure is exact and all seven recomputed base Seg fields match the retained T4 fields pixel-for-pixel. Candidate scorer transfer remains untested.

- Drop-532 and constrained pair-105 compensation interact favorably in the real HP4 container: C is 23 bytes smaller than B and 44 bytes smaller than A. Its exact contest-score effect remains untested.

## DEAD-ENDS

- A alone is closed for this instance: it misses the unchanged rate gate by 32 bytes.

- B alone is closed for this instance: it misses rate by 11 bytes and Pose by `4.843531960428793e-9`.

- Gate-boolean-only C rejection is closed: measured A/B complementarity justified the real C build, which passed every local gate.

- Additive projections, stale compensation transfer, and local admission as promotion authority remain closed. Own-vehicle frontier remains LC2 `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`.