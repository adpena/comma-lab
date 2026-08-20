Stage 0 stopped honestly at `BLOCKED_AXIS_MISMATCH`; no rho or frontier claim was admitted.

- CP135 local result: 50,395 flips versus the promoted reference’s 34,964.
- T1R1/C1 local result: 47,950 flips.
- Local diagnostic rho: `0.07530491560921523`, explicitly non-admitted.
- C1 target also differed by one pixel: 17,927 versus the batch-16 reference’s 17,926.
- The V0–V5 ladder did not fire because its prerequisite same-axis decomposition is unavailable.
- Modal connectivity failed before dispatch, so no paid job or unharvested call exists.

All 19 GiB of archives, raw streams, receiver state, logits, argmax fields, checkpoints, and per-pair edge rows were retained. The implementation and handoff landed in commit `3325810a9a`.

Artifacts:

- [Handoff memo](/Users/adpena/Projects/pact/.omx/research/ddm_js1_stage0_per_edge_20260812.md)
- [Stage-0 receipt](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/STAGE0_RESULT.json)
- [Runner](/Users/adpena/Projects/pact/experiments/ddm_js1_stage0_per_edge.py)
- [Tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_js1_stage0_per_edge.py)

Verification passed: 3 tests, Ruff, compilation, receipt consistency, payload-retention gate, two review passes, and serializer post-commit hashes. The broad developer preflight remained red on 8 of 25 existing repository-wide gates; individual inspection found none implicated these js1 files.

No PoseNet, complete evaluator score, contest-CPU/CUDA row, V0–V5 result, or pointer movement was measured.

Effective composed pointer remains `S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`. Own-vehicle frontier remains `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN/js1 scorer-lane owner. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/`. Fire trigger: a working 1:1 T4 CUDA lane with artifact return becomes available; retain both promoted argmax fields, recompute the matched n600 edge map and rho, then fire V0–V5 only if Stage 0 is admitted.**

## LIVE-HYPOTHESES

- CUDA survival may differ substantially because the local renderer moved CP135 by 15,431 flips relative to its promoted row.
- Road-incident edges may remain the dominant CUDA allocation hub; both earlier m91 evidence and this local n600 diagnostic concentrate most flips there.
- Road↔MyCar may be the largest recoverable non-Lane pocket, but CUDA custody must re-rank it.

## DEAD-ENDS

- Using the Mac CPU map as Stage-0 rho is closed at INSTANCE scope because it fails both reference controls.
- Repeating Modal dispatch from the unchanged sandbox is closed until connectivity returns.
- Rate-side implicit-edge probability calibration remains closed at FORMULATION scope by sr1.
- More CP135 lossless-coder hunting remains closed by LP135/lv2 and cannot replace the missing scorer surface.