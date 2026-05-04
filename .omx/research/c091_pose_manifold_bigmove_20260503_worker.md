# C091 Pose Manifold Big-Move - 2026-05-03 Worker

## Evidence Boundary

- Scope: local C091-native QP1 pose-manifold planning/build only.
- Remote dispatch: `false`.
- Score claim: `false`.
- Promotion eligible: `false` until exact CUDA auth eval on identical archive bytes.
- Existing in-flight duplicate guard: `exact_eval_pr65_pose_qp1_c091_c089_actions_p6_t4_20260503T1158Z` was not duplicated.

## Anchor

- C091 archive bytes: `276481`.
- C091 SHA-256: `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`.
- C091 score: `0.31516575028285976`.
- Pose contribution: `0.07026450028285976`.
- Seg contribution: `0.060804000000000004`.

## Candidate Verdict

- Top candidate: `c091_native_cem_pose_waterfill_top128_s025`.
- Archive bytes: `276489`.
- Archive SHA-256: `1c9be2dd14b2607b9a86ecc071d6a37842896d18d62449885dac58d55afbbd64`.
- Changed pose pairs: `128`.
- Local closure gates: `True`.
- Break-even plausible: `True`.
- Dispatch recommendation: `exact_eval_candidate_after_claim_not_dispatched`.
- Sub-0.314 component gain required: `0.0011710771544847232`.
- Proxy component gain: `0.0014904868358843566`.

Adversarial verdict: this is a real non-noop C091-native pose residual move, not public stream copying. It is still trace-proxy evidence only; exact CUDA auth eval can easily reject it if PoseNet/SegNet response is nonlocal or antagonistic.

## Artifacts

- Plan JSON: `/Users/adpena/Projects/pact/experiments/results/c091_pose_manifold_bigmove_20260503_worker/plan.json`.
- Recommendation JSON: `/Users/adpena/Projects/pact/experiments/results/c091_pose_manifold_bigmove_20260503_worker/exact_eval_recommendations.json`.

## Tests

- Focused test target: `src/tac/tests/test_plan_c091_pose_manifold_bigmove.py`.
