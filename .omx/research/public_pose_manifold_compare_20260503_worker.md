# Public Pose Manifold Compare - 2026-05-03 Worker

## Scope

- Tool: `experiments/compare_public_pose_manifolds.py`.
- Scope: local C102-native QP1 pose-manifold comparison and planning only.
- Remote dispatch: `false`.
- Score claim: `false`.
- Archive built: `false`.
- Exact-eval readiness: `false` for every emitted row.

## Anchor

- C102 archive bytes: `276485`.
- C102 archive SHA-256: `79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8`.
- C102 score: `0.31514430182167497`.
- C102 PoseNet: `0.00049337`.
- C102 SegNet: `0.00060804`.

## Sources

- `C102` available=`True` archive_bytes=`276485` pose_bytes=`1140` reason=`None`.
- `PR75` available=`True` archive_bytes=`276741` pose_bytes=`1140` reason=`None`.
- `PR77` available=`True` archive_bytes=`276551` pose_bytes=`1140` reason=`None`.
- `PR65` available=`True` archive_bytes=`284425` pose_bytes=`1140` reason=`None`.

## Top Planning Rows

- `c102_qp1_active_subspace_proxy_public_plus_smooth_top096`: pairs=`96`, estimated_payload_bytes=`240`, expected_benefit_proxy=`0.0008313743081992646`, break_even_required=`0.005304107970424299`, margin=`-0.004472733662225034`, readiness=`false`.
- `c102_qp1_velocity_col0_neighbor_smooth_top048`: pairs=`48`, estimated_payload_bytes=`112`, expected_benefit_proxy=`0.0004543149181376562`, break_even_required=`0.005218878024424667`, margin=`-0.004764563106287011`, readiness=`false`.
- `c102_qp1_dct_lowfreq_residual_pull_top064`: pairs=`64`, estimated_payload_bytes=`152`, expected_benefit_proxy=`0.00045568708654140363`, break_even_required=`0.005245512382549555`, margin=`-0.004789825296008151`, readiness=`false`.

## Artifacts

- Plan JSON: `/Users/adpena/Projects/pact/experiments/results/public_pose_manifold_compare_20260503_worker/pose_manifold_compare_plan.json`.
- Candidate CSV: `/Users/adpena/Projects/pact/experiments/results/public_pose_manifold_compare_20260503_worker/candidate_rankings.csv`.
- Pose sources JSON: `/Users/adpena/Projects/pact/experiments/results/public_pose_manifold_compare_20260503_worker/pose_sources.json`.

## Dispatch Boundary

Any future exact-eval candidate must be built by a separate byte-closed archive builder, must pass local stream-closure gates, and must claim the lane before dispatch. This artifact is not sufficient for a score, rank, promotion, or retirement claim.
