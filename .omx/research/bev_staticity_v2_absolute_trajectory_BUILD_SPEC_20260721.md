# BEV staticity v2 absolute-trajectory build spec

Status: `IMPLEMENTATION_SPEC` · `research_only=true` · `$0` · `[macOS-CPU advisory]` · `MAIN_REVIEW_REQUIRED=true`

## Authority and stopping rule

The delegated authority is
`/Users/adpena/Projects/pact/.omx/tmp/codex_runs/bev_staticity_v2_absolute_trajectory_20260721T174021Z.wrapped.prompt.txt`
with SHA-256 `a1851c6c126476e03adca459dc850dc3d16c851cbe9da5f453b194dc8ab22d66`.
The current contest-CPU pointer is `0.1910828242` and this lane cannot move it.

Run the D0 hood positive control at n64 first. It passes only when the bottom-connected MyCar/hood boundary has p50 residual at or below the registered 1 px floor, at least the registered 50% of samples are within 1 px, the absolute-pose chain is finite SE(3), and the hood world-to-ego round-trip closure is below `1e-9` m. The 50% threshold is the existing canonical probe-outcome reactivation criterion; do not silently tighten or weaken it. If any condition fails, stop before n600 and do not compute or interpret Road/Lane D1-D3. Report the narrow residual and the first failed stage.

## Input custody

- Frozen cache: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
- G1 calibration authority: `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json`, consumed through the existing LawRef resolver. Do not re-fit its constants.
- Seed: `1234`; CPU Torch deterministic algorithms; no MPS and no scorer proxy.
- Evidence root: `/Volumes/VertigoDataTier/pact/evidence/bev_staticity_v2_20260721/` with atomic per-chunk checkpoints and no durable `/tmp` references.

## D0.1 — canonical frame-zero labels

Replace the G1 batch-32 frame-zero sidecar with a source-hash-bound, resumable canonical sidecar produced one frame at a time, matching `precompute_gt` scorer geometry. For every processed pair, score both cached f0 and f1 as singleton calls; compare f1 to the frozen `lstars` plane and fail closed on any mismatch. The n64 prefix must therefore resolve the two known batch-32 mismatches in pairs 0-31 to exactly zero. Preserve the old batch-32 receipt as historical evidence; never rewrite it.

## D0.2 — absolute f1 trajectory

The cached `gt_poses[t]` is the within-pair raw PoseNet target f0[t]→f1[t], not an absolute pose and not the missing f1[t-1]→f0[t] transition. Build the missing transition directly with the same frozen CPU PoseNet singleton path on cached `(gt_f1[t-1], gt_f0[t])`.

Resolve each raw six-vector through the settled G1 `xi_from_pose_calibration` mapping. Anchor `A_f1[0]=I`. For `t>0`, compose with `tac.lie` in temporal order:

`A_f1[t] = A_f1[t-1] · exp(xi_cross[t]) · exp(xi_within[t])`.

Store the raw cross target, calibrated twists, cumulative pose, source hashes, and stage identity per chunk. Validate homogeneous last rows, finite values, and SE(3) inverse/compose closure. Never re-difference the pairwise targets.

## D0.3 — hood control

Isolate only the MyCar connected component touching the bottom image edge; other MyCar islands are not the hood. Lift its scorer-grid boundary into the registered camera/ground proxy, transform it with `A_f1[t]`, and bring it back with `inverse(A_f1[t])` for an explicit ground/world-to-ego closure check. Measure temporal hood silhouette residuals only after that closure. Record both closure error and label-boundary residual so a tautologically correct group inverse cannot hide a bad label/control surface.

## D1-D3 (strictly gated)

Only after D0 passes at n64 may n600 run.

- D1: transport Road/Lane bottom-ground boundaries by `A_f1[t]` into the one fixed world frame; report p50/p90 residuals and fraction within 1 px. Movable remains the expected nonstatic control.
- D2: on correspondence-preserving static segments, report directrix-plus-ruling residuals and the existing refusal to claim unstable raw discrete Gaussian curvature. Every negative is formulation-scoped.
- D3: estimate `{static ground coefficients + absolute-xi B-spline knots + sparse events}` against the same signature baseline only when D0 and the corresponding D1/D2 stratum pass. This remains a receiver-open byte estimate, not a score or promotion row.

## Required code and tests

Edit only:

- `tools/measure_bev_staticity_developability.py`
- `tools/tests/test_measure_bev_staticity_developability.py`

Tests must cover singleton scorer-geometry custody logic without loading scorers, exact temporal composition ordering with noncommuting synthetic SE(3) increments, bottom-connected hood isolation, world-to-ego closure, n64 refusal semantics, and no D1-D3 authorization when D0 fails. Run focused pytest and py_compile. Complete two explicit `review_tracker.py mark-file ... --status reviewed` passes for each Python file. Never use `REVIEW_GATE_OVERRIDE=1`.

## Deliverables and landing

Land a dated result memo, compact receipt JSON, DAG FEED, and REUSE MANIFEST. Include exact argv, hashes, test counts, measured D rows or exact stopped-stage blocker, SSD receipt paths/hashes, pointer-unmoved statement, verdict scope, and `MAIN_REVIEW_REQUIRED=true`. Commit through `tools/subagent_commit_serializer.py` with base/post content hashes. MAIN must review the branch diff before merge or downstream consumption.
