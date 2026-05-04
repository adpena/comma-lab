# Grand Council Stacking and Full-Pipeline Session - 2026-05-02

Evidence grade: `external` + `derivation` + local exact-artifact synthesis.
Score claim: `false` except where exact artifact paths are explicitly cited.

This session was spawned read-only to stress-test the fastest path from the
current public-floor basin toward the Shannon-floor objective. No files or
cloud resources were mutated by the council agent.

## Verdict

The critical path is no longer broad representation exploration. The fastest
measured route is the public-floor basin:

```text
semantic mask control
+ JointFrameGenerator
+ QZS3 renderer packing
+ QP1/qpose-style pose stream
+ single-blob archive layout
+ score-aware pose line search
```

This is the only local route already producing exact artifacts in the `0.31x`
band. Learned topology, Alpha/NeRV/soft-LUT, and inverse-steganalysis remain
valid parallel upside paths, but they should not displace the H100/T4
QZS3/QP1 pose loop until they can emit closed archives in the same basin.

## External Signals

- PR #67 reports `qpose14_qzs3_filmq9g_slsb1_r55`, score `0.31`, with QZS3
  grouped variable-bit-depth packing, delta/VLQ pose coding, and single-blob
  payload inside `inflate.py`: https://github.com/commaai/comma_video_compression_challenge/pull/67
- PR #65 reports `henosis_qz_n3z_r25_clean`, score `0.32`, with materially
  lower PoseNet but higher SegNet than PR #67/PR #63 style points:
  https://github.com/commaai/comma_video_compression_challenge/pull/65
- PR #64 reports `unified_brotli`, official eval `0.33`, and explicitly
  attributes its byte edge to single-stream Brotli, delta-encoded velocity,
  and dropping rotation from the pose side channel:
  https://github.com/commaai/comma_video_compression_challenge/pull/64
- PR #63 `qpose14` is the earlier Quantizr-style public-floor basin:
  https://github.com/commaai/comma_video_compression_challenge/pull/63
- PR #55 Quantizr and PR #56 Selfcomp remain design references for learned
  renderer/scorer-aligned training:
  https://github.com/commaai/comma_video_compression_challenge/pull/55
  https://github.com/commaai/comma_video_compression_challenge/pull/56
- NeRV is relevant as a post-deadline/parallel learned representation family,
  not as the immediate score loop unless it reaches the public-floor basin:
  https://arxiv.org/abs/2110.13903

## Local Evidence Boundary

Current local score truth remains exact CUDA archive evaluation of exact bytes
through `archive.zip -> inflate.sh -> upstream/evaluate.py`.

Known exact / diagnostic anchors at session time:

```text
C-053 A++ T4 QZS3/QP1 fixed-slice:
  score=0.3243472585872431
  bytes=276296
  sha=c5260473c26c4d4537d99d4a6a18b8ff0d9d1a901f6db17cd2208559e1010362
  evidence=experiments/results/lightning_batch/exact_eval_public_floor_qzs3_qp1_t4_20260502T0036Z/contest_auth_eval.adjudicated.json

C-054 A++ T4 first pose line-search checkpoint:
  score=0.3218613619571356
  bytes=276427
  sha=8c9000f67eb21f366299fe033e3e6031ab63992e8067758600e43d0091c9a9fa
  evidence=experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_t4_20260502T0100Z/contest_auth_eval.adjudicated.json

C-055 diagnostic H100 r8 continuation:
  score=0.3152653422017416
  bytes=276426
  sha=c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1
  evidence=experiments/results/vast_harvest/archive_eval_line_search_qzs3_qp1_fixedslice_continue_r3_20260502T0108Z/contest_auth_eval.json
  promotion_status=Lightning T4 running; no A++ claim until adjudicated T4 JSON lands
```

## Sub-0.30 Math

Using the H100 continuation as diagnostic only, current bytes and SegNet imply:

```text
rate_term = 25 * 276426 / 37545489 ~= 0.1840
seg_term  = 100 * 0.00061012 ~= 0.0610
rate + seg ~= 0.2451
```

For score `< 0.30`, the pose term must be roughly:

```text
sqrt(10 * pose_dist) <= 0.30 - 0.2451 ~= 0.0549
pose_dist <= (0.0549^2) / 10 ~= 0.000302
```

The current H100 diagnostic pose distance is about `0.000493`. The missing
score is therefore primarily PoseNet reduction, not raw bytes. Byte reductions
still matter, but only after they preserve the pose/SegNet basin.

## Lagrangian Rule

For every atom `a`, dispatch only when:

```text
E[score_drop(a | A)] >
  25 * byte_cost(a | A) / 37545489
  + uncertainty_penalty(a)
  + interaction_penalty(a | A)
```

This is water-filling over charged archive atoms. The first atom family with a
large measured ratio is pose-side QP1 refinement: C-054 paid about `+131` bytes
and gained about `0.00249` exact T4 score versus C-053.

## Anisotropic Pose-Manifold Search

Uniform scalar radii are now a baseline, not the optimal geometry. The pose
stream should be searched with anisotropic proposals over the one-scalar QP1
manifold first, then generalized to richer pose codecs if a charged qpose
residual clears the byte slope.

Immediate proposal families:

1. Sparse asymmetric integer stages with `--delta-sets`, e.g.
   `-34,-21,-13,-8,-5,-3,-1,0,1,2,3,5,8`.
2. Gradient-guided integer stages with `--gradient-delta-sets` and
   `--gradient-backtrack-deltas`, where the differentiable renderer/PoseNet
   path proposes `-sign(d PoseNetLoss / d col0)` moves and final acceptance
   remains rounded archive objective improvement.
3. Hard-pair one-hot windows from exact component traces.
4. DCT/spline/jerk temporal modes over velocity, because smooth pose changes
   can buy PoseNet reduction with fewer Brotli bytes than independent pair
   edits.
5. Transition endpoint and horizon/lane/log-zoom modes informed by video
   forensics and openpilot-like ego-motion priors.
6. Charged qpose residual atoms for non-velocity dimensions only after pair
   traces show that the byte cost clears the Lagrangian slope.

Gradient and directional search remain proposal mechanisms. The accepted
archive must still contain only charged payload bytes and must inflate without
scorer access.

## Reverse-Engineering Tactics

Contest-compliant tactics:

- Round-trip PR67/65/64 archives and compare decoded masks, poses, renderer
  tensors, generated frames, PoseNet outputs, SegNet outputs, and per-pair
  component traces.
- Attribute every segment: single-member container, QZS3 tensor groups,
  QP1/qpose streams, residual/DCT payloads, Brotli/VLQ length tables.
- Use public archives as references only. Submit only repo-built archives with
  our own provenance and exact CUDA eval.
- Use binary tools only when a submission ships native binaries. Current
  public-floor submissions are Python/Torch-style payloads, so byte anatomy,
  Python disassembly, tensor statistics, and exact trace comparison are higher
  EV than native RE.
- Treat exact comma hardware/camera metadata as priors until independently
  verified. The current usable constraints are video size, fps, route/ego
  motion proxies, vanishing point, horizon band, and openpilot-like PoseNet
  sensitivity.

## Next Dispatch Order

1. Harvest active T4 confirmations for line-search checkpoints.
2. If the r8 T4 packet confirms, promote it immediately and use it as deploy
   floor.
3. Continue H100 from the best accepted checkpoint with asymmetric/gradient
   proposal stages, not larger symmetric radii.
4. Build DCT/spline/hard-pair window proposal mode over QP1 col0; objective
   remains complete archive R(D), not proxy-only loss.
5. Exact component trace C-054/C-055 and compute pair-level marginal benefits.
6. Add qpose residual atoms only for pairs where PoseNet drop beats charged
   bytes.
7. Isolate PR65 residual/DCT/patch machinery as a possible stackable atom.
8. Keep Q-FAITHFUL/NeRV/soft-LUT as parallel H100/H200 work only after their
   dispatch claims and retraining gates are clean.

## Preflight Hardening Required

- Keep adaptive candidate caps for PyTorch `canUse32BitIndexMath`.
- Keep metadata-driven QZS3/QP1 slicing; never rely on one brittle split point.
- Copy exact evaluated `archive.zip` into result dirs and record custody JSON.
- Pin Torch wheels by driver and bootstrap scorer dependencies before eval.
- Clean inflated raw frames after preserving canonical JSON/provenance/report.
- Enforce active dispatch claims before any GPU job.
- Treat Lightning SDK status as telemetry only; terminal state-derived
  artifacts with local validation are the custody surface.

## Inspected Files And Commands

The council inspected:

- `src/tac/quantizr_qzs3_codec.py`
- `src/tac/qp1_pose_codec.py`
- `src/tac/quantizr_faithful_renderer.py`
- `experiments/line_search_pose_refinement.py`
- `experiments/reverse_engineer_top_submissions.py`
- `experiments/compare_component_traces.py`
- `.omx/research/top_submission_reverse_engineering_canonical_repro_20260501_codex.md`
- `.omx/state/active_lane_dispatch_claims.md`

Local verification after incorporating the council direction:

```text
.venv/bin/python -m py_compile experiments/line_search_pose_refinement.py src/tac/tests/test_line_search_pose.py
.venv/bin/python -m pytest -q \
  src/tac/tests/test_line_search_pose.py::test_gradient_guided_delta_matrix_orients_by_descent_direction \
  src/tac/tests/test_line_search_pose.py::test_gradient_delta_sets_run_differentiable_proposal_stage \
  src/tac/tests/test_line_search_pose.py::test_parse_magnitude_sets_normalizes_positive_magnitudes \
  src/tac/tests/test_line_search_pose.py::test_directional_delta_sets_allow_sparse_asymmetric_search
result=4 passed
```

## Execution Update - 2026-05-02T01:36Z

The first Grand Council dispatch branch has landed:

```text
claim=C-056
archive=experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_r8_t4_20260502T0110Z/archive.zip
archive_sha256=c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1
archive_bytes=276426
score=0.3159064496962538
pose=0.00049846
seg=0.00061244
hardware=Tesla T4
samples=600
evidence=A++ contest T4
promotion_eligible=true
```

The immediate next branch remains exactly aligned with the council verdict:

```text
r13_h100_score=0.31514356926681697
r13_archive_sha256=d3f3300531886d9dcb3553baffdd201567e3adaf7b746a7f405b15ad6c23b148
r13_t4_job=exact_eval_line_search_qzs3_qp1_r13_t4_20260502T0128Z
active_h100_session=pact_ls_qzs3_gradient_r13_0129
active_h100_policy=gradient/directional anisotropic pose proposal over charged QP1 archive
```

Local tool support for the next vector-manifold tranche is now in place:

```text
file=experiments/line_search_pose_refinement.py
new_cli=--basis-delta-sets
new_cli=--basis-modes
new_cli=--basis-pair-indices
new_cli=--basis-window-radius
contract=DCT/pair-window vector proposals over QP1 col0; accepted only by complete archive R(D) objective
verification=py_compile; 5 focused tests passed; git diff --check passed
```
