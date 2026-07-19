# VJP custody and positive-band implementation spec (2026-07-19)

Authority: `vjp_custody_positive_bands_20260719_20260719T044504Z.wrapped.prompt.txt`,
sha256 `4cb7251603e1e992608f96cd506766ea2801214723057767bc5a8f463a602114`.
This is an isolated-worktree implementation plan for lane
`lane_vjp_custody_positive_bands_20260719`; MAIN must review and merge it.

## Objective

Close the two explicit #549 positive-band blockers with real frozen CPU-Torch
derivatives and feed their hashed per-pair sidecars additively into
`tools/measure_joint_seg_pose_rate.py`. Preserve the existing zero-band behavior
and interval/lattice solver semantics. The eventual measurement must cover at
least 24 unique real `gt_n600.npz` pairs, two positive Seg scales, and two
`tau_pose` values, with actual Brotli-Q11/zstd-19 range-coordinate bytes, hard
Seg/Pose oracles, repairs, binding maps, and measured-curve waterfill.

This lane has no launch, paid-compute, score, promotion, or frontier-pointer
authority. All results are `[macOS-CPU advisory]`, `score_claim=false`, and the
pointer remains `0.1910828242 [contest-CPU] UNMOVED`.

## Ground-truth constraints

- Input cache is exactly
  `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`,
  sha256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
  It is ZIP_STORED and must be member-memmapped; never load the 4.7 GiB file as
  a whole. Its members are `gt_f0`, `gt_f1`, `lstars`, `margins`, `gt_poses`,
  and `n_pairs`. It contains no logits and no rival IDs. Therefore the producer
  uses cached `lstars` as winner, derives the rival from a fresh exact frozen
  native-fp32 logit forward, records cache/native winner and margin agreement,
  and fails closed for a requested pair when the active arrangement is not
  compatible. It must never label regenerated rivals as cached.
- Frozen sources are the pinned read-only files under
  `/Users/adpena/Projects/pact/upstream`: `modules.py` sha256
  `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`,
  `frame_utils.py` sha256
  `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90`,
  SegNet weights sha256
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`,
  and PoseNet weights sha256
  `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`.
- Sacred read-only tree:
  `/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z/`.
  Snapshot its metadata before/after producers and measurements and refuse any
  change.
- Durable large outputs go only under
  `/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/` (or the second
  SSD tier only if the first is unavailable). Refuse `/tmp`, local bulk, or an
  output tier without a conservative free-space preflight. Use same-directory
  temporary files, atomic `os.replace`, success-only scratch cleanup, and
  machine-readable reconstruction/cold-store metadata. Never delete final
  evidence automatically.
- Each producer invocation is 1..12 unique pairs, resumable from a manifest,
  with a distinct immutable per-pair NPZ and hash. Existing sidecars are reused
  only after full config/source/hash validation; never overwritten.

## Exact derivative definitions

### SegNet active-field VJP

For each pair, let `w(h,w)=cached L*` and let `r(h,w)` be the highest native
logit class excluding `w`. For the frozen spatial 3x3 head weights, define

`n(h,w) = (W_w - W_r) / ||W_w - W_r||_2`.

The scalar active-field functional is the sum over scorer cells of the
winner/rival logit difference divided by that exact head-weight norm. In code,
its exact logit cotangent is `+1/norm` at `w`, `-1/norm` at `r`, zero elsewhere.
One real SegNet autograd VJP must produce:

- `g_y = d functional / d y1`, fp32 `(384,512,3)`, where `y1` is the shared
  scorer RGB plane consumed by the interval solver;
- `g_x = A^T g_y`, fp32 `(874,1164,3)`, obtained through the exact upstream
  bilinear resize graph (or an explicitly parity-tested exact adjoint).

Persist both. This is a field-level aggregate VJP, not 196,608 independent
full-frame per-cell Jacobians. Do not claim otherwise. For the landed band law,
factor the measured scorer-plane VJP pointwise as
`g_y = Lip_local * q`, where `Lip_local=||g_y||_2` and `q=g_y/Lip_local`
(zero where the norm is zero). This makes `Lip_local*abs(q_c)=abs(g_y_c)` and
contains no guessed Lipschitz constant. Store raw `g_y`, `g_x`, `q`, the
measured `Lip_local` field, winners, rivals, cached/native margins, exact
head-pair norms, tensor hashes, dtype/shape, and an `A^T` relation residual.

Round-1 unit-normal review must independently verify: head kernel shape,
per-cell `||n||=1` within fp32 tolerance, seed signs/classes, and a directional
finite-difference check of the scalar functional/VJP on a bounded real pair.

### PoseNet first-six Jacobian

Use the solver-native representation:

`J_y = d pose[:6] / d (y0,y1)`, fp32 shape `(6,2,384,512,3)`.

Here `y0,y1` are the exact shared scorer planes before YUV6. Feed them through
the canonical differentiable `tac.differentiable_eval_roundtrip.
differentiable_rgb_to_yuv6`, then the frozen PoseNet core. First verify its
forward output against the upstream no-grad preprocessing on the same real pair.
For each of six scored outputs, run a real reverse pass. Also compute the
camera pullback `J_x=A^T J_y`, fp32 `(6,2,874,1164,3)`, or, if retaining all of
`J_x` would violate the preflight, persist its row hashes/norms plus an exact
rebuild recipe and a tested `A^T` residual. Prefer retaining both on the SSD.

The measurement consumer uses `J_y`, not `J_x`, because its decision variables
are exact shared-plane integer numerators. State this representation choice in
every manifest and the findings memo. A nonfinite, all-zero, missing-row, or
forward-parity-failing Jacobian is a hard refusal.

## Additive consumer behavior

- Keep all existing zero-band commands and outputs byte/behavior compatible.
- Extend sidecar loading to accept the new manifest/per-pair directory while
  retaining the legacy single-NPZ loader for zero/older tests. Validate pair
  order, per-pair NPZ hash, cache/upstream/model hashes, exact dtype/shape,
  receiver arithmetic, and active arrangement.
- Extend `derive_hyperplane_channel_band` additively with an optional measured
  `local_lipschitz_field`; the legacy scalar path is unchanged. With the field,
  use the stored `q` and `Lip_local` decomposition. This remains a proposal aid;
  only the frozen hard oracle admits a candidate.
- Add `--pose-jacobian-sidecar` and `--tau-pose`. Preserve the old
  `--pose-rgb-band=0` control and continue refusing a positive uncustodied
  scalar pose band.
- For each repair attempt, solve frame1 from the Seg band first. Compute its
  scorer-plane delta. Use `J_y` and the frame0 predictor direction to derive the
  largest deterministic scalar step toward the generated predictor whose
  first-order predicted pose MSE is within `tau_pose`; turn that step into an
  anisotropic frame0 scorer-plane box along the absolute predictor delta, solve
  it with the unchanged lattice solver, record predicted pose6/MSE, then run the
  frozen hard oracle. The linear model is proposal-only. If no step is feasible
  or the hard oracle fails, shrink/retry. A final failure emits a durable
  pair/operating-point refusal with verdict scope limited to that instance.
- Receipt rows must include Seg band admit/reject fractions, zero-Lipschitz
  fraction, channel-radius summaries, pose selected step, predicted-vs-real
  pose residual, hard-oracle repair count, both binding-map hashes/counts,
  actual Brotli-Q11/zstd-19 bytes, and byte attribution. Composition groups by
  `(seg_band_scale,tau_pose)` and reruns the existing measured waterfill; emit a
  KKT candidate only when measured adjacent secants exist, otherwise the exact
  `INCONCLUSIVE_FLAT_OR_NOISY` instance-scoped verdict.

## Owned implementation files

Expected new files:

- `src/tac/optimization/vjp_custody.py`
- `tools/produce_vjp_custody.py`
- `src/tac/tests/test_vjp_custody.py`

Expected additive edits only:

- `src/tac/optimization/joint_seg_pose_rate.py`
- `tools/measure_joint_seg_pose_rate.py`
- `src/tac/tests/test_joint_seg_pose_rate.py`

Do not touch any `upstream/` file, the sacred result tree, training code, live
run state, score/frontier pointers, autoconfig/DSL/provenance hot surfaces, or
unrelated ledgers. Do not commit; the parent Codex session owns review,
serializer, and branch commit discipline.

## Acceptance tests before real production

1. `python3 -m py_compile src/tac/optimization/vjp_custody.py tools/produce_vjp_custody.py src/tac/optimization/joint_seg_pose_rate.py tools/measure_joint_seg_pose_rate.py`
2. `.venv/bin/python -m pytest -q src/tac/tests/test_vjp_custody.py src/tac/tests/test_joint_seg_pose_rate.py`
3. `.venv/bin/ruff check src/tac/optimization/vjp_custody.py tools/produce_vjp_custody.py src/tac/optimization/joint_seg_pose_rate.py tools/measure_joint_seg_pose_rate.py src/tac/tests/test_vjp_custody.py src/tac/tests/test_joint_seg_pose_rate.py`
4. CLI help for both tools succeeds without loading the 4.7 GiB cache.
5. A one-pair real producer smoke writes exactly one immutable SSD sidecar and
   manifest, passes source/hash/shape/unit-normal/YUV-forward/`A^T` checks, and
   resumes without rewriting the sidecar.
6. A zero-band one-pair measurement regression remains green without any new
   sidecar flags.
7. A one-pair positive-band measurement consumes the manifest, records both
   linear predictions and frozen hard-oracle results, and either admits or
   produces an honest durable instance-scoped refusal.

The parent session will inspect the detached implementation diff before any
real 24-pair production, run the acceptance suite, perform the requested
round-1 adversarial review, and write the final dated findings memo and MAIN
landing handoff.

## Operator reframe consumed at 2026-07-19T06:27:52Z

The later per-arm inbox directive supersedes a tiny-only Pose sweep. Preserve
the already-running `tau_pose={1e-8,1e-7}` rows as calibration, but make the
decision surface include `tau_pose={1e-4,2.5e-4}` at both positive Seg scales
on the same 24 real pairs. The pre-registered hypothesis is: the Pose constraint
is inactive/slack at accepted source-centered Seg-band solutions below the
approximately `2.5e-4` crossover. Test this per pair with both the linear
proposal step limit and the frozen hard-oracle `d_pose/tau_pose` ratio; report
confirm/refute counts without promoting the hypothesis beyond this instance.

Every newly measured accepted pair also writes a content-hashed, immutable
compressed bindingness sidecar containing the full frame-0/frame-1 interval
binding maps, the positive Seg-radius map, and exact-source fallback maps.
Receipt rows bind those bytes and state the pose constraint activity criterion.
The operating point is explicitly rung E: the joint inverse solver chooses
exact reachable scorer-plane numerators for both frames and measures their
residual against the declared generated free predictor. This is the first
non-toy payload-coordinate point, but it is still not a receiver/archive or
contest-score claim.
