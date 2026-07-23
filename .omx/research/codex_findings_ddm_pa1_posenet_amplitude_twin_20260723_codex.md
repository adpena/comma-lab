# Codex findings — DDM PA1 PoseNet amplitude twin — 2026-07-23

`research_only=true` · `score_claim=false` · pointer unchanged ·
MAIN landing review required.

## Verdict

**PASS frame-0 PoseNet amplitude mechanism on the local advisory axis; REJECT
joint-frame placement; BLOCK promotion pending governed inflate composition.**

The source-bound E2 independent control is
`d_pose=162.580958694146`, `d_seg=0.02861480712890625`, 343,466 archive bytes.
The scorer-only first-stem-BN target on frame 0 reaches
`d_pose=147.49104204339514` with the exact same d_seg and zero incremental
payload bytes. Its joint score delta is `-1.9167666862136272`.

This is statistics matching, not pose-value encoding. It does not repair E2's
absent compact Pose6 stream or claim a code-to-photometry inverse. The
post-hoc-stored-corrections-dead law therefore remains binding.

## Step A — measured statistics and falsifier

Across all 600 E2 pairs through literal upstream resize + `rgb_to_yuv6`:

- standardized candidate-vs-GT mean-gap RMS: `0.20144544838533338`;
- log-standard-deviation-ratio RMS: `0.28378726561837414`;
- pre-registered equivalence margin: `0.1` on both coordinates.

The amplitude-gap-small falsifier did not fire. The first-stem scorer-only
inverse used both AT1x BN tables and a bounded variance solve. Weighted mean
residual RMS is `0.0026917954616931606`; relative variance residual RMS is
`0.8127839199072302` at solver optimality `1.6575489505404517e-11`.
The atlas still has zero direct amplitude factors; no direct-input-BN authority
is fabricated.

## Steps B/C — receiver-realized arms

| Arm | Target partition | Δ bytes | d_pose | d_seg | Δ pose term | Δ Seg term | Joint ΔS |
|---|---|---:|---:|---:|---:|---:|---:|
| frame0 GT-stat | COUNTED | 24 | 162.423102394524 | 0.028614807129 | -0.019579542475 | 0 | -0.019563561860 |
| frame0 scorer-stat | FREE candidate | 0 | 147.491042043395 | 0.028614807129 | -1.916766686214 | 0 | -1.916766686214 |
| joint scorer-stat | FREE candidate | 0 | 161.726887970608 | 0.035720909966 | -0.106047512370 | +0.710610283746 | +0.604562771375 |

Both frame-0 arms changed exactly zero frame-1 channel values and preserved
3,375,540 / 117,964,800 Seg errors. The joint rung added 838,270 Seg errors and
is rejected despite its pose improvement.

The scorer target is not approximately equal to the GT target under the
pre-registered `0.01 d_pose` margin; it is better by
`14.932060351129309 d_pose`. The zero-byte row therefore stands on its own
measured effect rather than an equality inference.

## FREE / NULL / COUNTED custody

- GT-video-derived target moments are COUNTED: 24 fp16 bytes for frame 0 or 48
  fp16 bytes for both frames.
- Scorer-only target constants are a rule-118 FREE candidate: they derive only
  from frozen convolution weights and BN running statistics. The dynamic
  affine derives from already-counted decoded E2 frames.
- Receiver-inert targets would be NULL.
- The historical Seg-side 30-byte row follows the same law: GT-derived target
  facts remain COUNTED; scorer-only expected-stat constants can be FREE only
  after receiver survival.

The zero-byte arm has not yet been composed into the governed E2 `inflate.py`.
That exact runtime composition and archive parse-back is the promotion blocker;
this memo does not promote or mutate the frontier.

## Independent R1 adversarial review

Disposition: **PASS evidence; BLOCK promotion.**

- Re-aggregated all 38 batch checkpoints per arm. Error counts, sites, pose
  SSE, coordinates, d_seg, and d_pose close to each immutable stage receipt.
- Verified literal upstream `frame_utils.rgb_to_yuv6` is used, not the
  differentiable training mirror.
- Verified source raw, E2 archive, target cache, upstream sources, and scorer
  weights by SHA-256 before measurement.
- Verified frame-0 placement leaves frame 1 byte-identical and therefore cannot
  hide Seg collateral.
- Rejected the joint arm using the full action, not its pose headline.
- Scoped the remaining gap correctly: no contest-CPU/CUDA evidence, no exact
  archive composition, no frontier motion.

## Review and execution receipts

- Immutable n600 batch checkpoints: 38 statistics + 38 per each of three arms.
- Independent re-aggregation: PASS.
- Ruff: PASS. Python compile: PASS.
- Focused tests: 6 passed. AT1x materialization tests: 15 passed.
- Review tracker: 3 consecutive reviewed marks for all 42 entities across the
  runner and canonical-equations module.
- Clean pass 3 independently re-aggregated all arms and rechecked the authority,
  composition-blocker, frame-ownership, and score-direction guards.
- Superseded pre-measurement bounded-solver trial was losslessly moved to
  `/Volumes/VertigoDataTier/pact/evidence/ddm_pa1_posenet_amplitude_twin_20260723T221923Z/superseded_unbounded_variance_clip`;
  no evidence was deleted.

Canonical machine receipt:
`.omx/research/ddm_pa1_posenet_amplitude_twin_20260723T221923Z/ddm_pa1_posenet_amplitude_twin_receipt.json`.

MAIN landing review is required.
