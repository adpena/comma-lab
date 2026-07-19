# Joint SegNet/PoseNet inverse solve and range-rate telemetry

**Date:** 2026-07-19 UTC  
**Axis:** `[macOS-arm64 CPU advisory subset] NON-PROMOTABLE`  
**Authority:** build + local measurement only; no launch, contest score, promotion, or pointer authority.  
**Pointer:** `0.1910828242 [contest-CPU]` **UNMOVED**.  
**Verdict scope:** 24 selected, unique pairs from the real frozen `gt_n600.npz` cache; native-float32
CPU-Torch SegNet/PoseNet hard oracle. No contest-Linux/CUDA or receiver-archive claim.

## Outcome

The additive solver and bounded measurement runner are built. The clean n24 zero-band control is
hard-oracle exact for SegNet (`mean/max d_seg = 0`) and retains effectively exact PoseNet output
(`mean d_pose = 5.351929655623205e-10`, `max = 2.039703258146138e-9`). The counted range-coordinate
description is still rate-dead at this operating point: actual Brotli-Q11 averages
`2,337,608.42 bytes/pair` for both frames (`1,180,983.63` frame0 + `1,156,624.79` frame1).

Only one admissible operating point was measured. The composed waterfill verdict is therefore
`INCONCLUSIVE_FLAT_OR_NOISY` at **instance scope**. No KKT allocation is forced. The analytic pose/Seg
marginal crossover `d_pose = 2.5e-4` remains **DERIVED**, not selected by these data.

## Score-native formulation

Let `A` be the one shared bilinear camera-to-scorer resize. The joint variables are camera uint8
frames `(z0,z1)` and the only counted description is their selected scorer-numerator coordinate in
`range(A)`. `ker(A)` is never serialized: decoder-side generated fill plus deterministic integer repair
supplies that complement at zero payload bytes.

The declared free predictor is a piecewise-constant camera fill computed only from the counted scorer
plane description. It never reads a hidden/source camera frame. The measured payload is the signed
little-endian int32 residual between selected and predictor scorer numerators, compressed with real
Brotli-Q11 and zstd-19. Camera-frame residual bytes are deliberately not attributed because doing so
would charge deterministic `ker(A)` fill and repeat the old min-norm rate mistake.

For an active SegNet winner/rival cell `(c,c')`, the implemented anisotropic proposal law is

`d_feat = scale * margin / ||Delta w_cc'||`,

`r_channel = min(r_max, d_feat / (3 * Lip_local * |q_channel|))`,

where `q` is the real scorer VJP pullback of the unit frozen-head hyperplane normal. The channel box is
a conservative inner approximation of the pullback slab by the triangle inequality. This consumes the
registered `segnet_head_rank4_linear_flipdist_v1` law; it does not replace it with an isotropic RGB ball.
Positive isotropic bands fail closed. A positive operating point needs a custodied winner/rival VJP
sidecar; none was available in this task's frozen inputs, so the real measurement stayed at the zero-band
control. Pose remains a global six-dimensional hard-oracle constraint; a positive pose band likewise
requires a custodied PoseNet-6 Jacobian/proposer and is not fabricated here.

## Telemetry and custody

The receipt schema records:

- actual range-coordinate Brotli-Q11 and zstd-19 bytes, per frame;
- additive independently-compressed tile attribution in arrangement coordinates:
  winner cell, winner/rival hyperplane, boundary codimension-1 versus cell interior;
- binding-map hashes/counts, integer-repair counts, and chosen-numerator hashes;
- bound/search/repair, frozen-oracle, and codec/tile wall times;
- write-once per-pair reconstruction manifests plus atomic resumable state;
- receiver arithmetic declaration: native float32 with CPU-Torch conv/eval and native argmax tie policy;
- sacred-tree before/after equality, false score/promotion flags, and unchanged pointer.

Across n24, the mean per-pair timing was **MEASURED** as 24.61 s integer search/repair, 0.58 s frozen
oracle, 15.64 s compression/tile telemetry, and 43.16 s total. Mean repaired blocks were 546,188.58 for
frame0 and 540,774.08 for frame1. At zero band all 14,155,776 frame1 channel-block constraints were the
degenerate equality control; no positive-band bindingness claim follows.

## Receipts

- Chunk A, 12 pairs: `.omx/research/joint_seg_pose_inverse_solve_receipt_chunkA_20260719.json`,
  SHA-256 `79e7bd59626641181a453d3a313d5a66bd38f6caa797fe73901a3d43589972ae`.
- Chunk B2, 12 disjoint pairs: `.omx/research/joint_seg_pose_inverse_solve_receipt_chunkB2_20260719.json`,
  SHA-256 `edc3ee50e0b0e618cb21c058e17291aa30a4dffe3f766f58664a63d05e1fb24b`.
- Composed n24: `.omx/research/joint_seg_pose_inverse_solve_receipt_n24_20260719.json`,
  SHA-256 `7a6fdbdfb8f6084a6fd79bb0a63490335b22ae308774032fff7471bb4281e3e9`.
- Pair-125 refusal: `.omx/research/joint_seg_pose_inverse_solve_pair125_diagnostic2_stages/pair_0125.hard_oracle_refusal.json`,
  SHA-256 `d92372509e0f5915bb93ac8508d4736cc3e82a419781c186f805d59b33280969`.

The receipts bind the exact executed tool bytes. Later fail-closed diagnostics changed the working tool,
so the executed `.txt` snapshots are preserved at `.omx/research/source_snapshots/`; their hashes exactly equal
the receipt `config.tool_sha256` values (`cf7dc55a...` for A and `41748c84...` for B2).

## Round-1 adversarial self-review

1. **Can the margin aid admit a hard-oracle rejection? Yes.** It is only a proposal. Every admitted row
   passes native frozen SegNet and PoseNet; failure is durable and instance-scoped.
2. **Is the predictor genuinely free?** Yes under the declared representation: it is a deterministic
   function of the counted scorer-plane description and fixed decoder code. No video-derived table is
   hidden in code, and no camera source crosses the solver boundary.
3. **Does integer repair sneak `ker(A)` bytes back in?** No payload stores camera residuals. The receiver
   repeats the deterministic repair; only range-coordinate numerators are counted. Candidate frame hashes
   remain custody checks, not payload.
4. **Is exact rational `A` equality sufficient under declared arithmetic? No.** Pair 125 produced one
   native CPU-Torch argmax mismatch (`d_seg=5.086263020833333e-6`) despite `d_pose=3.917505209116712e-11`.
   Source control was exactly zero on both metrics and cached/native winner disagreement was zero. This is
   a measured arithmetic-sensitive single-cell refusal, not a family verdict; it demonstrates why final
   native-float32 hard-oracle repair is mandatory.
5. **Is the requested waterfill solved?** The solver is implemented for measured adjacent secants, including
   `5/sqrt(10d)` and the `2.5e-4` crossover, but the evidence has one admissible point. The correct measured
   answer is inconclusive. Positive curves remain blocked on real winner/rival and PoseNet pullback custody.

## MAIN landing review required

MAIN must review the additive diff, especially (a) the channel-inner-box interpretation of the hyperplane
pullback, (b) range-only payload packing and cell attribution, (c) native float32 source-control behavior,
and (d) the honest no-waterfill verdict. Existing solver semantics and `upstream/` were not modified.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; delegated task #549 authority; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; `.omx/research/segnet_recursive_fractal_factorization_20260715.md`;
`.omx/research/frozen_scorer_exact_factorization_20260715.md`;
`.omx/research/v10_lattice_rate_verdict_and_composition_20260719.md`;
real `gt_n600.npz`; per-arm and broadcast inboxes.
