# ADVISORY restart handoff — v7.5.2 / v7.5.3 / v8 fresh-eyes sweep — 2026-07-10

## Purpose

Durable no-signal-loss handoff from the pre-`multi_agent_v2` Codex task into a freshly started task
that will reload `/Users/adpena/.codex/config.toml` and continue the same advisory campaign.

The operator explicitly requires model/reasoning calibration for every delegated lane. The restarted
primary must use **gpt-5.6-sol / ultra**. It must choose subagent model and reasoning by difficulty:

- deep scorer reverse engineering, mathematical architecture, and cross-vehicle synthesis:
  frontier model at `ultra`/maximum supported reasoning;
- bounded code/spec/custody audits: balanced capable model at `xhigh` or higher;
- mechanical link/test/hash checks only: faster model at high reasoning, never for conclusions.

The old collaboration tool did not expose model selection; that limitation was disclosed. The new
config requests visible spawn metadata and the `agents` namespace:

```toml
[features.multi_agent_v2]
hide_spawn_agent_metadata = false
tool_namespace = "agents"
```

It was TOML-parse-verified at `/Users/adpena/.codex/config.toml:96-98`.

## Durable completed advisories

These contain the complete first-pass findings, exact code/spec locations, falsifiers, and proof
matrices. Read them before spawning or re-deriving anything:

1. `.omx/research/t5_crucible2/ADVISORY_v752_fresh_eyes_20260710.md`
2. `.omx/research/t5_crucible2/ADVISORY_v753_texture_trunk_fresh_eyes_20260710.md`
3. `.omx/research/t5_crucible3/ADVISORY_v8_fresh_eyes_20260710.md`

All are untracked at handoff time and must be preserved. Do not absorb unrelated dirty state.

## First-pass dispositions

- **v7.5.2:** HOLD full launch and hold/rename the pilot until the exact intended program is
  instantiated. P0: #383 absent from launch, epoch-726 bypass, R1 fallback is telemetry rather than
  artifact selection and is not proven graft-composable, inherited chroma violates the clean rung,
  amber unresolved. P1: beta endpoint, taper after self-orient removal, pose-mode resume protection,
  dual defaults, dry-start semantic incompleteness.
- **v7.5.3/texture:** DESIGN/BUILD-ONLY. P0: MLX trains T while NumPy verdict/inflate omit it; byte
  close counts a 430,878-byte deterministic bank then ignores it; current RGB/all-frame T does not
  implement the frame1 luma-null exact-D home law. P1: missing event/optimizer state, no matched A2,
  palette/bias gauge, coupled G gradients, no xi advection, overstated [4,8] theorem, annulus/resume.
- **v8:** HOLD training EVENT. P0: reversed kill inequality, receipt-less naked-dseg gate, declarative
  rather than executable byte closure, unresolved E-edge versus K-class architecture/integrability.
  P1: shared-trunk theft persists, per-class/tie-flicker gates are prose, carrier byte claims are not
  receiver sections, hardcoded class IDs, mixed rate generations, undefined seed statistics, power
  diagram terminology overclaim.

## Live evidence that must not be lost

The preserved owed16v2 arm reported at epochs 669 and 673:

```text
classification=DEGENERATE_GUARD_TRIPPED
should_ship_banked_r1=true
actuated=false
```

in `experiments/results/owed16v2_rebalanced_ON_20260710T114759Z/safe_run.out`. This is direct evidence
that the banked fallback is not consumed. The process/checkpoints must not be stopped or mutated by
the advisory task.

## Frozen evaluator/video custody

The local source was found before any download. The clean SSD checkout at
`/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709` is at git
`991b317c41fe3aac657e0f0cb88fd831b2e4185a`, clean, and byte-identical on the inspected scorer files
and video to the working `upstream/` copies (the latter differ from their old git only by executable
mode on scorer files).

SHA-256:

- `upstream/evaluate.py`: `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`
- `upstream/modules.py`: `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`
- `upstream/frame_utils.py`: `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90`
- `upstream/videos/0.mkv`: `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
- `upstream/models/segnet.safetensors`:
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`
- `upstream/models/posenet.safetensors`:
  `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`

Video: HEVC Main, yuv420p TV range, 1164×874, 20 fps, 60 s, 1200 frames, 37,545,489 B, exactly
600 non-overlapping pairs.

Exact scorer semantics:

- SegNet uses only frame1, bilinear-resized to 512×384, and scores hard argmax disagreement.
- PoseNet uses both frames. Each is bilinear-resized to 512×384, converted to YUV6: four luma
  polyphase channels plus 2×2-averaged U and V at 256×192. The two frames become 12 channels.
- Pose head outputs 12 values but distortion compares only the first six, using MSE.
- Score is `100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489`.

Exact marginal arithmetic to preserve:

- one corrected Seg output pixel in one of 600 frames is worth
  `100/(600*384*512) = 8.477105e-7` score;
- one archive byte costs `25/37,545,489 = 6.658589531e-7` score;
- break-even is about **0.785 Seg cells saved per added byte**, or **1 corrected Seg cell buys about
  1.273 bytes**, absent pose effects.

## Exact camera-to-scorer sampling geometry

Both networks use the same PyTorch bilinear 874x1164 to 384x512 resize with default
`align_corners=False`, `antialias=False` (`modules.py:73,109`). Adjacent output coordinates are about
2.276/2.273 input pixels apart, so their two-tap one-dimensional supports never overlap. The map
uses exactly 768 input rows and 1,024 input columns: 786,432 of 1,017,336 camera pixels. The remaining
230,904 pixels per frame (22.6969%; 692,712 RGB coordinates) are exact joint scorer-preprocess blind
coordinates. The full RGB resize is surjective rank 589,824 with kernel dimension 2,462,184 per
frame.

This makes a direct camera-grid preimage block-separable: each scorer RGB sample is one weighted sum
of its own disjoint 2x2 camera footprint. A future exact-D texture carrier can solve independent
bounded integer 4-to-1 lifts rather than assume a global bicubic inverse. Quantization/range can
still make a desired target unreachable, so fresh-raw preprocessing equality remains the gate.
Unsampled pixels may be filled only by a deterministic generic rule, never with hidden video-derived
data.

Evaluator completeness caveat: `TensorVideoDataset` floors raw length to whole frames and
`zip(dl_gt, dl_comp)` truncates to the shorter iterator. The existing STRICT compliance guard must
therefore require `0.raw` exactly `1164*874*1200*3 = 3,662,409,600` bytes / 1,200 frames before
scoring. A short raw is a NO-FAKE failure, not a rate lever.

## Measured GT/video geometry from the frozen cache

Cache: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, n600, 384×512 labels.

Class occupancy over 117,964,800 scored cells:

- Road 23.2332%
- Lane 0.58546%
- Undrivable 49.5176%
- Movable 1.23793%
- MyCar 25.4258%

Mean consecutive scored-frame label turnover (10 Hz odd frames): 1.24564%. A streaming remeasurement
after restart distinguishes two boundary conventions: one-sided forward-stencil incidence (a cell
differs from its right or down neighbor) is 1.2388865%; symmetric four-neighbor boundary support
(either endpoint marked) is 2.1628333%. The earlier 1.23911% value was the former but was mislabeled
as the latter. Both support static bulk plus sparse moving/boundary treatments; neither proves
scorer-Jacobian locality.

Counts are 1,461,450 one-sided source cells versus 2,551,382 symmetric boundary cells. The symmetric
count is independently preserved in
`.omx/research/segnet_fragile_support_codec_budget_20260609.json:29-41`. Do not size an undirected
edge carrier from the one-sided statistic: it is class-biased (for example MyCar 6,026 / 0.0201%
one-sided versus 307,901 / 1.0266% symmetric).

Margin facts:

- only 2.6702% of cells have top1-top2 margin < 1 and 4.8332% have margin < 2;
- Lane is fragile: 74.31% < 1; Movable 17.69%; Road 5.63%; Undrivable 0.85%; MyCar 1.14%.

RAG adjacency counts (unordered, horizontal+vertical):

- Road-Lane 814,066
- Road-MyCar 317,679
- Road-Undrivable 290,167
- Undrivable-Movable 99,530
- Road-Movable 90,322
- Lane-MyCar 5,112
- Lane-Undrivable 1,587
- Lane-Movable 1,297
- Movable-MyCar 157
- Undrivable-MyCar absent

There are 6,703 2×2 triple-or-higher junction blocks, dominated by Road-Undrivable-Movable (3,344),
Road-Lane-Movable (1,348), Road-Lane-MyCar (1,114), and Road-Lane-Undrivable (875). Any E-edge v8
must enforce antisymmetry and cycle/cocycle integrability at these junctions or solve for global class
potentials after quantize/parse-back.

## Scorer Jacobian signal recovered as summary evidence

The v8 scorer cross-check reported frozen-model local Jacobians on canonical decoded inputs:

- Pose, six spread pairs: gradient energy frame0 54.37%, frame1 45.63%; luma 95.97%, chroma 4.03%.
  Singular values approximately
  `[3.371e-4, 6.590e-4, 8.305e-4, 1.804e-3, 2.341e-3, 1.14673e-1]`, condition about 340.
  This is INSTANCE/local signal supporting a luma-dominated frame0 actuator, not a structural proof
  that chroma is Pose-null.
- Seg, minimum-margin Lane-vs-Road cell: gradient nonzero at 100% of 384×512 input; energy outside
  radii 64/128/192 px = 9.15%/3.56%/2.21%. A high-margin Road-vs-MyCar cell still has
  5.05%/1.97%/0.81% outside those radii. EfficientNet-B2 SqueezeExcite global pooling plus the deep
  U-Net makes formal dependence global.

Therefore v8's pairwise tie-locus decomposition is an empirical error representation, not a
factorization of the SegNet response. Add a remote-patch/Jacobian block-norm falsifier
`J(edge e <- paint region e')` before claiming independent edge optimization or byte allocation.

Restart validation found no raw Jacobian artifact or exact reproduction command for these headline
numbers in the workspace. Treat them as recovered, instance-level summary evidence until rerun into
a hashed receipt. The source-level global dependency through EfficientNet-B2 SqueezeExcite remains
independently inspectable; the numerical tail percentages and Pose spectrum do not yet carry the
same reproducibility authority.

## Exact Pose-preprocess nullspace to formalize

At the 512×384 scorer grid and away from clamp/uint8 discontinuities, require per-pixel

```text
0.299*dR + 0.587*dG + 0.114*dB = 0
```

and per 2×2 block require

```text
sum(dR) = 0
sum(dB) = 0.
```

Then each of the four Pose Y polyphase channels is unchanged and the averaged V/U channels are
unchanged. Equivalently choose zero-sum dR and dB patterns (3+3 degrees of freedom per 2×2 block) and
derive `dG = -(0.299*dR + 0.114*dB)/0.587`. This gives a six-dimensional linear nullspace per block
before resize/round/clamp. v7.5.3 should parameterize this space, then solve its camera-grid preimage
through the exact bilinear/uint8/R operator and verify the first six Pose outputs bit-stable. Generic
“chroma HF” is weaker.

The flushed texture reviewer supplied a concrete basis. With luma-null pure-chroma directions

```text
c_U = (0, -0.3441362862, 1.772)
c_V = (1.402, -0.7141362862, 0)
```

and zero-sum 2×2 Haar patterns

```text
h_x  = [[ 1,-1],[ 1,-1]]
h_y  = [[ 1, 1],[-1,-1]]
h_xy = [[ 1,-1],[-1, 1]],
```

the six atoms `h_k tensor c_U` and `h_k tensor c_V` span the local kernel: 6 degrees per block,
294,912 per frame at 512×384. Important implementation blockers:

- current T is pre-sigmoid; unequal channel sigmoid derivatives break RGB luma-nullness;
- pixelwise soft class placement and annulus gates break each atom's 2×2 zero-sum condition;
- renderer-grid nullness can leak through bicubic lift, uint8, and evaluator bilinear downsample;
- U/V and RGB clamping make the kernel piecewise and require active-set/amplitude guards.

Compose placement first, project after sigmoid, and verify after exact R/fresh `.raw` reload. The
six Haar/chroma atoms are a precise scorer-preprocess family, not proof of the current generic
period-[4,8] bank.

## Late arithmetic/custody corrections to preserve

The v7.5.2 scorer reviewer identified two stale sentences in
`.omx/research/r1_dxi_shippability_byteclose_20260708.md` that must not propagate:

- `sqrt(10*0.018) = 0.424`, not approximately 0.02. The old sentence likely confused d_pose
  `3.4e-5` with an approximately 0.018 contribution.
- `0.022 / 0.00161` is about 13.7, not 1000; moreover the 0.022 value is n24, so it is not an n600
  matched ratio. The earlier same-axis n24 approximately-20x statement is the defensible one.

These corrections do not revoke #238 shippability; they narrow two follow-on arithmetic claims.

## Public frontier/apparatus findings

- `.omx/state/canonical_frontier_pointer.json` still points locally to
  0.19109982419209975 `[contest-CPU]`, archive sha b4689726..., and must remain unchanged.
- Its upstream watcher is wrong and stale: `src/tac/canonical_frontier_pointer.py:477-485` queries
  `commaai/commavq`, not `commaai/comma_video_compression_challenge`; the cached snapshot is from
  2026-06-07. `reports/latest.md` is also stale at 0.1919853363.
- Official merged PR112 reports 0.191126. Open PR128 now claims 0.187991 after exact CPU-axis latent
  coordinate descent and deterministic byte-exact rebuilding. Treat it as unratified public signal,
  not authority: `https://github.com/commaai/comma_video_compression_challenge/pull/128`.
- Current v7.5.2/v7.5.3/v8 work lacks visible registered lane IDs in the canonical lane/dispatch
  registries even though subagent-progress entries exist. This defeats deduplication/maturity views.

## Work remaining in the restarted task

1. Re-read the three durable vehicle advisories; do not re-derive settled first-pass findings.
2. Spawn calibrated scorer/vehicle reviewers with visible model/reasoning metadata.
3. Finish `.omx/research/ADVISORY_evaluator_video_geometry_20260710.md`.
4. Append scorer-derived additions to the three vehicle advisories without weakening their current
   evidence labels.
5. Write `.omx/research/ADVISORY_vehicle_line_synthesis_20260710.md`, including apparatus/public
   frontier repairs and the recommended action order.
6. Validate Markdown, links, exact math, dirty-tree preservation, and lane checkpoint. Commit only
   the new ADVISORY files through the canonical serializer if safe; do not stage shared dirty state.

## Dirty/shared state at handoff

At the last check, `main == origin/main`. Pre-existing/shared changes:

```text
M .omx/state/harness_failure_ledger.jsonl
M .omx/state/lane_maturity_audit.log
M .omx/state/lane_registry.json
?? paper/__marimo__/
```

The advisory lane registration also touched the already-dirty registry/audit files. Preserve all of
them and stage none of them. The only owned files are the `ADVISORY_*.md` documents created by this
campaign.

**Pointer delta:** none. **Launches:** none. **Runs stopped:** none.
