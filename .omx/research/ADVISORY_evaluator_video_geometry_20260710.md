# ADVISORY — frozen evaluator and video geometry — 2026-07-10

`research_only=true`

**Disposition:** evaluator evidence supports narrower, cheaper formulations for all three vehicles,
but authorizes **no training EVENT, dispatch, pointer move, or archive promotion**. The scorer is
asymmetric across frames and channels, and its SegNet dependence is empirically global. Any vehicle
that treats all pixels, frames, or edges as exchangeable is solving a stronger and more expensive
problem than the contest asks; any vehicle that treats edge-local Seg behavior as an exact
factorization is claiming more locality than the scorer supplies.

This is a read-only advisory artifact. The frozen scorer, video, GT cache, live owed16v2 run, and
canonical pointer were not mutated.

## Answer first — the obligation matrix

The exact receiver path imposes three different obligations:

| Surface | frame0 | frame1 | Authority |
|---|---|---|---|
| SegNet | structurally absent | hard 5-class argmax at 384x512 | exact scorer source |
| PoseNet | four luma polyphases + averaged U/V | four luma polyphases + averaged U/V | exact scorer source |
| Rate | counted once in `archive.zip` | counted once in `archive.zip` | exact archive bytes |

Therefore frame0 is **Seg-free but not Pose-free**. Frame1 carries both obligations. Pose is strongly
luma-dominated at the measured instances, but generic chroma or generic high frequency is not a
structural Pose-null space: all four luma polyphase samples survive. The exact local linear kernel
must be luma-null per pixel and zero-sum in both chroma coordinates per 2x2 scorer-grid block, then
must survive camera-grid lifting, uint8, clamp, evaluator resize, and fresh `.raw` reload.

Seg errors are sparse and boundary-heavy in the frozen video, which motivates sparse treatments.
That video geometry does **not** establish a SegNet edge-local factorization. EfficientNet-B2
SqueezeExcite global pooling and the deep U-Net create source-inspectable global dependency paths.
A recovered summary reports nonzero individual-margin gradients across the full image, but its raw
receipt is missing; v8 edge carriers and v7.5.3 local texture gates remain empirical representations
whose cross-region effects must be reproduced and measured.

## Frozen custody receipt

The local source was found before any download. The authority checkout is
`/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709`, clean at git
`991b317c41fe3aac657e0f0cb88fd831b2e4185a` on its historical `master` branch. The inspected files
are byte-identical to the working `upstream/` copies; only executable-mode metadata differs from the
working tree's older git state.

| Object | SHA-256 |
|---|---|
| `evaluate.py` | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` |
| `modules.py` | `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa` |
| `frame_utils.py` | `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90` |
| `videos/0.mkv` | `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` |
| `models/segnet.safetensors` | `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` |
| `models/posenet.safetensors` | `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576` |
| `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` | `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` |

The GT cache is 5,078,017,610 bytes and was consulted in place. It was not copied, moved, or
regenerated.

## Exact evaluator DAG

1. `evaluate.py:55-69` reads the one-name public list, loads the source video, charges the exact
   `archive.zip` size, and divides by the total bytes of files in `videos/`. This frozen checkout has
   only `videos/0.mkv`, so the denominator is exactly 37,545,489 bytes.
2. `frame_utils.py:10-13,185-216,218-253` forms non-overlapping two-frame sequences and reads the
   submission's inflated camera-resolution `.raw` as uint8 RGB with shape
   `(B,2,874,1164,3)`.
3. `modules.py:143-158` converts to float RGB and sends the same decoded pair to PoseNet and SegNet.
4. `modules.py:70-84` bilinear-resizes both frames to 384x512, applies `rgb_to_yuv6`, concatenates
   two 6-channel frames into 12 channels, produces 12 pose-head values, and scores MSE on the first
   six.
5. `modules.py:103-113` selects only the last frame, bilinear-resizes it to 384x512, produces five
   logits, and scores hard argmax disagreement.
6. `evaluate.py:71-100` averages both distortions over pairs and computes

```text
S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37,545,489.
```

CPU and CUDA remain separate authority axes. The source read is reproducible; the recovered local
Jacobian headlines currently are summary-only advisory evidence. Neither manufactures a contest
score or permits inferring one axis from another.

## Frozen video geometry

`ffprobe -count_frames` and file stat independently confirm:

- HEVC Main, yuv420p, TV range;
- 1164x874, 20 fps;
- 1,200 decoded frames, hence 60 seconds derived at 20 fps;
- 37,545,489 bytes;
- exactly 600 non-overlapping evaluator pairs.

The scored Seg surface therefore has

```text
600 * 384 * 512 = 117,964,800 cells.
```

One corrected Seg cell in one scored frame changes the total score by exactly

```text
100 / 117,964,800 = 8.477105034722222e-7.
```

One added archive byte costs exactly

```text
25 / 37,545,489 = 6.658589531221713e-7.
```

Thus one added byte must save more than `0.7854791823` Seg cells, absent Pose effects; one Seg cell
buys `1.2731082153` bytes. The earlier handoff spelling `6.658587e-7` was a rounding typo and is
corrected here without changing the break-even conclusion.

Pose economics are operating-point dependent:

```text
d/d(d_pose) sqrt(10*d_pose) = sqrt(10) / (2*sqrt(d_pose)).
```

No fixed "Pose units per byte" is valid without the control arm's measured `d_pose`. Every payload
admission receipt must report the before/after square-root contribution on exact decoded bytes.

Illustrative local derivatives make the regime change explicit:

| Operating point | `d_pose` | `d sqrt(10*d)/dd` | bytes bought by local `1e-6` reduction |
|---|---:|---:|---:|
| v7.5 run-1 | 1.79 | 1.18179865 | 1.77485 |
| v7.5.2 banked R1 n600 | 0.001610 | 39.4055203 | 59.17998 |
| ancestor/reference only | 0.000034 | 271.163072 | 407.238 |

The ancestor row is non-transferable evidence, not a v7.5 promise. For a finite change, the exact
byte equivalent is

```text
[sqrt(10*d_pose_control) - sqrt(10*d_pose_candidate)] / 6.658589531221713e-7,
```

not the tangent approximation.

## Exact camera-to-scorer sampling geometry

Both networks call the same PyTorch bilinear resize from 874x1164 to 384x512 with default
`align_corners=False`, `antialias=False` (`modules.py:73,109`). For an output coordinate `o`, the
one-dimensional source coordinate is

```text
x(o) = (o + 0.5) * input_size/output_size - 0.5.
```

The height and width steps are 2.2760417 and 2.2734375 input pixels. Both exceed two, so adjacent
outputs' two-tap supports are disjoint. Exact consequences:

| Quantity | Value |
|---|---:|
| input rows used | 768 / 874 |
| input columns used | 1,024 / 1,164 |
| camera pixels entering either scorer resize | 786,432 / 1,017,336 |
| exact unsampled camera pixels per frame | 230,904 (22.6969261%) |
| exact unsampled RGB coordinates per frame | 692,712 |
| RGB input/output dimensions | 3,052,008 -> 589,824 |
| full resize rank / kernel dimension | 589,824 / 2,462,184 |

Rank is full because every output/channel is a nonzero weighted sum over its own disjoint 2x2 input
footprint. The kernel contains both the unsampled raw coordinates and real zero-weighted-sum changes
inside each used footprint. The 692,712 unsampled coordinate axes are exactly blind even for uint8.
The remaining 1,769,472-dimensional within-footprint kernel is a continuous-linear derivation until
bounded integer/float-axis survival is receipted.

The same rank bookkeeping exposes the whole pair's preprocessing burden:

- camera RGB per frame: 3,052,008 dimensions;
- camera-to-Pose-preprocess rank/kernel per frame: 294,912 / 2,757,096;
- jointly observed two-frame pair: frame0 Pose rank 294,912 plus frame1 RGB rank 589,824;
- pair rank/kernel: 884,736 / 5,219,280 out of 6,104,016 input dimensions, a real-linear kernel share
  of 85.5057% before uint8/range constraints.

This dimension count is DERIVED from the exact linear preprocessing graph. It is not a count of
independently reachable integer code symbols.

Receiver consequences:

- fill unsampled raw pixels by a deterministic generic rule rather than spend video-derived payload
  on values no scorer reads;
- target exact sampled footprints rather than treating camera fidelity uniformly;
- solve camera preimages independently per scorer pixel/channel as bounded integer 4-to-1 problems;
- keep a torch/uint8 metamorphic receipt because rounding, range, and implementation parity can make
  a desired float target unreachable.

For v7.5.3, first choose the desired six-dimensional Haar/chroma Pose-null perturbation on each 2x2
scorer block, then lift each constituent scorer RGB sample through its independent camera footprint.
An existing bicubic renderer is an additional model constraint, not an evaluator requirement.

### Strict raw cardinality remains a compliance gate

`frame_utils.py:225-232` floors submission raw length to a whole-frame count, while
`evaluate.py:71-83` zips source and submission iterators and can therefore average a shorter prefix.
The scorer does not itself assert exact raw cardinality. Preserve the existing STRICT contest guard:

```text
0.raw bytes = 1164 * 874 * 1200 * 3 = 3,662,409,600
frame count = 1,200.
```

A short raw is a NO-FAKE/compliance failure and must REFUSE before scoring. It is not a byte or
runtime lever.

## Measured partition geometry — instance facts, not scorer factorization

From `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, with 117,964,800 labels:

| Class | Occupancy | Share with top1-top2 margin below 1 |
|---|---:|---:|
| Road | 23.2332% | 5.63% |
| Lane | 0.58546% | 74.31% |
| Undrivable | 49.5176% | 0.85% |
| Movable | 1.23793% | 17.69% |
| MyCar | 25.4258% | 1.14% |

Across all cells, 3,149,890 / 117,964,800 = 2.67019484% have margin below 1 and
5,701,511 / 117,964,800 = 4.83323076% below 2. Consecutive scored-frame turnover at 10 Hz is
1,466,965 changed cells across 599 transitions, or 1.24563770%. Boundary sparsity depends on the
stated stencil: one-sided
right/down incidence is 1.2388865153%, while symmetric four-neighbor support marking either endpoint
is 2.1628333198%. The earlier 1.23911% value was the former, mislabeled as the latter. Both
measurements support static bulk plus sparse moving/boundary treatments. Lane is rare and
exceptionally fragile, so an aggregate-only d_seg gate is inadequate.

In counts, the two conventions are 1,461,450 versus 2,551,382 cells. The latter is independently
recorded in `.omx/research/segnet_fragile_support_codec_budget_20260609.json:29-41`. The one-sided
stencil is class-biased—MyCar is only 6,026 / 0.0201% one-sided but 307,901 / 1.0266% symmetric—so
v8 must not size or gate undirected carriers from the 1.2389% statistic.

The unordered horizontal-plus-vertical RAG adjacency counts are:

| Edge | Count |
|---|---:|
| Road-Lane | 814,066 |
| Road-MyCar | 317,679 |
| Road-Undrivable | 290,167 |
| Undrivable-Movable | 99,530 |
| Road-Movable | 90,322 |
| Lane-MyCar | 5,112 |
| Lane-Undrivable | 1,587 |
| Lane-Movable | 1,297 |
| Movable-MyCar | 157 |
| Undrivable-MyCar | absent |

There are 6,703 two-by-two triple-or-higher junction blocks, led by
Road-Undrivable-Movable (3,344), Road-Lane-Movable (1,348), Road-Lane-MyCar (1,114), and
Road-Lane-Undrivable (875). Any E-edge v8 formulation must preserve antisymmetry and cycle/cocycle
integrability at these junctions after quantization and parse-back, or reconstruct and render a
globally consistent set of class potentials.

## Pose geometry — what is measured and what is structural

### Recovered local frozen-model Jacobian summary

On six spread canonical decoded pairs, the recovered Pose Jacobian energy was:

- frame0 54.37%, frame1 45.63%;
- luma 95.97%, chroma 4.03%;
- singular values approximately
  `[3.371e-4, 6.590e-4, 8.305e-4, 1.804e-3, 2.341e-3, 1.14673e-1]`, condition about 340.

Workspace-wide restart validation did not find the raw tensor receipt, script invocation, or hashed
artifact underlying these six-pair numbers. They are therefore recovered, local, instance-level
summary evidence—not independently reproducible measurement authority yet. If reproduced, they
would support a luma-dominated frame0 pose actuator and one much stronger pose-output direction.
They do not prove a structural chroma null, a globally fixed low-rank subspace, or
interchangeability of the two frames.

### Exact linear kernel of the Pose preprocessing map

`frame_utils.py:50-78` keeps all four luma samples in every 2x2 scorer-grid block and averages U and
V. Away from clamp and uint8 discontinuities, preserve luma per pixel:

```text
0.299*dR + 0.587*dG + 0.114*dB = 0,
```

and impose, within each 2x2 block,

```text
sum(dR) = 0,
sum(dB) = 0.
```

The luma constraint determines `dG`; the two zero-sum four-pixel fields each have three degrees of
freedom. The local kernel is therefore six-dimensional per 2x2 block. A concrete basis uses

```text
c_U = (0, -0.3441362862010222, 1.772)
c_V = (1.402, -0.7141362862010221, 0)

h_x  = [[ 1,-1],[ 1,-1]]
h_y  = [[ 1, 1],[-1,-1]]
h_xy = [[ 1,-1],[-1, 1]].
```

The six atoms `h_k tensor c_U` and `h_k tensor c_V` leave all four Y polyphases and both averaged
chroma coordinates unchanged in the linear scorer-grid model. With 49,152 blocks, that is 294,912
linear degrees per frame before reachability and active-set constraints. Direct float64 algebra puts
the basis residual below `5.6e-17`.

This is a preprocessing-null family, not proof that v7.5.3's current generic period-[4,8] bank lies
in it. Four implementation facts remain binding:

1. current texture is composed before sigmoid, whose unequal per-channel derivatives destroy RGB
   luma-nullness;
2. soft per-pixel class placement and annulus gates destroy the blockwise zero sums;
3. bicubic camera-grid lift plus evaluator bilinear resize can leak out of the desired scorer-grid
   kernel;
4. uint8 rounding and RGB/YUV clamping make the feasible kernel piecewise.

Compose placement first, project after the final nonlinear RGB construction, solve a reachable
camera-grid preimage through the exact resize operator, guard the active set, and require the first
six Pose outputs to be bit-stable after fresh `.raw` reload. A proxy tensor equality is not enough.

## Seg geometry — sparse errors, global dependence

The recovered summary reports two frozen-model margin Jacobians:

- a minimum-margin Lane-vs-Road output cell had nonzero gradient at 100% of the 384x512 RGB input;
  energy outside radii 64/128/192 pixels was 9.15%/3.56%/2.21%;
- a high-margin Road-vs-MyCar cell still had 5.05%/1.97%/0.81% outside those radii.

Those numerical tails must be reproduced into a hashed receipt before they can serve as measured
gate evidence. Independently, EfficientNet-B2 SqueezeExcite pooling creates a direct global path,
and the deep U-Net adds broad contextual paths. Consequently:

- a local texture or edge patch may move remote logits;
- pairwise tie-locus ownership is an empirical error decomposition, not an exact factorization of
  SegNet;
- per-carrier d_seg gains cannot be added without a decoded composite interaction check.

The minimum falsifier is a block response matrix

```text
J(edge e <- painted or perturbed region e')
```

measured for same-edge, adjacent-edge, and remote-edge blocks, followed by the actual nonlinear
argmax drift after quantize, decode, and R. If off-diagonal mass is material, use a global trust
region or reconcile in class-potential space instead of independent per-edge admission.

## Cross-vehicle consequences

### v7.5.2

- A banked fallback is a complete scorer-facing artifact, not a scalar telemetry choice. Its exact
  checkpoint, decoder state, per-pair codes, archive grammar, and parse-back output must travel
  together; scorer geometry provides no license to graft arbitrary `dxi` onto an incompatible EMA.
- Any follow-on claim using `sqrt(10*0.018) approximately 0.02` is arithmetically invalid: the value
  is 0.424264. Likewise `0.022/0.00161` is 13.6646, and the 0.022 row is n24 rather than n600.
- The recovered Pose-Jacobian summary motivates testing a luma-dominated pose actuator, but its raw
  receipt is owed. It does not excuse inherited chroma in the clean attribution rung or replace the
  missing bank selector.

**Launch disposition remains HOLD.** Scorer evidence narrows the eventual pilot; it does not repair
the missing composed #383 gate, epoch-726 bypass, artifact selector, clean rung, or amber semantics.

### v7.5.3 texture trunk

- Frame0 texture cannot improve Seg by construction; it should be zero unless it earns its bytes on
  Pose. Frame1 texture is jointly Seg- and Pose-active.
- Replace generic RGB texture with an exact-D candidate family: post-placement, post-nonlinearity
  projection into the six-dimensional per-block Pose-preprocess kernel, followed by camera-grid
  preimage and exact raw verification.
- The source-level global path and recovered Seg-Jacobian summary require a reproduced remote-patch
  interaction test; an annulus or class mask does not make texture effects local.
- Count only video-derived payload consumed by fresh-process inflate. A deterministic regenerated
  bank can be free code; a 430,878-byte bank in the archive that inflate ignores is neither a valid
  lever nor valid byte closure.

**Launch disposition remains DESIGN/BUILD-ONLY.** The exact-D family is a replacement formulation,
not evidence that the present MLX/deploy/inflate programs agree.

### v8

- The frozen RAG contains many cycles and 6,703 high-order junctions. E pairwise fields require
  antisymmetry plus integrability, or a globally specified graph-labeling decoder; K class
  potentials avoid that ambiguity but must be named honestly.
- Sparse GT boundaries justify edge-focused rate ownership, but the source-level global path blocks
  any untested claim of exact independent-edge scorer ownership. The decoded composite needs a
  reproduced off-diagonal interaction receipt and per-class/topology guardrails.
- Frame0 should not pay Seg-carrier bytes. Its representation is Pose-only, while frame1 carries the
  partition and texture obligations.
- Every carrier still needs `encode -> exact bytes -> fresh-process decode -> render -> R`; an
  analytic description length or in-memory GT construction is not receiver closure.

**Launch disposition remains HOLD increment-1a.** Scorer/video geometry strengthens the need for an
integrable, receipt-backed design and does not repair the reversed predicate or absent matched arm.

## Smallest convincing evaluator proof matrix

1. Source/hash/video receipt reproduced from the frozen checkout.
2. Reproduce the Pose/Seg Jacobian summaries with command, model/input hashes, tensors, axis, and
   block/tail calculation in a durable receipt.
3. Strict raw cardinality: exactly 3,662,409,600 bytes and 1,200 frames before scorer iteration.
4. Resize-footprint metamorphic test: unused pixels and within-footprint kernel deltas leave both
   preprocessed scorer tensors unchanged after uint8 reload.
5. Metamorphic frame-role test: arbitrary frame0 change gives exact zero Seg change, while frame1 is
   active; Pose order remains fixed.
6. Algebraic kernel test for all six Haar/chroma atoms at the 384x512 preprocessing surface.
7. Camera-grid preimage, sigmoid/placement, clamp, uint8, resize, and fresh-raw survival test.
8. First-six Pose-output bit-stability and Seg authority on the same exact decoded candidate.
9. Same/adjacent/remote Seg Jacobian block matrix plus nonlinear argmax interaction receipt.
10. Per-class, topology, turnover, aggregate d_seg, and square-root Pose contribution at n600.
11. Exact counted payload and fresh-process receiver closure.
12. Contest-CPU and contest-CUDA exact replay only after rows 1-11 pass; never infer one from the
   other.

## Stores consulted

- frozen `evaluate.py`, `modules.py`, `frame_utils.py`, `videos/0.mkv`
- `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
- `.omx/research/upstream_scorer_alldim_reread_20260710.md`
- `.omx/research/segnet_texture_perception_20260710.md`
- `.omx/research/segnet_fragile_support_codec_budget_20260609.json`
- `.omx/research/ADVISORY_RESTART_HANDOFF_v752_v753_v8_20260710.md`
- the three vehicle advisories dated 2026-07-10
- recovered local Pose and Seg Jacobian summary values in the restart handoff; raw receipts absent

**Pointer delta:** none. **Launches:** none. **Dispatches:** none. **Runs stopped:** none.
