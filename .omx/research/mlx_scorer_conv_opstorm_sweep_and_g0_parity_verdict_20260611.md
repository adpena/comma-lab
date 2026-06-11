# MLX scorer conv op-storm sweep + g0 native-swap parity verdict (2026-06-11)

**Authority:** `[macOS-MLX research-signal]` — NON-PROMOTABLE, advisory only, $0 spend, NO MPS.
**Frontier:** UNMOVED (0.19110). This is a LOCAL-THROUGHPUT enabler, NOT a contest-score mover.
**Subagent:** `mlx-vectorize-g0-20260611`. Reuse-first (no scorer rebuilt; existing adapters + tests used).

## Headline

**g0 is REJECTED.** The fixed-order FP32 reference conv path is **load-bearing for
d_seg-exactness**, not merely a backward-gradient probe. Swapping the 4 grouped+strided
depthwise convs in the LIVE SegNet forward to native `mx.conv2d` produced **1 deterministic
per-pixel argmax flip** on real `0.mkv` frame 1125. d_seg is the argmax-flip RATE, so even
one flip CORRUPTS the authority. A "faster" scorer that changes d_seg by one flip is a FAKE
optimization (it corrupts the authority) — forbidden. The slowness is the load-bearing price
of d_seg-exactness; that parity gap is itself the finding.

## Task A — repo-wide op-storm bug-class sweep

Bug class = a Python `for` loop emitting `O(channels × kernel-taps × spatial × pairs)`
MLX/array ops per call on a hot path (vs `O(num-layers)` loops, which are fine).

| File:line | Construct | Classification | Op-inflation | Hot path? |
|---|---|---|---|---|
| `mlx_scorer_adapters.py:539-578` | `mlx_reference_conv2d_nhwc` (3 nested loops kh×kw×in_ch) | **BUG-CLASS op-storm** (but parity-load-bearing) | ~25,056 mul-ops for the 4 SegNet grouped+strided convs (blocks.1/2/3/5.0.conv_dw: 864+3600+2592+18000) ≈ 98% of encoder op-launches | YES — live SegNet eval |
| `mlx_scorer_adapters.py:451-453` | `MLXExplicitSpatialConv2dAdapter.__call__` (kh·kw·in terms, full-spatial each) | **BUG-CLASS op-storm** (parity-load-bearing, MOST sensitive layer = final logits) | SegNet head 3×3×16=144 terms over 196k px; PoseNet-stem SE fc1/fc2 = in terms over 1×1 spatial (negligible) | YES — live SegNet head feeds argmax directly |
| `mlx_segnet_repaired_se_probe.py:299` | `_explicit_ordered_1x1_conv_nhwc` per-channel loop | DIAGNOSTIC PROBE (variant experiment, not live eval) | in_channels terms | NO |
| `mlx_segnet_se_conv_variants.py:192` | `_explicit_ordered_1x1_conv_nchw` per-channel loop | DIAGNOSTIC PROBE (variant experiment) | in_channels terms | NO |
| `deterministic_primitives.py:888-902,966-980` | `kahan_conv2d_3x3` / `fp64_intermediate_conv2d_3x3` (numpy 5-deep loops) | COLD/REFERENCE (pure-numpy CPU reference convs, drift probes) | n·h_out·w_out·c_out (numpy, not MLX ops) | NO |
| `pr95_hnerv_numpy_reference.py:136-138` | numpy N·H·W triple loop | COLD numpy reference (portability oracle) | numpy, not MLX | NO |
| `cool_chic_carrier.py:331` | `for bi in range(b)` per-pair `_synth` | TORCH-carrier (refuted smaller-basis lane) | O(batch) not channel×tap×spatial | LOW (refuted lane) |
| `cool_chic_carrier.py:358/369` | `for p in range(n_pairs)` rate accounting | FINE (O(n_pairs) under no_grad, accounting) | O(n_pairs) scalar | NO |
| `mlx_preprocess.py:262/499/1009` | `for start in range(0, …, batch)` | FINE (O(num-batches) batching) | O(batches) | minibatch loop, fine |
| `pr95_hnerv_mlx.py` training loops, `mlx_trainer.py`, `score_aware_loop/trainer.py` | epoch/minibatch `for` | FINE (O(epochs)/O(batches)) | — | fine |

**Ranked BUG-CLASS hits (op-count inflation × hot-path frequency):**
1. `mlx_reference_conv2d_nhwc` (4 grouped+strided SegNet convs) — ~25k ops, every SegNet forward. **Parity-load-bearing → NOT swappable (g0 verdict below).**
2. `MLXExplicitSpatialConv2dAdapter` (SegNet head) — 144 full-spatial terms, every SegNet forward, on the MOST argmax-sensitive layer. **Parity-load-bearing → NOT swappable.**

There is no safe, bit-exact native vectorization to land here: the two op-storms that
dominate the encoder op-count are exactly the two deliberate FP32-exact reference paths whose
fixed accumulation order protects the d_seg argmax. The other loops are either diagnostic
probes (cold), numpy portability oracles (cold), refuted-lane torch carriers, or fine
O(layers)/O(batches) loops.

## Task B — g0 d_seg parity gate result

Routing (`torch_conv2d_to_mlx`, lines 1064-1097): grouped (`groups>1`) AND strided
(`stride≠(1,1)`) convs route to `MLXReferenceConv2dAdapter` "fixed_fp32". The 4 such convs in
SegNet (EfficientNet-B2): `blocks.1.0.conv_dw` (96, 3×3, s2), `blocks.2.0.conv_dw` (144, 5×5,
s2), `blocks.3.0.conv_dw` (288, 3×3, s2), `blocks.5.0.conv_dw` (720, 5×5, s2).

Probe: built the SegNet adapter two ways (current reference-routing vs forced-native
`mx.conv2d` for the 4 convs), ran real `0.mkv` frames through the actual frozen `modules.py`
SegNet, compared per-pixel argmax (the d_seg authority).

| Test | Frames | Argmax flips | Worst logit Δ | Verdict |
|---|---|---|---|---|
| First real frame | 1 | **0** | 2.3e-5 | bit-exact |
| every-25th sweep | 48 | **1** (frame 1125) | 1.08e-3 | **NOT bit-exact** |
| every-50th sweep | 24 | 0 (missed 1125) | 1.9e-4 | — |
| dense @1124/1125/1126 | 3 | 1 (1125); native-vs-native = 0 (deterministic) | — | flip is a stable float-order disagreement |

The frame-1125 flip is at a pixel with top-2 logit margin **6.2e-5**; the native-vs-reference
logit delta there (1.08e-3) exceeds the margin and flips the SegNet class → changes d_seg.
The flip is **deterministic** (native-vs-native = 0 flips), confirming it is a genuine
accumulation-order disagreement, not run-to-run noise.

**Speed (advisory, ratios only):** native SegNet forward ≈ 18× faster than reference forward
(0.023s vs 0.417s, b=1, first call). Real but irrelevant — the swap is rejected on authority.

**Decision:** native forward swap **NOT LANDED**. d_seg is not bit-exact across the frame
distribution; the reference's FP32-exact fixed accumulation order is load-bearing. This is
consistent with the already-documented `SEGNET_MLX_TORCH_LOGIT_DRIFT_BOUND = 0.05` and the
`deterministic_tie_resolved_segnet_argmax` apparatus, which exist precisely because the MLX
SegNet port is NOT bit-exact vs torch at near-ties.

## What WAS landed (parity-gated)

- **Regression test** `test_grouped_strided_reference_path_is_also_forward_d_seg_exactness_crux`
  in `src/tac/tests/test_mlx_scorer_adapters.py` — encodes the NEW reason the reference routing
  is load-bearing: not only the backward-gradient crux (existing test, line 181) but also
  FORWARD d_seg-exactness (near-tie argmax stability). Deterministic, no video needed; pins
  that native `mx.conv2d` is numerically close (<5e-2) but NOT bit-identical to the fixed-order
  reference on a grouped+strided depthwise conv — the residual that flips near-tie argmaxes.
  A future agent who routes these convs to native must first re-run the real-frame d_seg gate.
- No production code edited. 43/43 adapter tests pass (was 42 + 1 new). ruff clean.

## Residual opportunities NOT landed + why

- **Native `mx.conv2d` for the 4 grouped+strided SegNet convs** — REJECTED (1 d_seg flip; authority corruption).
- **Native for the SegNet-head explicit-spatial conv** — NOT ATTEMPTED; even more argmax-sensitive (final logits feed argmax directly), same near-tie flip risk, higher than the encoder convs. Routing it native would be strictly riskier.
- **PoseNet-stem SE `fc1`/`fc2` explicit-spatial convs** — negligible op count (1×1 spatial after global pool); no throughput win to chase.
- **`cool_chic_carrier.py:331` torch per-pair `_synth` loop** — a moderate O(batch) torch inefficiency on a refuted smaller-basis lane; could be vectorized with batched `_synth`, but it is torch-carrier (not MLX op-storm) and low-EV; left alone.
- **A tie-tolerant native fast path** (native everywhere, then `deterministic_tie_resolved_segnet_argmax` to torch-resolve only near-ties) is a *possible* future enabler — it already exists as an apparatus — but it (a) needs torch in the loop and (b) is out of scope for a $0 parity-gated forward swap; it does not make the reference path swappable for the bit-exact forward.

## Wire-in (6 hooks)

1. sensitivity-map — N/A (no score-affecting bytes; advisory diagnostic).
2. Pareto — N/A.
3. bit-allocator — N/A.
4. cathedral autopilot — N/A (not archive-deployable).
5. continual-learning posterior — N/A (no exact anchor; advisory MLX research-signal).
6. probe-disambiguator — **ACTIVE**: the regression test IS the disambiguator pinning
   "reference routing is load-bearing for forward d_seg" so the swap cannot be silently reintroduced.
