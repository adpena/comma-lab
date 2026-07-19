# Yhat-native generator and receiver-fold findings — 2026-07-19

**Lane:** `lane_yhat_native_generator_20260719`  
**Posture:** `research_only=true`; default OFF; local CPU measurement and design only  
**Axis:** `[macOS-CPU advisory] NON-PROMOTABLE`  
**Verdict scope:** the supplied 24 real pairs, the recorded exact-rational lattice receiver, and the
frozen CPU oracle arithmetic only. No compact generator, archive, contest-CPU/CUDA, launch,
promotion, score, or pointer authority.

## Outcome first

**Pointer delta: exactly zero.** The pointer remains
`0.1910828242 [contest-CPU Linux x86_64]`.

**One-line verdict:** exact rational scorer-plane replay is proven on both planes of all 24 real
pairs (`28,311,552 / 28,311,552` samples, zero failures), but native-float32 frozen-oracle replay is
not bit-identical and the arbitrary-rational receiver projects to **249.075 minutes** at n600.
Yhat-native is therefore a sound optimization coordinate and packaging spine, not yet a production
decoder: a viable child must either learn two independently described integer-uint8 scorer planes,
measure their scorer debt and runtime, or land a substantially faster exact rational receiver.

The durable revision-2 aggregate is
`/Volumes/VertigoDataTier/pact/evidence/yhat_native_20260719/revision2/receipt.json`, `62,332` bytes,
SHA-256 `1ad1cf84672c696b46f62ca8586bb29d5c70f55de5803902b6c37666e5b85c0f`. It binds the
formatted tool bytes that land in this branch. The tracked custody
summary is `.omx/research/yhat_native_generator_20260719_receipt.json`.

## What the n24 measurement proves

The harness computes the exact separable resize numerators for each donor frame, solves a camera
uint8 preimage, replays the exact operator, and requires numerator equality before invoking the
frozen oracle. PoseNet consumes both reconstructed frames after the official RGB-to-YUV6 path;
SegNet consumes reconstructed frame 1. Only PoseNet's declared scored half-head outputs are used.

| quantity | result | label |
|---|---:|---|
| real pairs / scorer planes | `24 / 48` | **MEASURED** |
| exact rational blocks | `28,311,552` | **MEASURED** |
| exact rational samples | `28,311,552` | **MEASURED** |
| exact failures | `0` | **MEASURED** |
| classification | `EXACT_RATIONAL_PLANE_NATIVE_F32_ULP_CLASS` for all 24 | **MEASURED, narrow scope** |
| bit-identical pairs | `0` | **MEASURED** |
| SegNet argmax disagreements | `7` | **MEASURED** |
| direct mean `d_seg` | `0.03724628016546679` | **MEASURED [macOS-CPU advisory]** |
| yhat-native mean `d_seg` | `0.03724606840599639` | **MEASURED [macOS-CPU advisory]** |
| mean `Delta d_seg` | `-2.1175947040319443e-7` | **MEASURED, yhat minus direct** |
| direct mean `d_pose` | `129.43778800964355` | **MEASURED [macOS-CPU advisory]** |
| yhat-native mean `d_pose` | `129.43754800160727` | **MEASURED [macOS-CPU advisory]** |
| mean `Delta d_pose` | `-0.000240008036286099` | **MEASURED, yhat minus direct** |

The recorded maxima are: Pose scorer-input `0.0092830658` absolute / `9,728` native-float32 ULP;
Seg scorer-input `0.0100097656` / `19,584` ULP; scored Pose output `0.0001516342` / `18,560` ULP;
and Seg logits `0.0018863678` / `1,925,542,652` ULP. The large logit ULP count is a sign-crossing
near zero, not permission to call the paths equal. The classification means only that the exact
rational planes agree and every observed finite native-float32 delta is recorded under the fixed
receiver-arithmetic law. It does not erase the seven argmax differences or authorize a score.

### Runtime boundary

- **MEASURED:** `597.7790400451 s` total exact-solve time; `12.4537300009 s` per plane on average;
  `25.7238992518 s` per two-plane pair on average; `621.8934431670 s` total invocation runtime.
- **DERIVED from the measured n24 mean:** `14,944.4760011276 s = 249.0746000188 min` for n600
  arbitrary-rational two-plane preimage projection.
- **DERIVED negative projection within scope:** the measured implementation and supplied feasible
  donor-derived fractional planes project beyond the 30-minute contest boundary. Full n600 decode
  was not run; generic expansion, packaging, and output I/O remain explicitly owed.
- **Separate settled comparator, narrower than this ABI:** the factor-2 integer receiver measured
  `4.53 s` for n12 by realizing one integer frame-1 plane and copying it under `repeat-frame1`; its
  linear n600 projection is `3.775 min`. It is not a timing result for two independently described
  Pose planes and cannot be transferred to the fractional arbitrary-rational planes measured here.

The sacred donor run stayed byte-stable: its 18-entry metadata snapshot was
`0bdc3e39c5eac970625f91e6803b1bb33330e412514b78ce35dab2c4c351842c` both before and after.
No reconstructed raw frames or scorer tensors were persisted. The cleanup verdict is
`NO_BULK_CREATED_NO_DELETION_REQUIRED`. A subsequent `--resume` reused all 24 stages under the same
binding and left the receipt, state, and aggregate-stage hashes byte-identical.

## Yhat-native head: the actual design and cost

Let `A` be the frozen camera-uint8 to scorer-RGB resize map and let `P` be an exact receiver when one
exists. The current trainer forms a scorer-grid RGB head, upsamples it to camera geometry, applies
the uint8/round surrogate, then downsamples again before the frozen scorers:

```text
y_theta in R^(384x512x3) -> camera surrogate in uint8^(874x1164x3) -> A -> y'_theta
```

The yhat-native training coordinate instead declares the head output itself to be the scorer plane:

```text
description z -> generic f32 generator G_theta(z) = yhat_theta in R^(384x512x3)
              -> SegNet(yhat_frame1)
              -> PoseNet(rgb_to_yuv6(yhat_frame0, yhat_frame1))

contest decode: yhat_theta -> exact P -> camera uint8 frames, with A(P(yhat_theta)) checked
```

This is a semantic and conditioning change, not a free model shrink. One scorer plane has
`384*512*3 = 589,824` values; one camera frame has `874*1164*3 = 3,052,008`, a coordinate ratio of
`5.1744384765625`. But the incumbent ep725 trunk already evaluates its MLP at `384x512`. Therefore:

- **DERIVED:** yhat-native eliminates the camera intermediate and fused-`R` burden from the
  *training objective coordinate*.
- **NOT DERIVED:** it does not reduce current MLP sample count by 5.17x, and it does not reduce
  archive bytes without a learned compact description.
- **RECEIVER DEBT:** contest decode still owes `874x1164` uint8 frames and exact parse-back under
  30 minutes.

The initial head may preserve the incumbent parameter shapes: Fourier coordinate features,
per-pair codes, input projection, FiLM, hidden trunk, three-channel output, SDF/palette/carrier
structures, chroma parameters, EMA state, and stage checkpoints. This avoids inventing a new
capacity claim before an A/B.

The through-`R` terms change as follows:

| incumbent force/surface | yhat-native disposition |
|---|---|
| differentiable bicubic upsample to camera | no-op in the training scorer path |
| camera uint8 rounding/STE | no-op in the training scorer path; exact receiver gate at decode |
| bilinear camera-to-scorer resize | identity in the training coordinate; exact `A(P(yhat))` check at decode |
| camera anti-alias/noise and `ker(A)` penalties | remove or rebase; they are not native yhat degrees of freedom |
| Seg loss | retain on scorer-plane frame 1 |
| Pose loss | retain on RGB-to-YUV6 of both scorer-plane frames |
| chroma/luma regularization | retain only when justified by measured scorer debt; neither channel is free |
| rate/model-size pressure | retain on the counted description, never on the expanded plane |
| EMA, stage boundaries, resume state | retain unchanged and preserve every stage |

Two settled signals motivate the coordinate without granting byte savings. The camera resize has
`230,904` blind camera pixels per frame (`22.6969%`), and the incumbent n32 measurement put
`52.42%` raw / `52.88%` mean-removed rendered energy in `ker(A)`, with roughly `50-53%` of marginal
output-layer effect scorer-invisible. These are **MEASURED on the incumbent** capacity signals. They
are not an archive-rate result for a yhat-native model.

The authoritative optimization objective remains

```text
S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489.
```

Admission requires a receiver-closed improvement in that objective or a scoped blocker; proxy head
loss alone is insufficient.

## Packaging spine: `YhatNativeDescription.v1`

This is a proposed wire contract, not implemented archive code. The parser must be strict and the
expanded two-plane digest must bind what the receiver realizes:

```text
counted YhatNativeDescription.v1
  -> strict parse + canonical re-encode identity
  -> deterministic NumPy-fp32 generic expander
  -> ordered two-plane yhat tensor [N,2,384,512,3]
  -> exact lattice preimage receiver
  -> camera uint8 frames [2N,874,1164,3]
  -> exact A replay + archive parse-back
```

The header must bind schema/version, source/runtime hashes, float32 byte order and arithmetic order,
geometry, pair count, seed, section IDs/order/lengths/hashes, and the expanded two-plane hash. A
decoder must refuse unknown or missing sections, reordering, trailers, non-finites, negative zero,
noncanonical encodings, and any hash drift.

Proposed inner sections, all optional only when the header makes absence canonical:

1. PDW2 margin-preserving certificate when class margins are consumed; partition-only PDP2 only
   when positive-scale quotient information is sufficient.
2. Yhat-generator weights and fixed-shape architecture metadata.
3. Pair/frame state codes.
4. Cell, edge, phase, or texture seeds.
5. Optional sparse yhat residual.
6. Canonical framing and integrity hashes.

### Rule-118 counted/free split

| FREE generic receiver material | COUNTED video-derived material |
|---|---|
| deterministic expander code and fixed operations | PDW2/PDP2 payload selected or fitted for this video |
| fixed non-video priors | learned or video-fitted generator weights |
| PDW2 parse/evaluation code | per-pair/per-frame codes and fitted seeds |
| exact lattice algorithm | fitted scales, normalizers, palettes, or RNG state |
| camera output assembly | per-video entropy-model parameters |
| generic coder implementation | yhat residuals, section framing/hashes, and archive overhead |

Any scorer-derived shipped payload is counted. Scorer weights, SegNet, PoseNet, and a GT-argmax
table are forbidden from the archive. A direct dense yhat payload is rate-dead; the compact
generator must learn or fit the plane.

## PDW2 dependency and no-duplication boundary

Sibling commit `edf47756ba629e079a2a63233bf8f0293cf85f3d` owns PDW2 implementation details and must be
reviewed by MAIN before integration. Its measured n600 packet results are:

- margin-preserving PDW2: 20 float32 coefficients, `138` raw / `133` Brotli-q11 bytes;
- partition-only PDP2: 19 scalars, `134` raw / `122` Brotli-q11 bytes;
- strict parse/re-encode and the frame-195 native-float32 tie gate pass;
- verdict remains `TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`.

Yhat-native consumes that format only as an inner target/certificate. It does not duplicate the
PDW2 codec, arithmetic-coder, or neural self-compression siblings. Margin-preserving PDW2 is the
default when the generator uses margin magnitude; PDP2 cannot silently substitute because it
forgets that magnitude.

## Chroma, luma, and Pose boundary

PoseNet owns both scorer-plane RGB frames through RGB-to-YUV6. SegNet owns frame 1 RGB. Therefore:

- the first plane cannot be omitted as “Seg-invisible” because Pose uses it;
- the second plane cannot be reconstructed from a Seg-only certificate without measuring Pose;
- chroma cannot be dropped merely because the active structural target is luma-heavy;
- luma cannot be frozen merely because Seg labels survive; Pose remains sensitive to motion and
  luminance structure;
- any integer-yhat projection is a new optimization target whose Seg and Pose debt must be measured,
  not an exact reparameterization of the fractional incumbent.

The n24 mean deltas happen to be slightly favorable, but seven Seg argmax pixels changed. That is an
empirical warning against promoting “same rational plane” to “same frozen-oracle output.”

## Triality and solver-stack disposition

- **DSL leg:** `YhatNativeGeneratorPolicy` and the keyword-only `YhatNativeGenerator(policy=...)`
  factory are argv-inert, research-only, and default OFF. All authority flags are sealed false; the
  n24 receipt gate is recorded closed, receiver/archive gates remain owed, and the lever appears in
  discovery/duty tracking but not the nilary/composable launcher set.
- **DAG leg:** proposed feed is
  `compact description -> deterministic yhat expander -> integer-fast or exact-rational receiver ->
  exact A replay -> archive parse-back -> separate contest CPU/CUDA`. No autopilot dispatch hook is
  enabled while receiver/runtime/archive leaves remain open.
- **Equation leg:** the measurement uses the existing
  `f32_receiver_arithmetic_exactness_admissibility_v1` law and exact rational lattice equations. No
  new law is registered by this arm.
- **Sensitivity/Pareto/bit allocator:** do not admit a row until the compact description yields
  receiver-closed `(Delta bytes, Delta d_seg, Delta d_pose)` custody.
- **Resume/storage:** measurement stages are atomic and preserved; `--resume` binds scientific
  inputs independently of the transport bit. SSD preflight selected VertigoDataTier. No bulk output
  was created.

## Open gates and exact next decision

1. Choose and implement one bounded child: two distinct integer-uint8 yhat planes plus measured
   scorer-debt/runtime A/B, or a faster exact rational receiver. Do not infer equivalence between
   them or reuse the repeat-frame1 timing as a two-plane measurement.
2. Implement the compact `YhatNativeDescription.v1` parser/expander with strict canonical re-encode.
3. Preserve full training resume state and every stage checkpoint; yhat-native trainer integration
   remains unimplemented and unauthorized.
4. Measure the full n600 expander + receiver + output I/O below 30 minutes; no projection is enough.
5. Build exact archive bytes, parse them back, and evaluate the same bytes separately on
   contest-CPU and contest-CUDA.
6. MAIN must review this branch and sibling PDW2 commit
   `edf47756ba629e079a2a63233bf8f0293cf85f3d` before merging or changing the active v10 program.

## Stores consulted

- `docs/operating_manual_craft_handoff.md`
- `.omx/research/SPEC_v10_capstone_RECONCILED_20260719.md`
- `.omx/research/power_diagram_witness_20260718.md`
- `.omx/research/production_receiver_543_20260719_codex.md`
- `.omx/research/production_receiver_543_byteclose_receipt_20260719.json`
- `.omx/research/null_subspace_rate_measure_20260717.md`
- `.omx/research/yhat_native_generator_20260719_implementation_spec.md`
- sibling commit `edf47756ba629e079a2a63233bf8f0293cf85f3d`
- `experiments/train_witness_realized_through_R_mlx.py`
- `upstream/modules.py`
- `src/tac/optimization/uint8_lattice_feasibility.py`
- `src/tac/canonical_equations/f32_receiver_arithmetic_law_20260719.py`
- the external n24 receipt, state, storage preflight, cleanup manifest, and run journal named above

This artifact follows `docs/operating_manual_craft_handoff.md`: outcome first, re-derivation from
the actual code and bytes, labels attached to every number, the strongest counterexample exposed,
and negatives limited to their measured scope.

## MAIN landing requirement

This branch grants no live authority. MAIN must review the complete base-to-head diff, verify the
external receipt and cleanup hashes, rerun the focused and resume tests, review the PDW2 dependency,
and only then merge or fold the proposed amendments into shared planning surfaces.
