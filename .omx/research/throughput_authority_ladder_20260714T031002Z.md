# Task #494 — maximum-throughput compute-substrate authority ladder

**UTC:** 2026-07-14T03:10:02Z  
**Lane:** `throughput_authority_ladder`  
**Task:** `#494`  
**Status:** `BUILT; RUNG2_CLASS_PAIR_EXACT_N600_MEASURED; HOST_DEVICE_GATES_OWED`
**Authority:** `[macOS-CPU Torch one-thread advisory/QDQ feasibility]` plus
`[source-built custom Metal; host measurement owed]`  
**Flags:** `research_only=true` · `score_claim=false` · `promotion_eligible=false` ·
`rank_or_kill_eligible=false` · `pointer_moved=false`

## Pointer status — MEANS, not ENDS

The submittable pointer is **UNMOVED at `0.19108282419209976 [contest-CPU]`**
(`ad02b012…`, PR110 click-polish). The exact but borrowed-lineage defensive bank is
`0.1880443979880752 [contest-CPU]` (`196acd18…`). Task #494 builds and measures
throughput/reproducibility apparatus. It creates no archive, exact evaluator row, score claim, or
pointer move. Its only justification is accelerating the imminent V9·CGauge / pose-finish exact
row.

## Executive verdict

The optimal authority-preserving architecture is heterogeneous:

1. keep differentiable witness/teacher work on **MLX/Metal fp32**;
2. make the slow local SegNet verdict a receipt-gated **dynamic-scale fixed-point custom-Metal
   candidate**, but retain one-thread CPU-Torch as the automatic fallback; the label/frame-free
   W27..W31 weight-L1 arithmetic leaves one exact-reference tie flip, a global epsilon head fails
   heldout, and the frozen ordered `(4,0)->0` class-pair head then preserves all 117,964,800 real
   n600 source-corpus argmax pixels across its disjoint second validation;
3. skip PoseNet only while pose is frozen, using **explicitly NON-LIVE banked telemetry plus live
   canaries**, and restore CPU-Torch immediately after pose engages;
4. use exact integer accumulation for the render-R adjoint only as a training-reproducibility
   upgrade; it is not the slow authority verdict;
5. retain NumPy-fp32 decode and exact contest CPU/CUDA replay as terminal authorities.

The decisive bottleneck correction is load-bearing. The **MEASURED** n96 macOS-CPU Torch
one-thread verdict is 59.615 s, 0.621 s/pair, with SegNet/PoseNet shares 0.774/0.226. The n600
372.6 s = 6.21 min figure is a **DERIVED linear projection**, not a measurement. The fast MLX
teacher single-call backward/forward split is not the same cost center and is not used to rank this
ladder.

## Per-rung disposition

| Rung | Current verdict | Scope | What is built | Remaining authority gate |
|---|---|---|---|---|
| 1 — full render-R adjoint | **OWED ON MAIN** | full four-axis, real n600, 1,200 frames, N=10/process variant | full NumPy-fp32/int32 authority, overflow/error proof, float-atomic and Q15/int32-atomic Metal children, resumable receipt | `tools/run_full_r_adjoint_bitident_host.command` |
| 2a — fixed calibration forward | **MEASURED NO-ADMITTED-PRECISION-IN-LADDER** | `INSTANCE`: frozen SegNet/PoseNet, real 0.mkv pairs 0..599, one-thread CPU control, W8..W24 QDQ/fp32 accumulation | full receipt with exact row/hash custody | failure is **FORMULATION**, not fixed-point family; held-out calibration clipping causes a high-bit plateau |
| 2b — dynamic max-absolute QDQ forward | **MEASURED NO-EXACT-ARM THROUGH W26** | `INSTANCE`: same real n600 SegNet surface, W16..W26, label-free runtime scale, QDQ with fp32 Conv accumulation | exact 0..599 rows/hashes; W20 first tolerance arm; W26 leaves 3 flips | finite single-int64 QDQ ceiling is closed; this is a **FORMULATION** negative, not a direct-int64 negative |
| 2c — uniform dynamic exact-int64 forward twin | **MEASURED NO-EXACT-W26 INSTANCE** | W26A26 signed codes, exact int64 Conv2d accumulation, one fp32 finalization, all 125 Conv2d, unchanged fp32 non-Conv ops | exact real 0..599 custody; 4 flips at pairs 64, 362, 371, 507; training tolerance passes | `INSTANCE` negative only; label-free mixed precision and multi-limb formulations remain open |
| 2d — geometry-safe mixed exact-int64 twin | **MEASURED NO-EXACT INSTANCE** | per-layer largest W26..W30 whose `fan_in*qmax^2` fits signed int64 | exact real 0..599 custody; 1 flip at pair 11; aggregate 8.4771050e-9; training tolerance passes | `INSTANCE` negative only; tighter frozen-weight-L1 allocation remains open |
| 2e — frozen-weight-L1-safe exact-int64 twin | **MEASURED INSTANCE NEGATIVE** | per-layer largest W26..W31 whose `activation_qmax * max_oc sum(abs(weight_q[oc]))` fits signed int64 | exact real 0..599 custody; 1 flip at pair 11; aggregate 8.4771050e-9; 36 uncertified; training tolerance passes | the zero-margin tie semantics remain open; arithmetic family is not killed |
| 2f — global lowest-class epsilon tie head | **MEASURED FORMULATION/INSTANCE NEGATIVE** | dyadic epsilon ladder `0, 2^-24..2^-10`; minimum calibration-exact epsilon `2^-19` on pairs 0..119 | full split-honest n600 receipt; calibration exact; 3 heldout flips at pairs 195, 263, 587 | global near-tie correction is too broad; class-pair restriction remains distinct |
| 2g — ordered class-pair tie head | **MEASURED ARGMAX-FEASIBLE INSTANCE** | if candidate top2 is `(4,0)` and gap `<=2^-19`, choose 0; otherwise plain argmax | rule frozen from pairs 0..263: 0 flips, one snap; untouched pairs 264..599: 0 flips, zero snaps; full 0..599: 0 / 117,964,800 flips | CPU feasibility closed; device placement/latency and evolving-witness shadow remain separate |
| 3 — fixed-point verdict substrate | **BUILT; PRECURSOR ADMITTED; HOST MEASUREMENT OWED** | custom direct NHWC dense/grouped/depthwise Conv2d, exact int64 MAC, dynamic scale | all 125 Conv2d replacement, NumPy/CPU exact-integer twins, realized W27..W31 precision map, frozen receipt-selected MLX tie head, cross-process n600 harness | exact source-n600 argmax + one candidate digest across 10 processes + speedup >1; interval enclosure reported separately; actual-witness shadow/certificate before CPU suppression |
| 3 — ANE | **PUBLIC-API FORMULATION BLOCKED** | CoreML 9 public activation compute exposes W8A8; settled W8A8 PTQ failed 45.836809% held-out flips | settled-state-aware ticket compiler; refuses duplicate W8A8 and unrepresentable higher-bit requests | a genuinely distinct W8 formulation, or a public higher-bit ANE compute surface with proved placement |
| 4 — integer render-R backend | **BUILT; HOST MEASUREMENT OWED** | four axes, Q15 weights, Q7/Q5 state, no atomics, exact int32 gather | default-off VJP backend, static overflow proof, exact NumPy-state hash gate, n600 matched benchmark | full-R source receipt + exact int-state parity + repeat identity + bounded fp32 error + speedup >1 |
| 4 — integer megakernel | **UNREFUTED DISTINCT FORMULATION** | integer/exact reductions only | the integer R kernel establishes the first exact component | #356 refuted fp32 reorder/fusion; no graph-wide integer lowering or speed receipt exists |

Every negative above is scoped. Fixed calibration W24 failing does not kill dynamic scaling, mixed
precision, integer convolution, Metal, ANE, or the authority-ladder paradigm. The geometry-only
one-pixel failure does not kill tighter static bounds or multi-limb accumulation. A direct-int64 Metal
speed loss would kill that kernel formulation only; exact limb/tensor decomposition and per-layer
precision allocation would remain open.

Round-1 implementation review found a real pre-measurement throughput defect: the custom-Metal
adapter reconstructed all 125 immutable weight/scale/bias arrays on every forward. Both uniform and
mixed/weight-L1 adapters now prepare, evaluate, and retain device constant buffers once per adapter;
coverage and signature tests make that cache contract load-bearing. No speedup is claimed until MAIN
measures the corrected host path.

## Rung 2 fixed-calibration n600 measurement

Receipt:
`experiments/results/throughput_authority_ladder_20260714/fixedpoint_scorer_forward_n600_v2.json`.
It contains exact pair indices 0..599, recomputed one-thread CPU-Torch control logits/argmax/margins,
legacy-cache divergence audit, all arm rows, candidate hashes, and continuous PoseNet first-six
debt. QDQ uses fp32 accumulation and explicitly sets `native_integer_speed_claim=false`.

| Arm | Seg flips / 117,964,800 | aggregate | worst pair | uncertified pixels | Pose d_pose | sqrt(10 d_pose) |
|---|---:|---:|---:|---:|---:|---:|
| W8A8 | 1,200,717 | 1.0178604e-2 | 3.9138794e-2 | 31,063,007 | 43.8406812 | 20.9381664 |
| W10A10 | 369,799 | 3.1348250e-3 | 8.0718994e-3 | 4,636,382 | 0.1959466 | 1.3998094 |
| W12A12 | 180,728 | 1.5320502e-3 | 1.0019938e-2 | 2,033,163 | 0.0956819 | 0.9781711 |
| W14A14 | 56,640 | 4.8014323e-4 | 1.2003581e-3 | 536,430 | 0.0668171 | 0.8174170 |
| W16A16 | 13,197 | 1.1187236e-4 | 9.3078613e-4 | 182,824 | 2.6319077e-3 | 0.1622316 |
| W18A18 | 9,584 | 8.1244575e-5 | 9.3587240e-4 | 141,702 | 6.4231642e-5 | 0.0253440 |
| W20A20 | 9,066 | 7.6853434e-5 | 9.4095866e-4 | 135,112 | 4.1703818e-6 | 0.0064578 |
| W22A22 | 8,965 | 7.5997247e-5 | 9.3587240e-4 | 134,080 | 1.4909469e-6 | 0.0038613 |
| W24A24 | 8,960 | 7.5954861e-5 | 9.3587240e-4 | 133,890 | 1.3420506e-6 | 0.0036634 |

The W18→W24 plateau is evidence for held-out activation clipping in this fixed-calibration
formulation. It is not evidence that additional arithmetic precision is useless. The distinct
dynamic arm removes that clipping mechanism by selecting `max(abs(x))` on the current operator
input; max is commutative/idempotent and label-free.

The legacy GT cache differs from the recomputed one-thread control at exactly one pixel in one pair;
maximum margin delta is 3.6239624e-5. The computed one-thread control owns this experiment. The cache
is custody/audit only.

During review, the original summary incorrectly allowed `fp32_control` to become the minimum
fixed-point arm and therefore emitted a false positive. The corrected summary excludes fp32,
records exact pair-index digests, and keeps the original numerical-row producer SHA separate from
the summary-finalizer SHA. No numerical row was recomputed or relabelled.

## Rung 2 dynamic result

Receipt:
`experiments/results/throughput_authority_ladder_20260714/dynamic_fixedpoint_scorer_forward_n600.json`,
SHA-256 `feaf29ab8d1ca3fef20976586141b57dcfdb6da23c77140d142813c02f97fb5f`.
It is **MEASURED** with exact 0..599 custody. No W16–W24 arm preserves every argmax. W20 is
the minimum arm satisfying the registered aggregate and worst-pair training-tolerance gate.

| Arm | flips / 117,964,800 | aggregate | worst pair | uncertified pixels | max logit error | tolerance |
|---|---:|---:|---:|---:|---:|---|
| W16 | 5,448 | 4.6183268e-5 | 2.1870931e-4 | 67,754 | 0.2461141 | fail |
| W18 | 1,392 | 1.1800130e-5 | 4.0690104e-5 | 16,821 | 0.0515766 | fail |
| W20 | 334 | 2.8313531e-6 | 2.0345052e-5 | 4,191 | 0.0122431 | pass |
| W22 | 82 | 6.9512261e-7 | 1.5258789e-5 | 1,087 | 0.0038233 | pass |
| W24 | 19 | 1.6106500e-7 | 5.0862630e-6 | 244 | 0.0012636 | pass |

Dynamic scaling therefore removes the fixed-calibration plateau and cuts the W24 flip mass from
8,960 to 19 (about 471.6x), but it does **not** establish exact authority. The 19 flips occur one
each on 19 distinct pairs, so this is not one pathological frame.

The real frozen SegNet maximum Conv2d fan-in is 4,248. The uniform W26A26 static worst-case
accumulator bound is `4,782,822,519,189,016,728 < 2^63`, requiring 64 signed bits; W27A27 requires
66 bits and exceeds signed int64. The finite **single-int64 QDQ ceiling check** is complete:

| Arm | flips / 117,964,800 | aggregate | worst pair | uncertified pixels | max logit error | tolerance |
|---|---:|---:|---:|---:|---:|---|
| W25 | 13 | 1.1020237e-7 | 5.0862630e-6 | 139 | 5.9628487e-4 | pass |
| W26 | 3 | 2.5431315e-8 | 5.0862630e-6 | 83 | 3.2696128e-4 | pass |

The corrected receipt is
`dynamic_fixedpoint_scorer_forward_int64_ceiling_corrected_n600.json`, SHA-256
`a04a8e2672981faeda9a2a1adb086c8e1a4c073c0e1319dcd78ee1536c594c91`. W26's
three flips are on pairs 64, 371, and 587, whose reference winner-rival margins are respectively
`4.7683716e-7`, `4.0531158e-6`, and `1.4305115e-6`.

An adversarial audit found that fp32 cannot represent positive W26 `qmax=33,554,431`; a float-domain
clamp could admit `33,554,432`. Quantization now rounds to int64, clamps against the exact integer
qmax, and only then returns to the requested representation. The predecessor diagnostic receipt
`dynamic_fixedpoint_scorer_forward_int64_ceiling_n600.json` (SHA-256
`d6ccc273c0b2a9f1313588237eeb412773757c91a1e21e9b06c09dd9280a8a41`) is excluded from
authority even though its final flip totals happened to match. No result is trusted through a broken
representation contract.

This closes only **QDQ with fp32 Conv accumulation**. W26 codes returned to fp32 cannot retain every
odd integer above the 24-bit significand range, and fp32 Conv reduction is not the exact-int64 custom
kernel. The separate exact-int64 CPU twin replaced all 125 Conv2d and completed exact pairs 0..599.
Uniform W26 still has 4 / 117,964,800 flips, aggregate `3.3908420e-8`, worst pair
`5.0862630e-6`, 77 conservative uncertified pixels, and maximum absolute logit error
`2.5255978e-4`. The flips occur one each at pairs 64, 362, 371, and 507. Receipt SHA-256 is
`b4bd48f580501926492d826a8a2504f5420fa266d6270f4aff915e7820f60af2`.

This is a **MEASURED INSTANCE negative** for uniform W26 direct-int64, not an integer-convolution or
fixed-point-family negative. Its static worst-layer bound already consumes 64 signed bits, but 120
of 125 layers have smaller geometry. The successor therefore assigns each layer the largest label-free
precision in W26..W30 satisfying `fan_in*qmax^2 <= 2^63-1`: W26:5, W27:30, W28:22, W29:19,
W30:49. The full n600 receipt closed with exactly 1 flip / 117,964,800 pixels at pair 11,
aggregate `8.477105034722222e-09`, worst pair `5.086263020833333e-06`, 38 conservative
uncertified pixels, and maximum absolute logit error `7.62939453125e-05`. Training tolerance
passes, but exact authority does not. Receipt SHA-256 is
`129e9d39d09ff2e019cdab7ac04f699b64a846d319390d71d3bd12d9497959f5`.

This is a **MEASURED INSTANCE negative** for the geometry-only W26..W30 allocation. The failure is
an exact zero-margin tie at pair 11; it is not a family-level statement.

The tighter frozen-weight L1 inequality
`|acc_oc| <= activation_qmax * sum_i |weight_q[oc,i]|` is independent of frames, labels, logits,
and margins. It assigns W27:4, W28:28, W29:32, W30:41, W31:20, with worst proven accumulator
`9,035,402,569,620,285,889` and signed-int64 headroom `187,969,467,234,489,918`. Its full
real-0.mkv n600 run is **MEASURED**: one flip / 117,964,800 at pair 11, aggregate
`8.477105034722222e-09`, worst pair `5.086263020833333e-06`, 36 conservative uncertified
pixels, maximum absolute logit error `7.2479248046875e-05`, and training tolerance pass. The CPU
integer twin is only `0.080916x` the one-thread fp32 reference, so it is a numerical surface—not a
throughput claim. Receipt SHA-256 is
`bc8ce702189246b46970f85a79a78b94e68a74d59e9787d766c8c52deb96d7d5`.

The remaining flip is an exact reference tie at pair 11 between classes 0 and 4. The W27..W31
candidate separates them by `1.430511474609375e-06` in favor of class 4, while upstream
`torch.argmax` selects class 0 at the exact tie. The preregistered global lowest-class epsilon
ladder selected `2^-19` on calibration pairs 0..119, but full heldout validation falsified it:
three single-pixel false snaps occur at pairs 195, 263, and 587. Full flip mass is
3 / 117,964,800 (`2.5431315104166668e-08`). Receipt SHA-256 is
`651df3364a8921ad5b1936a9f831251c33fce2703a3c5675dc7b92607f239386`. This is a
**FORMULATION-at-n600-INSTANCE** negative for a global epsilon head, not for decision correction.

Fresh design inspection was then frozen at pair 263: pair 11 is ordered candidate top2 `(4,0)`,
whereas the pair-195 and pair-263 false snaps are `(1,0)`. Before reading any pair >=264, the
successor rule was preregistered in code: only `(4,0)` with gap `<=2^-19` snaps to class 0. Runtime
uses candidate logits/classes only; labels selected the rule on design pairs 0..263. The completed
receipt is **MEASURED**: design 0..263 has 0 flips and one intended snap; the previously untouched
second-validation pairs 264..599 have 0 flips and zero snaps; full 0..599 has 0 / 117,964,800
flips and exactly one snap. The full argmax corpus SHA-256 is
`f9458f5a37089541c2690b3d48230224132e486fe571d30f1e910c2d32729938`.
Receipt SHA-256 is
`65b7ac09705b769968429ad2cfe9dc781972348ac6da061b9d1fcdda313d7da7`;
fingerprint is
`799496b7d55a056136a621756e11d71a02b55d7711f656c3e3a6a5a7b9a52ec2`.
This proves an **INSTANCE** source-corpus decision-head feasibility result. It does not prove the
custom-Metal kernel, ANE placement, evolving reconstructed witness frames, or either contest axis.

Pair-0 smoke (**MEASURED, INSTANCE only**) was:

| Arm | flips | uncertified pixels |
|---|---:|---:|
| W16 | 8 | 126 |
| W18 | 2 | 40 |
| W20 | 0 | 5 |
| W22 | 0 | 1 |
| W24 | 0 | 0 |

No native-integer latency claim follows from QDQ or the slow CPU numerical twin. The custom Metal
command fails closed unless its selected exact-int64 CPU receipt is complete and exact; it cannot
launder a tolerance-only QDQ, uniform-W26 arm, or the geometry-only one-flip arm into authority.

## Op × substrate × precision authority-throughput assignment

| Operation | Current/default substrate | Precision | Authority grade | Final assignment / gate |
|---|---|---|---|---|
| witness forward/backward | MLX/Metal | fp32 | training signal | **ACTIVE**; keep portable NumPy/Torch parity surfaces |
| frozen teacher used for gradients | MLX/Metal | fp32 | training signal | **ACTIVE**; do not confuse with local verdict wall |
| render-R forward | MLX/Metal | fp32 | training signal | **ACTIVE**; NumPy-fp32 receiver remains reference |
| render-R adjoint | fixed-order custom Metal fp32 today | fp32 | training signal | candidate Q15/int32 gather only after exact n600/parity/speed gate |
| SegNet local verdict | CPU-Torch one thread | fp32 | local deterministic reference | automatic fallback; custom Metal may become a default-off candidate only after full conjunction |
| SegNet candidate verdict | custom Metal | frozen-weight-L1-safe per-layer W27..W31 dynamic signed int32 codes, exact int64 MAC; frozen `(4,0)->0` tie head | local candidate filter | **PRECURSOR ADMITTED; DEVICE HELD/OWED** until exhaustive source-n600 Metal equality + cross-process identity + positive speed; actual evolving witness frames still require shadow/certificate before CPU suppression |
| SegNet advisory | CoreML CPU_AND_GPU/ANE-selected | fp32 | local advisory only | retain as detached forward signal; placement and equivalence unproved |
| SegNet W8A8 | CoreML/ANE | W8A8 | no authority | **FORBIDDEN settled formulation**; 45.836809% held-out flips |
| SegNet via MPS | torch-MPS | fp32 | no authority | **FORBIDDEN**; distinct numeric drift, never rehabilitated by integer R evidence |
| PoseNet pre-pose-finish | banked telemetry + CPU canary | fp64 scalar / CPU fp32 | labelled NON-LIVE advisory between canaries | default OFF until governed n96 dry-start; K=8 candidate |
| PoseNet after pose engages | CPU-Torch one thread | fp32 | local reference | **ACTIVE**; no fixed-point/ANE certificate exists |
| archive inflate/decode | NumPy CPU | fp32 + exact integer receiver ops | portable local reference | **ACTIVE**, bit-identical bytes required |
| terminal evaluator | contest CPU Linux x86_64 | torch fp32 | score authority | **ACTIVE on exact archive bytes only** |
| terminal evaluator | contest CUDA | torch fp32 | separate score authority | operator-GO + lane claim; never inferred from CPU |

CUDA integer lowering is technically plausible but unmeasured here. It does not replace either
terminal contest axis; it first needs its own exact cross-process kernel receipt under operator-GO.

## Final witness loop — train fast, retain authority

```text
resume registry + stage checkpoint
  -> MLX fp32 differentiable witness / teacher / render-R forward
  -> receipt-admitted integer R adjoint OR fixed-order fp32 fused-R fallback
  -> stage-boundary checkpoint (EMA shadow + optimizer + stage position, atomic)
  -> local verdict:
       SegNet = admitted custom-Metal dynamic fixedpoint candidate
                + actual-witness shadow/certificate while default-off
                OR one-thread CPU-Torch fp32 authority fallback
       PoseNet = live CPU canary at index 0 and every K while frozen;
                 labelled banked value otherwise;
                 always live after pose engagement
  -> controller consumes source-labelled telemetry only
  -> candidate archive / NumPy-fp32 byte-close
  -> exact contest CPU and separately contest CUDA replay on identical bytes
```

There is no silent fallback. The host gate binds custom Metal to the QDQ and selected exact-int64
receipt fingerprints, complete per-layer precision map, decision head, scale mode, exact source-n600
custody, one candidate digest across processes, and positive speed. Source-n600 equality is the
requested feasibility surface and local candidate-filter evidence; it is not a universal theorem for
evolving reconstructed witness frames. CPU suppression therefore additionally requires a governed
actual-witness shadow/certificate gate. The interval enclosure remains separately reported and cannot
overrule direct equality. Any missing conjunct returns to CPU.

The compiled policy now reports the actual nonzero W27..W31 histogram (minimum realized precision
27), keeps custom Metal `held_owed`, and keeps one-thread CPU active because the Metal and integer-R
receipts are absent. Policy receipt SHA-256 is
`4dc658e73e608b81ae4fe661c50b2613cf43d35c234dfb99106233b710b61d67`.

For a SegNet speedup `r_seg` and Pose live cadence `K`, ignoring boundary overhead `h`, the verdict
fraction is

```text
T_new/T_old = 0.774/r_seg + 0.226/K + h.
```

This is **DERIVED**, not measured. With no SegNet cost and live Pose every time, the upper bound is
`1/0.226 = 4.4248x`. With no SegNet cost and a pre-finish K=8 Pose canary, the idealized upper bound
is `1/(0.226/8) = 35.3982x`; real custom-kernel, data-boundary, and canary costs must replace this
symbolic ceiling. The governed n96 dry-start must measure drift before the pose lever is admitted.

## ANE disposition

The public CoreML route exposes calibrated 8-bit activation quantization, not a programmable W16+
ANE convolution surface. The already-settled calibrated CoreML W8A8 formulation has 1,081,426 /
2,359,296 held-out flips (45.836809%). It is not rerun. If dynamic n600 selects more than eight
bits, `compile_ane_fixedpoint_authority_ticket.py` emits
`PUBLIC_ANE_PRECISION_UNREPRESENTABLE`. The compiled class-pair ticket requires the realized
minimum 27 bits and emits that refusal; receipt SHA-256 is
`0eb970b31cb4eafe059e747bc056149ad859735992f2aa40f9b645a529d9bd44`. That is a
**FORMULATION/API** negative, not an ANE-family negative. CoreML fp32 remains a forward-only
advisory route; #490 established zero backward selectors.

## Rung 4 and #356

Integer addition is reorder-independent only with exact partial sums, no overflow/saturation, and
one deterministic finalization. The integer-R build proves these conditions per stage and compares
the raw final int32 state against the NumPy authority hash for every real frame. It does not make
the score faster: render-R is negligible in the n24 single-call timer, while the slow local verdict
is forward-only CPU-Torch.

#356's megakernel negative was for fp32 whole-step reorder/fusion. An integer-lowered megakernel is
a distinct, unrefuted formulation, but it is not build-authorized by analogy. Every reduction and
nonlinearity in a claimed fusion domain needs an exact/bounded lowering plus a matched speed receipt.

## OSS reconciliation (#451)

The external ecosystem supports the architecture but does not supply an authority shortcut:

- PyTorch explicitly does not guarantee identical results across releases/platforms or CPU versus
  GPU, and its deterministic mode either selects a deterministic implementation where one exists or
  throws when none exists. That supports fail-closed CPU/CUDA separation; a framework flag cannot
  prove Pact's exact argmax surface. See the official
  [PyTorch reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- PyTorch's built-in quantized types center on qint8/qint32 and its documented quantization equation
  is Q/DQ with clamping; additional schemes require custom operators. Task #494's W20+ and exact
  reduction route is therefore genuinely custom rather than a missed built-in switch. See the
  [PyTorch quantization API](https://docs.pytorch.org/docs/stable/quantization-support).
- MLX officially exposes JIT-compiled custom Metal kernels with caller-declared input/output dtypes,
  grid, and threadgroup geometry. The Task #494 kernel uses that supported surface, caches each
  kernel object, and assigns one thread per output—while still requiring the real host compile and
  receipt. See [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
- Core ML's public activation-quantization path is specifically 8-bit and is paired with W8 weights
  for NE int8-int8 compute. That independently corroborates the typed higher-bit public-API blocker;
  it does not generalize the settled W8A8 failure to ANE as a family. See the
  [Core ML Tools optimization API](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html)
  and [optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html).
- TensorRT exposes explicit INT8/FP8/INT4/NVFP4 schemes, but its INT4 path is weight-only and none of
  those documents promises CPU-identical frozen-scorer argmax. CUDA remains a separately measured,
  operator-GO axis rather than an inferred equivalent. See the
  [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html).

Thus the four local probes were not rediscovering a packaged authority backend: the missing piece is
the exact, receipt-bound scorer-specific reduction/decision implementation and its real n600
cross-process measurement.

## Host execution packet for MAIN

Run on the M5-Max Metal host, in order; these are the prepared commands, not executions by this arm:

```bash
tools/run_full_r_adjoint_bitident_host.command
tools/run_fixedpoint_authority_kernels_host.command
tools/run_integer_r_adjoint_backend_host.command
tools/run_pose_verdict_gate_dry_start_host.command
tools/run_ane_fixedpoint_authority_host.command
tools/run_throughput_authority_policy_host.command
tools/run_throughput_authority_anchor_registration_host.command
```

The Pose command is a governed bounded dry-start and remains MAIN/operator execution. Paid/CUDA
dispatch, live-run/config mutation, and any run stop remain operator-GO containment.

## Triality and system intelligence

- **Equations:** empirical anchor builders target
  `exact_commutative_reduction_reorder_invariance_v1` and
  `interval_argmax_enclosure_certificate_v1`, including distinct uniform, geometry-mixed, and
  weight-L1 exact-int64, global-tie, and ordered-class-pair SegNet anchors. Fixed, dynamic,
  corrected-ceiling, uniform, geometry-mixed, weight-L1, global-tie, and class-pair anchors are
  registered; the final class-pair anchor is
  `task494_weight_l1_class_pair_tie_snap_segnet_n600_65b7ac09705b7699`.
- **Trajectory/DAG:**
  `.omx/research/throughput_authority_ladder_DAG_FEED_20260714T031002Z.md` is the standalone,
  collision-safe feed; the shared DAG was not edited.
- **DSL:** `PoseVerdictGate` and `PoseVerdictGateDryStart` are typed curriculum levers. The backend
  assignment is compiled by `throughput_authority_policy_20260714.py`; no hand-invented live flag is
  activated.
- **Sensitivity/Pareto:** fixed-point rows expose flips, worst pair, interval-uncertified mass,
  continuous Pose debt, and bit width. QDQ does not claim cost; the Metal receipt supplies measured
  latency.
- **Autopilot:** CPU fallback is unconditional until every receipt predicate passes. MPS and settled
  CoreML W8A8 are explicit refusals.
- **Continual learning:** the fixed-calibration plateau, cache-thread delta, fp32-control false
  admission bug, W26 qmax representation bug, Metal legacy-cache overwrite, per-forward constant
  buffer rebuild, global-tie heldout failure, and configured-vs-realized precision drift are durable
  typed guards/tests, not chat-only observations.

## STORES CONSULTED

Full `CLAUDE.md`; full `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`;
v7.5 §8 and v8 canonical specs; `reports/latest.md`;
`.omx/state/canonical_frontier_pointer.json`; lane/task/subagent/equation registries; latest
Codex findings/session summary and latest Claude council/design/directive memos;
`.omx/research/pythagorean_exact_arithmetic_bitident_20260713.md` (#348 lineage);
`.omx/research/cheaper_exact_forward_transfer_95kill_20260713.md` (#456);
`.omx/research/precision_backend_matrix_20260713.md` (#477);
`.omx/research/custom_metal_conv_build_20260713.md` (#478);
`.omx/research/ane_unlock_correction_20260713.md` (#482/#490);
`.omx/research/GO_PACKET_inloop_component_timer_20260713.md` (#449);
`.omx/research/throughput_frontier_math_20260714T015118Z.md`; exact model/cache/source bytes and
the Task #494 receipts named above. Paid provider state, contest evaluator execution, protected live
runs, and run pointers were not actuated.
