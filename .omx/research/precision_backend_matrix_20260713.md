# Frozen-SegNet precision x backend matrix — 2026-07-13

`research_only=true` · `$0 local` · `score_claim=false` · `pointer_moved=false` ·
`training_launched=false` · lane `lane_precision_backend_matrix_20260713`

## Verdict in one line

**Native useful conv exists for fp32/fp16 across MLX/MPS/CPU, bf16 on MLX/CPU (MPSGraph per-op
execution remains unverified), CPU W8A8, and CoreML/ANE W8A8; MLX has no native quantized conv but
im2col + weight-quantized matmul is built; MPSCNN UInt8 weights dequantize to fp16; and usable optimized
int16 neural conv is absent on every candidate backend, although raw same-dtype CPU `aten` int16 conv
exists and overflows.** ANE, A3, and MLX-im2col device timing are **NOT MEASURED** in this process: the
first was blocked by DNS while installing `coremltools` into the required scratch venv, and the latter
two reached MLX's preflight but the process had no Metal device.

That is an environment-scoped blocker, not a format-family negative. The exact commands and resumable
receipts are preserved below.

## Status vocabulary

- **EXISTS:** a documented or locally executed convolution path consumes that compute precision.
- **STORAGE:** compressed weights exist but are dequantized/reconstructed for floating convolution;
  this is not integer convolution.
- **BUILDABLE:** all necessary lower-level primitives exist, but this repository has no admitted full
  frozen-SegNet convolution path yet.
- **RAW-ONLY:** a generic primitive executes, but its arithmetic/gradient contract is not a usable
  quantized neural-network path.
- **ABSENT-PUBLIC:** no documented supported conv path or installed dispatcher kernel was found. A
  datatype enum alone is not an operator implementation.
- **UNVERIFIED:** headers/API make the route plausible, but device execution was unavailable here.

## Full existence matrix

| Format | MLX-GPU 0.31.2 | Custom Metal / #212/#443 program | MPSGraph / MPSCNN / torch-MPS | ANE via CoreML | CPU-torch 2.12.1 |
|---|---|---|---|---|---|
| **fp32** | **EXISTS** `nn.Conv2d`; settled 20.1974 ms full forward | **BUILDABLE/EXISTS-primitives**; existing explicit-order kernels; a sibling-owned forward-conv build is in progress but has no admitted receipt yet | **EXISTS** MPSGraph/MPSCNN and torch-MPS floating conv | **EXISTS-conversion**, but CoreML selects placement/precision; no claim of native fp32 NE compute | **EXISTS**, autograd executed; exact 1-thread teacher standard, 302.06995825 ms/pair |
| **fp16** | **EXISTS + MEASURED**; 20.44 ms, 0.9915x vs fp32, cosine 0.9999586 | **BUILDABLE** direct/simdgroup/tensor conv; no full build; native MLX already speed-neutral | **EXISTS**; MPSCNN's declared kernel precision includes fp16 | **EXISTS** W16A16 / default ML Program fp16 route | **EXISTS**, autograd executed; not selected for authority or speed |
| **bf16** | **EXISTS API/dtype**, full path **UNVERIFIED** here; settled NumPy bridge blocker prevents n600 receipt | **BUILDABLE** on M5 tensor formats; no repo conv implementation | MPS datatype **EXISTS**, MPSGraph conv is present, but bf16 conv execution is **UNVERIFIED**; torch-MPS process had no device | **ABSENT-PUBLIC** as a documented ANE convolution precision | **EXISTS**, conv + autograd executed locally |
| **int8 W8A8 / W8-only** | Native quantized conv **ABSENT**. **BUILDABLE** as im2col + `quantized_matmul`; W8A8 proof is activation-QDQ + W8 matmul, not native int8 activation compute | **BUILDABLE** W8A8/W8-only using M5 int8 tensors/simdgroup or MPP matmul; calibration, accumulators, depthwise/grouped variants, and VJP still owed | Native W8A8 conv **ABSENT-PUBLIC**. MPSCNN UInt8 is **STORAGE→fp16**; MPSGraph exposes Q/DQ; installed `quantized::conv2d.new` has QuantizedCPU=true, MPS=false | **EXISTS** W8A8 int8-int8 on M4-class-and-newer NE; W8-only **STORAGE/float-runtime** path also exists | W8A8 **EXISTS** through QuantizedCPU/QNNPACK. W8-only conv **ABSENT-native**, QDQ buildable |
| **int16** | **ABSENT-PUBLIC usable neural conv**; dtype exists, but no documented optimized/quantized conv contract | Integer arithmetic is **BUILDABLE**, but there is no demonstrated int16 neural-conv speed path and widened accumulation/scaling must be designed | Datatype enum exists, but conv support is **ABSENT-PUBLIC**; MPSCNN admits only UInt8/fp16/fp32 weights and its UInt8 route becomes fp16 | **ABSENT-PUBLIC** | **RAW-ONLY EXISTS**: `F.conv2d(int16)` returns int16 without autograd; 810000 wrapped to 23568. Usable quantized teacher path **ABSENT** |
| **int4 / int6 group or palette** | Native conv **ABSENT**; **BUILDABLE W-only** through im2col + `quantized_matmul` at 4/6 bits | int4 tensor path **BUILDABLE** on M5; int6 requires packed unpack/LUT code; neither is a full repo conv | int4/i8 dequantization **EXISTS-STORAGE** in MPSGraph; MPSCNN LUT weights become fp16; no native int4/int6 conv established | CoreML 1/2/3/4/6/8-bit palettization **EXISTS-STORAGE**; int4 weight quantization exists, but runtime floating op is not native int4 conv | Native int4/int6 conv **ABSENT**; pack/dequant then float/int8 is buildable |

### Receipts behind the matrix

MLX's official 0.31.2 API describes `quantized_matmul` as multiplication by a packed quantized **matrix**,
and `nn.quantize` documents Linear and Embedding as the default quantized layers; its layer inventory has
ordinary Conv2d but no QuantizedConv2d. This supports **ABSENT native quantized conv**, not absence of a
buildable lowering: [MLX quantized_matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html),
[MLX neural-network inventory](https://ml-explore.github.io/mlx/build/html/python/nn.html).

The installed macOS 26.4 SDK is sharper than the ambiguous phrase “MPSCNN int8 support.”
`MPSCNNConvolution.h` says quantized UInt weights use the same scheme to **dequantize to fp16 for CNN
convolution**, and requires UInt8 weights to select fp16 kernel weights. Therefore it is compressed weight
storage, not W8A8 integer compute. The header excerpts and SHA-256 are in
`cpu_sdk_support.json`. MPSGraph separately exposes i8/u8 quantize-dequantize and i4/u4 dequantize ops,
but its public convolution header does not establish an integer convolution. Apple exposes the general
framework and dtype enumeration here: [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph),
[MPSDataType](https://developer.apple.com/documentation/MetalPerformanceShaders/MPSDataType). PyTorch's
MPS backend is a floating MPSGraph/tuned-kernel backend; the installed dispatcher is the decisive receipt
for this wheel: [PyTorch MPS notes](https://docs.pytorch.org/docs/stable/notes/mps).

For custom Metal, Apple's current M5 feature tables include Int4, Int8, Int16, fp16, bf16, and fp32 tensor
formats, and Metal Performance Primitives exposes the tensor/matrix program. That proves lower-level
**buildability**, not ready convolution: [Metal feature-set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf),
[Metal Performance Primitives programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).
The repository's own #443 stack supplies the stronger local precedent: the custom grouped-conv backward is
already ~17x and explicit-order fused-R/persistence kernels are parity-gated. At final collision audit,
the sibling lane `lane_custom_metal_conv_20260713` owned an in-progress pointwise/depthwise forward build;
it had no completed benchmark/admission receipt and its files were not inspected or edited here. Therefore
this matrix still marks the full replacement BUILDABLE rather than EXISTS/MEASURED.

CoreML distinguishes real integer compute from storage compression. W8A8 can use faster int8-int8 compute
on Neural Engine beginning with A17 Pro/M4-class hardware; weight-only int4/int8 normally reconstructs a
floating op, and 1/2/3/4/6/8-bit palettization is a LUT representation. Apple recommends W8A8 only when a
model runs mostly on NE because CPU/GPU activation quantization can slow down:
[CoreML quantization overview](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html),
[quantization performance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html),
[palettization overview](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html).
`CPU_AND_NE` excludes GPU and is available from macOS 13:
[CoreML compute units](https://apple.github.io/coremltools/source/coremltools.models.html).

## New local measurements

All numbers below are `[macOS-CPU/MLX research signal; NON-PROMOTABLE]`. They are means, not contest score.

| Probe | Input / scope | MEASURED result | Verdict |
|---|---|---|---|
| CPU raw int16 | 3x3 scalar overflow witness | mathematical 810000; torch int16 23568; same-dtype output, no autograd | literal “int16 primitive absent everywhere” **REFUTED**; usable teacher int16 path **ABSENT** |
| CPU W8A8 stem, B=1 | real RGB pixels, synthetic deterministic 3→32 stem weights, 384x512 | fp32 1.4805 ms; QNNPACK 1.4716 ms; **1.0061x**; cosine 0.9991040; 7,279/49,152 channel-argmax flips | establishes CPU QuantizedConv2d existence only; no frozen-SegNet fidelity claim |
| CPU W8A8 stem, B=8 | same, eight copies | fp32 13.3931 ms; QNNPACK 10.9861 ms; **1.2191x**; cosine 0.9991040; 58,232/393,216 flips | batching exposes a modest CPU kernel win; still not the requested teacher path |
| MLX W8/W8A8/W6/W4 im2col stem | exact v7.5.2 EMA, real states, B={1,8}, n16 quality planned | `timing=null`, `quality=null`; Metal preflight `available=false` | **BLOCKED-NOT-MEASURED**, environment scope |
| A3 full frozen teacher W8A8 QDQ | exact v7.5.2 EMA, real n600 requested | `timing=null`, `quality=null`; Metal preflight `available=false` | **BLOCKED-NOT-MEASURED**, no format verdict |
| A2 CoreML/ANE | exact frozen SegNet, CPU_AND_NE, B=1/best batch requested | `latency=null`; scratch venv created, `coremltools` fetch failed after 3 DNS retries | **BLOCKED-NOT-MEASURED**, no conversion/placement/latency verdict |

The CPU W8A8 row intentionally uses synthetic weights: it audits the installed kernel and stem geometry,
not the scorer's accuracy. The requested frozen-weight, real-frame fidelity belongs to the MLX im2col
receipt and remains null. It is not backfilled from the CPU experiment.

## Rung B1 — MLX im2col + quantized matmul proof

`tools/probe_mlx_im2col_quantized_stem_conv.py` is the minimal built proof. It replaces only the frozen
EfficientNet-B2 3x3x3 stride-2 stem, lowers the convolution to nine strided NHWC views, concatenates them,
pads K=27 to the supported group size 32, and invokes affine `mx.quantized_matmul`. It preregisters:

- W8-only, W6-only, and W4-only weight matmul;
- W8A8 as dynamic symmetric activation QDQ with identity STE plus W8 matmul, explicitly **not** a native
  integer-activation kernel;
- B=1 and B=8 timing against native fp32 stem convolution;
- stem cosine/relative-L2/channel flips and full-SegNet final-logit cosine/class flips on real states;
- atomic per-pair resume and fingerprint refusal.

The exact attempted command is preserved in the receipt:

```bash
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python \
  tools/probe_mlx_im2col_quantized_stem_conv.py \
  --quality-pairs 16 --batch-sizes 1 8 --warmup 3 --repeats 10 \
  --out experiments/results/precision_backend_matrix_20260713/im2col_quantized_stem.json
```

It reached device discovery and emitted `BLOCKED-NOT-MEASURED`; this process reports no Metal device.
Therefore **im2col speed and frozen-SegNet fidelity are NOT MEASURED**, and no native quantized-conv claim
is made.

## Rung B2 — custom Metal feasibility and honest effort

The mechanism is feasible: M5 tensor/MPP matmul supports the relevant fp16/int8 formats, and the #212/#443
program already has explicit-order, parity-gated Metal kernels plus a ~17x grouped-conv backward. A
sibling-owned pointwise/depthwise implementation is now in progress, but no completed receipt establishes
the full forward operator family needed by EfficientNet-B2: dense 3x3, depthwise/grouped, pointwise 1x1,
stride variants, bias/BN/SiLU fusion, scale/zero-point policy, fp32 accumulation, and a declared gradient
surrogate.

Effort is **ENGINEERING-ESTIMATED, not measured**, and will be superseded by the sibling lane's actual
receipt if/when it completes:

| Build | Prototype | Admission-quality total | Main risk |
|---|---:|---:|---|
| stem-only fp16 direct/tensor conv | 2–4 engineer-days | 4–7 days including cross-shape parity/timing | native MLX fp16 is already 0.9915x, so prize may be zero |
| full SegNet fp16 forward family | 1–2 weeks | 2–3 weeks with all shapes, launch gates, n600 and VJP interaction | maintenance surface and reduction-order drift |
| stem-only W8A8 int8 | 3–6 days | 1–2 weeks with calibration, saturation and real-state gradient checks | activation scales and accumulator order |
| full SegNet W8A8 forward + training surrogate | 2–4 weeks | 4–6 weeks with depthwise/grouped variants, n600, resume and typed policy | a fast forward is insufficient without an admitted VJP |

The in-progress sibling build should be admitted only if its receipt shows a prize over native conv at B=1
or B=8 **and** full-logit/gradient fidelity survives. A follow-on W8A8 build should additionally wait for
the im2col/full-teacher signal; otherwise it repeats #443's rejected below-noise build class.

## A2 CoreML/ANE attempt

The authorized isolated environment was created at
`experiments/.scratch/precision_backend_matrix_coreml` using CPython 3.12.13. The second install attempt
failed after three retries because this process could not resolve `pypi.org`; no package was installed,
model converted, or compiled asset created. `coremlcompiler` itself exists in Xcode, but it cannot compile
a missing converted model. The 68 KiB empty venv and 64 KiB failed-download cache were certified; targeted
cleanup was rejected by execution policy before process creation, so 132 KiB remains rather than being
silently deleted.

Consequently:

- ANE B=1 latency: **NOT MEASURED**;
- best-batch latency: **NOT MEASURED**;
- Neural Engine placement: **NOT CONFIRMED**;
- speedup vs settled CPU 1-thread 302.06995825 ms: **NOT DERIVABLE**;
- speedup vs settled MLX fp32 20.1974 ms: **NOT DERIVABLE**.

When package/network authority exists, the receipt's exact continuation is: convert exact frozen SegNet at
NCHW `[B,3,384,512]` for B in `{1,8}`, load with `ComputeUnit.CPU_AND_NE`, inspect the compute plan to prove
NE placement, then time warm medians and compare real-frame argmax. CoreML model loading can include device
specialization, so compilation/load time must be kept separate from steady prediction latency:
[CoreML model prediction](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html).

## A3 full W8A8 ticket

The committed probe was run exactly at n600 quality coverage:

```bash
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python \
  tools/probe_mlx_real_n600_int8.py \
  --quality-pairs 600 --timing-pairs 8 --warmup 1 --repeats 3 \
  --out experiments/results/precision_backend_matrix_20260713/a3_quality_n600.json
```

It emitted a durable preflight receipt with correct checkpoint/cache custody and null quality/timing. The
existing admission predicate remains unchanged: n600, global gradient cosine >=0.99, minimum per-pair
gradient cosine >=0.99, and measured step speedup >=1.5x. This is a training-tolerance gate, not verdict
authority.

## Optimal heterogeneous assignment

### Current executable optimum

**Training:** MLX fp32 frozen-teacher forward + the already-landed custom Metal grouped-conv backward.
Keep the NumPy/CPU reference and settled asynchronous 1-thread CPU-torch verdict. fp16 is quality-admitted
but speed-neutral; bf16 bridge and W8A8 are unmeasured; ANE has no backward.

**Detached monitoring:** CoreML/ANE W8A8 is the highest-EV future forward-only route if A2 proves mostly-NE
placement, B=1/best-batch latency, and real-state class fidelity. It can offload monitoring; it does not by
itself kill the gradient-bearing teacher.

**Conditional future training candidate:** ANE W8A8 forward + MLX fp32/QDQ custom VJP. Admit only if A2
passes latency/placement/fidelity and A3 passes the n600 gradient predicate. A CoreML logit followed by a
surrogate MLX VJP is an approximate treatment and must remain training-only.

### Honest Amdahl composition

Use the operator-set wall split `f_fwd=0.78`, `f_bwd=0.17`, `f_other=0.05`. Let
`r=T_ANE,fwd/T_MLX,fwd` and let `h` be synchronization, cast, copy, Python/CoreML boundary, and extra-VJP
cost normalized by the current total wall.

For an admitted approximate VJP that does **not** recompute the MLX forward:

`T_new/T_old = 0.22 + 0.78*r + h`, so `speedup = 1/(0.22 + 0.78*r + h)`.

- **DERIVED ideal upper bound:** with `r=h=0`, speedup is `1/0.22 = 4.54545x`. ANE forward alone can
  remove at most the 78% forward slice, not the 95% forward+backward slice.
- **DERIVED 95%-forward-reduction scenario:** if effective ANE+boundary cost is 5% of the current MLX
  forward (`r + h/0.78 = 0.05`), total speedup is `1/(0.22+0.039) = 3.86100x`. Against the settled
  20.1974 ms MLX forward, the ANE+boundary budget is <=1.00987 ms/pair. This is a requirement, **not a
  measured claim**.
- **Current measured composition:** **REFUSED** because `T_ANE,fwd=null`, placement is unconfirmed, and
  boundary/VJP cost is null. The measured-only Amdahl equation correctly cannot be instantiated.

For an **exact** gradient obtained by recomputing the MLX forward before MLX backward:

`T_new/T_old = 1 + 0.78*r + h`, hence speedup is `<1` for every positive ANE/boundary cost. ANE then adds
work; it is not the optimum.

Per-step CoreML round trips carry Python call, MLMultiArray/Image conversion, synchronization, and tensor
copy/cast costs. Batching may amortize those costs, but it cannot batch across sequential optimizer states.
Only already-admitted within-step accumulation pairs may be batched, and batch placement/fidelity must be
measured separately. Detached monitoring is naturally batchable and is therefore the first honest ANE use.

### Fidelity regime

- **Verdict:** exact CPU/NumPy authority only.
- **fp16 training:** settled final-output cosine 0.9999586 and 3,895/117M class flips, but no speed win.
- **W8A8/custom VJP training:** n600 global and minimum-pair input-gradient cosine >=0.99 plus the speed
  gate; output class/margin custody is also required.
- **Why training tolerance is legitimate but bounded:** #456 measured the same 15/600 pair hashes changing
  between 1-thread and 6-thread CPU reductions, including a 2.384e-7 razor-tie margin, and the operator
  accepted 1-thread as the training teacher while retaining exact verdict authority. That precedent permits
  scoped reduction-order tie drift in training; it does not authorize unknown ANE placement, null gradient
  fidelity, or arbitrary flips.

## Triality and apparatus

- **DSL:** no trainer flag is invented. All new candidates are tool/backend-level and recorded with
  `dsl_na_reason`; any admitted runtime path must become a typed default-off backend policy with resume
  persistence before launch.
- **DAG:** `.omx/research/precision_backend_matrix_DAG_FEED_20260713.md` carries the executable gates and
  six-hook wire-in. No shared hot DAG file was edited.
- **Equations:** no new law closes because the pivotal ANE/im2col timings are null. Reuse
  `heterogeneous_overlap_seconds` only for independent forward monitoring; its docstring explicitly excludes
  a missing VJP. Reuse `amdahl_measured_disjoint_wall_split_with_async_cpu_verdict_v2` for final composition;
  it fails closed on non-measured inputs. The symbolic bounds above are DERIVED preregistration, not a new
  empirical equation.
- **Pool:** buildable rows are appended through `record_candidate` with the canonical hyphenated statuses;
  none is marked measured.
- **Pointer delta:** zero. These are throughput means, no archive/evaluator row.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`;
`.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`;
`.omx/research/SPEC_v8_perclass_decomposition_20260708.md`;
`.omx/research/int8_training_rungs_20260713.md`;
`.omx/research/cheapen_real_95_tilehalo_fp16_20260713.md`;
`.omx/research/cheaper_exact_forward_transfer_95kill_20260713.md`;
`.omx/research/kernel_stack_sweep_443_20260711.md`;
`.omx/research/n205_mlx_metal_new_kernel_plan_20260702.md`;
`.omx/research/mlx_metal_252_execution_20260703.md`;
`.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`;
installed MLX 0.31.2 Python sources; installed PyTorch 2.12.1 dispatcher; installed Xcode macOS 26.4 SDK
headers; current official MLX/Apple/PyTorch documents linked above.

## Durable receipts

| Artifact | Bytes | SHA-256 | Authority |
|---|---:|---|---|
| `experiments/results/precision_backend_matrix_20260713/cpu_sdk_support.json` | 14,198 | `b309b5c5b5025a017b9a5d7bf69f0ee9448afdb5ec02ac2d6babeca54e12a5ba` | measured CPU/SDK existence semantics |
| `experiments/results/precision_backend_matrix_20260713/im2col_quantized_stem.json` | 2,698 | `45156f4b34c14b82d259224425993285d3f77070d45384bc6e1dd8aec0ac148f` | Metal blocker; no timing/fidelity |
| `experiments/results/precision_backend_matrix_20260713/a3_quality_n600.json` | 2,383 | `33532dcebcf6a43425ecaf48a3e9d2fdea9b11d9e2e752a196ac60d34f302ecc` | n600 Metal blocker; no timing/fidelity |
| `experiments/results/precision_backend_matrix_20260713/a2_coreml_ane.json` | 3,555 | `53a8574f992b2b44385f0584e6d1371e6c33f7feed40f64d984990f186710a56` | DNS/install blocker; no ANE claim |

No paid/provider dispatch, training step, run-dir mutation, score evaluation, or frontier promotion occurred.
