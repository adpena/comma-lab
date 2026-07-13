# DAG FEED — frozen-SegNet precision x backend matrix — 2026-07-13

`research_only=true` · `score_claim=false` · `pointer_moved=false` · `$0 local` ·
lane `lane_precision_backend_matrix_20260713` · shared hot-DAG append `DEFERRED-MAIN`

## Executable dependency graph

```text
installed API / SDK / dispatcher audit
  ├─ MLX: floating Conv2d + weight-quantized matmul; no QuantizedConv2d
  ├─ MPSCNN: UInt8 weight storage -> fp16 convolution
  ├─ MPSGraph: Q/DQ + floating convolution; no admitted integer conv
  ├─ CoreML: W8A8 int8-int8 candidate on M5-class NE
  └─ CPU-torch: floating conv + QuantizedCPU W8A8 + RAW-ONLY int16
        ↓
RUNG B1: exact-stem im2col + MLX quantized_matmul
  ├─ B={1,8} speed vs native fp32 conv
  ├─ W8 / W8A8-QDQ / W6 / W4
  └─ exact frozen-SegNet real-state output fidelity
        └─ BLOCKED-NOT-MEASURED: no Metal device in this process

RUNG A3: exact frozen-teacher W8A8-QDQ n600
  ├─ global and min-pair input-gradient cosine >= 0.99
  ├─ measured speedup >= 1.5x
  └─ required n600 coverage
        └─ BLOCKED-NOT-MEASURED: no Metal device in this process

RUNG A2: CoreML frozen SegNet -> CPU_AND_NE
  ├─ convert exact model at B={1,8}, NCHW [B,3,384,512]
  ├─ compute-plan/device-usage proof of NE placement
  ├─ real-state argmax fidelity
  └─ warm B=1 and best-batch latency vs CPU/MLX
        └─ BLOCKED-NOT-MEASURED: scratch coremltools fetch failed on DNS

{A2 placement + latency + fidelity} AND {A3 gradient gate}
  -> candidate ANE W8A8 forward + MLX custom VJP
  -> measured disjoint Amdahl composition
  -> typed default-off backend policy + additive resume persistence
  -> governed launch eligibility

Any missing edge -> REFUSE wall-clock composition and REFUSE trainer actuation
```

## Node dispositions

| Node | Status | Authority / verdict scope |
|---|---|---|
| `precision_native_float_conv` | `EXISTS` | fp32/fp16 MLX/MPS/CPU; bf16 MLX/CPU, with MPSGraph bf16 execution unverified here |
| `precision_mlx_quantized_conv` | `ABSENT-PUBLIC-NATIVE__BUILDABLE-LOWERING` | installed MLX 0.31.2; im2col + qmatmul remains open |
| `precision_mpscnn_uint8` | `STORAGE-ONLY` | UInt8 weights dequantize to fp16; no W8A8 compute claim |
| `precision_int16_teacher` | `NO-GO-USABLE-PATH__RAW-CPU-PRIMITIVE-EXISTS` | current frozen-SegNet neural-conv formulation across audited backends; custom integer math remains buildable but unmotivated |
| `precision_cpu_w8a8_stem` | `MEASURED-EXISTENCE-ONLY` | real pixels + synthetic stem weights; no frozen-SegNet fidelity transfer |
| `precision_mlx_im2col_stem` | `BUILT-NEVER-FIRED` | device preflight blocked; exact command preserved |
| `precision_ane_w8a8_forward` | `NEEDS-BUILD` | conversion/placement/latency/fidelity all null; no ANE negative |
| `precision_ane_mlx_vjp` | `NEEDS-BUILD` | requires both A2 and A3; no exact-gradient claim |
| `precision_custom_metal_fp16_conv` | `NEEDS-BUILD__SIBLING-IN-PROGRESS` | sibling `lane_custom_metal_conv_20260713` owns pointwise/depthwise source; still requires a completed measured prize vs native MLX fp16 |
| `precision_custom_metal_w8a8_conv` | `NEEDS-BUILD` | reopen after im2col/full-teacher signal and explicit accumulator/VJP policy; do not collide with sibling source |
| `precision_heterogeneous_wall` | `REFUSED-INCOMPLETE-MEASUREMENTS` | A2 latency, placement, boundary, and VJP cost are null |

## Composition law and closed bounds

For the operator-set normalized wall split `(forward, backward, other)=(0.78,0.17,0.05)`, an admitted
ANE-forward + no-recompute MLX VJP has

`T_new/T_old = 0.22 + 0.78*(T_ANE/T_MLX) + h`.

This yields a **DERIVED** 4.54545x zero-cost upper bound and a **DERIVED conditional** 3.86100x total
speedup if forward+boundary cost is 5% of current forward. No measured composed row exists.

Exact MLX recomputation has

`T_new/T_old = 1 + 0.78*(T_ANE/T_MLX) + h > 1`,

so it cannot be the optimal training assignment. This is arithmetic closure, not a new calibrated equation.
The canonical measured-only consumer remains
`amdahl_measured_disjoint_wall_split_with_async_cpu_verdict_v2`; it refuses the current null inputs.
`heterogeneous_overlap_seconds` is valid only for independent monitoring and explicitly excludes a missing
VJP. No new canonical-equation registry row is minted.

## Six-hook wire-in

- **Sensitivity map:** A3's per-pair input-gradient cosine and full-output class flips are the admission
  signal; no score sensitivity is inferred from CPU synthetic-weight stem timing.
- **Pareto constraint:** optimize training wall subject to n600 gradient fidelity, real-state class fidelity,
  exact CPU verdict custody, and zero archive/score authority.
- **Bit allocator:** N/A-with-reason; teacher precision changes compute, not witness/archive bytes. If later
  used for a shipped scorer surrogate, a separate counted-byte contract is mandatory.
- **Cathedral/autopilot:** `REFUSE` until A2+A3 pass and a typed resumable backend policy exists. No launch
  argv is emitted by this lane.
- **Continual learning:** memo, this FEED, four content-addressed receipts, and candidate-pool rows preserve
  the signal.
- **Probe disambiguator:** B1 decides im2col vs custom Metal; A2 B=1 vs B=8 separates boundary amortization;
  A2+A3 separates detached monitoring from approximate gradient-bearing use.

## Triality and custody

- **DSL:** N/A-with-reason today; backend tools are not trainer flags. Future admitted path must be typed,
  default-off, and resume-persisted before launch.
- **DAG:** this file.
- **Equations:** existing measured-only laws reused; no new law calibrated.
- **Receipts:** `experiments/results/precision_backend_matrix_20260713/`.
- **Memo:** `.omx/research/precision_backend_matrix_20260713.md`.

No paid/provider dispatch, training launch, exact evaluator, sacred run-dir write, or frontier-pointer move.
