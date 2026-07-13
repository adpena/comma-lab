# Int8 training rungs: frozen-teacher W8A8 and train-native witness QAT

Date: 2026-07-13  
Lane: `lane_int8_training_rungs_20260713`  
Mode: `research_only=true`; local probes only; **no training launch, provider dispatch, evaluator run, or pointer movement**

## Executive verdict

| rung | result | evidence label | verdict scope |
|---|---|---|---|
| A1 MLX-native int8 Conv | **NO** in installed MLX 0.31.2 | `SOURCE_VERIFIED` | public installed Python/API surface; not an int8-family kill |
| A2 CoreML/ANE W8A8 | **latency NOT MEASURED; compile ticket** | `BLOCKED_NOT_MEASURED` | this contained interpreter lacks `coremltools`; ANE capability remains open |
| A3 W8A8 quality/gradient | **flips/cosine NOT MEASURED; runnable n600 ticket** | `BLOCKED_NOT_MEASURED` | contained process has no Metal device; no zero or borrowed fp16 metric |
| B post-hoc witness int8 gap | **`-0.0004301114 d_seg` (int8 improved)** | `MEASURED_N600` | exact v7.5.2 EMA, parsed LVLS1, first real n600, Seg-only macOS advisory |
| B QAT-native arm | **TICKETED, default OFF** | `UNMEASURED_TICKET` | achieved recovery is owed a paired finishing-stage A/B |

The central systems result is that “frozen teacher” does **not** mean “no backward.” Scorer weights are
frozen, but the witness update needs the scorer-input cotangent `J_S(x)^T dL/dS`. ANE forward-only can
offload monitoring/verdict work. It cannot replace the 95%-dominant training teacher unless an exact
recomputation or a measured custom VJP supplies that cotangent.

## Recalled anchor: why int8 was the next rung

The immediately prior fp16 receipt is not re-derived here. It measured global gradient cosine
`0.9999586469813505`, 3,895 / 117,964,800 SegNet argmax flips (`0.0033%`), and forward+backward speedup
`0.9915227256787699`; quality passed and M5-host speed failed. Those numbers are **MEASURED** in
`experiments/results/mlx_precision_probe_local_20260713/receipt.json`, not transferred to int8. They only
justify testing the next precision rung with the same bars.

## A1 — installed MLX native int8 convolution support

### Source inspection

The durable audit is `experiments/results/int8_training_rungs_local_20260713/backend_support.json`.

- Installed package: MLX `0.31.2` (**MEASURED from package metadata**).
- `mlx.core.quantized_matmul` exists.
- `mlx.nn.layers.quantized` defines `QuantizedLinear` and `QuantizedEmbedding`.
- `mlx.nn.layers.quantized.quantize()` only transforms modules that expose `to_quantized()`.
- the installed convolution layer has no `to_quantized()` and there is no `QuantizedConv1d/2d/3d` class.

**Verdict A1: `NO_NATIVE_QUANTIZED_CONV` for MLX 0.31.2.** Raw integer dtype acceptance or hand-lowering a
convolution to im2col + quantized matmul would be a different formulation; neither is the supported native
quantized-convolution path requested here. The reformulation queue retains (1) hand-lowered selected 1x1
convolutions, (2) custom Metal kernels, and (3) CoreML/ANE. No global int8-family negative is issued.

## A2 — heterogeneous CoreML/ANE teacher

### What was measurable locally

`coremlcompiler` is present at the Xcode toolchain path, while `coremltools` is not importable in the main,
system, or inspected local environment. The contained offline environment therefore cannot convert,
activation-calibrate, compile, or time a W8A8 SegNet. **ANE forward latency is NOT MEASURED.** The historical
#88 backend receipt established that FP32 CoreML SegNet can be argmax-faithful (0 flips / 1,966,080 pixels
on its 10-frame probe), but its per-call Python `.predict()` row was latency-dominated and was not an ANE
W8A8 throughput measurement. It is provenance for feasibility, not a borrowed latency.

### Exact gradient split

Let `x(theta)` be the receiver-realized witness frame and `S` the frozen SegNet:

`L(theta) = ell(S(x(theta)), y)`

`dL/dtheta = (dx/dtheta)^T * J_S(x)^T * dell/dS`.

There are three distinct paths:

1. **ANE monitoring/verdict forward — exact use.** Run `S_ANE,W8A8(x_detached)` for periodic loss/argmax
   monitoring. No gradient is requested. This can overlap with independent GPU witness work and frees GPU
   memory/compute during that monitoring call. It does not accelerate the gradient-bearing teacher step.
2. **ANE forward + GPU fp32 recomputation — exact gradient, weak systems win.** Use ANE logits for the reported
   forward, then recompute the scorer on MLX fp32 to obtain the VJP. This still pays the GPU scorer forward as
   part of backward/recomputation; it cannot honestly claim a 95%-kill.
3. **ANE forward + MLX QDQ custom VJP — approximate training treatment.** Use declared forward logits
   `z_ANE`; obtain `dell/dz` at those logits; apply the input VJP of the W8A8-QDQ/fp32-accum MLX surrogate.
   Admission requires A3 n600 global and minimum-pair cosine bars plus measured end-to-end speed. If CoreML
   and MLX quantization groups/logits differ, the mismatch is part of the treatment and must be measured.

For the independent monitoring case only:

`T_step = max(T_GPU,witness, T_ANE,forward) + T_sync`.

For the custom-VJP case, the honest expression includes the MLX VJP:

`T_step = max(T_GPU,witness, T_ANE,forward) + T_MLX,QDQ,VJP + T_sync`,

unless a measured schedule proves overlap with that VJP too.

### CoreML/ANE measurement ticket

The ticket is **YES** and remains local/default-OFF:

1. convert the frozen upstream SegNet to `mlprogram`;
2. calibrate activation ranges on the same 600 real scorer inputs;
3. apply int8 activation quantization and int8 weight quantization;
4. compile and load a persistent model with `ComputeUnit.CPU_AND_NE` (not `ALL` or CPU/GPU ambiguity);
5. confirm Neural Engine placement from the compiled plan/profile;
6. warm, then report median/p05/p95 forward latency and argmax flips against the exact fp32 scorer inputs;
7. compare with the measured MLX-fp32 median `20.1974 ms` and run a concurrent GPU-witness overlap probe;
8. preserve the compiled model hash, calibration-index hash, OS/CoreML versions, compute units, and all timings.

Apple’s current Core ML optimization API exposes activation linear quantization followed by weight linear
quantization for W8A8, and `CPU_AND_NE` explicitly excludes GPU. That API capability is **INFERRED as the
intended compilation route**; whether this SegNet maps efficiently on this M5 is **NOT MEASURED**.
Primary references: [Core ML optimization API](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html)
and [Core ML compute-unit enum](https://apple.github.io/coremltools/source/coremltools.models.html).

## A3 — W8A8 quality and scorer-input gradient

### Landed probe

- runnable tool: `tools/probe_mlx_real_n600_int8.py`
- instrumentation: `tac.local_acceleration.mlx_int8_teacher_fakequant`
- receipt: `experiments/results/int8_training_rungs_local_20260713/a3_quality_n600.json`

The policy is explicit: per-operator symmetric int8 weight QDQ, dynamic per-operator-input symmetric int8
activation QDQ, float32 convolution accumulation, and identity STE through QDQ. Standard Conv/Linear leaves
use one weight scale per operator tensor; the explicit SegNet spatial head uses one scale per stored kernel
slice because that is its actual adapter representation. Bias and normalization arrays remain fp32. The
instrumentation receipt refuses zero wrapped operators, preventing the prior “input-only cast” ambiguity.

The tool restores the exact checkpoint SHA
`ef2c097f98f74dbd16e77c6f7b60f05e0a630b6bd65ee55bf334336c4549c965`, uses the same real-state loader and
the same bars as fp16, performs no optimizer step, and saves one atomic row per pair for n600 resumability.

### Local result

Metal preflight failed with `[metal::load_device] No Metal device available` in this contained process.
Therefore:

- argmax flips: **NOT MEASURED**;
- global/min-pair gradient cosine: **NOT MEASURED**;
- speedup: **NOT MEASURED**;
- verdict: `NO_VERDICT_BLOCKED`, not `NO_GO` and not a zero-valued result.

Exact rerun ticket on a Metal-entitled local process:

```bash
.venv/bin/python tools/probe_mlx_real_n600_int8.py \
  --quality-pairs 600 --timing-pairs 8 --warmup 1 --repeats 3 \
  --out experiments/results/int8_training_rungs_local_20260713/a3_quality_n600.json
```

The conjunctive equation is:

`GO_A3 = [n=600] [C_global>=0.99] [min_i C_i>=0.99] [T_fp32/T_int8>=1.5]`.

QDQ emulation timing is labeled as emulation overhead; it cannot establish native int8 or ANE speed. If
quality fails, the scoped reformulation is an operator-wise precision waterfill (or W8A16 relaxation), not
an int8-family kill.

## B — train-native int8 witness QAT

### Heritage and exact deploy grid

- PR95’s historical Stage 4 used 500 epochs at `lr=1e-4` with per-tensor symmetric int8 fake quantization,
  `n_levels=127`, and STE (**source-inspected heritage**, not a result on this witness).
- `tac.quantization.FakeQuantSTE`, `tac.torch_vehicle.score_aware_qat._fake_quantize_n`, and the variable-grid
  `tac.frontier_int5_qat` machinery establish reusable fake-quant patterns.
- the current level-set receiver’s actual ship grid is `absmax/127` independently per learned tensor; the
  deterministic `B`/`*_B` banks are rule-118-free and excluded. The code table is also quantized on its own
  per-tensor grid.

### Receiver-closed post-hoc gap

The probe `tools/probe_int8_witness_byteclose_gap.py` does not compare a proxy or the trainer’s already-int8
verdict with itself. It builds an actual LVLS1 payload, parses it through the canonical parser, proves parsed
arrays equal `int8_dequant_params`, and compares:

- control: preserved fp32 EMA weights/codes;
- treatment: parsed LVLS1 int8-dequant weights/codes;
- common path: NumPy witness -> real R -> frozen CPU-torch SegNet argmax;
- coverage: first 600 real cached pairs, one-thread exact-forward law;
- fp32 EMA `d_seg`: **`0.03752662658691406`** (`MEASURED_N600`);
- parsed int8 `d_seg`: **`0.037096515231662325`** (`MEASURED_N600`);
- signed gap `int8 - fp32`: **`-0.00043011135525173466`** (`DERIVED_FROM_MEASURED`);
- Seg score-unit gap `100 * Delta`: **`-0.043011135525173466`**;
- per-pair sign: int8 improved `415`, worsened `182`, tied `3`; median pair gap `-0.0004679362`;
- arm-to-arm SegNet argmax changes: `474,649 / 117,964,800 = 0.0040236494` (about `0.4024%`);
- receiver packet: `83,796`-byte LVLS1 blob inside an `83,093`-byte deterministic zip; parser/direct-QDQ
  equality passed.

The signed equation is:

`Delta_dseg_post8 = d_seg(R(Q8(W_fp32))) - d_seg(R(W_fp32))`.

`max(0, Delta_dseg_post8)` is only an upper bound on the aggregate positive QAT recovery prize. It is not a
prediction that QAT recovers all of it. A negative gap means post-hoc int8 helped aggregate d_seg on this
checkpoint and leaves no positive aggregate gap to “close”; it does not prove QAT cannot improve another
facet or checkpoint.

Here the measured positive recovery ceiling is therefore **zero**. The user premise that the current
post-hoc cast “costs Delta d_seg” is **FALSIFIED for aggregate d_seg on this exact v7.5.2 checkpoint**. The
quantizer is acting as a beneficial projection/regularizer on this receiver cell. This is not a score win:
`d_pose` was not measured and the archive-rate term is unchanged relative to the existing int8 ship grid.

### Default-OFF QAT A/B ticket — YES, but no longer a positive-gap recovery arm

The typed stub is `Int8WitnessQATProposal` in `tac.witness_dsl.int8_training_rungs_policy`. It is structurally
OFF/unwired, emits no argv, and refuses enablement. The future parser-backed arm must:

- start both arms from the same preserved stage-boundary EMA + optimizer state and seed;
- preserve the control: current fp32 finishing stage, then canonical post-hoc LVLS1 int8;
- treatment: finishing-stage-only FakeQuantSTE on **the exact LVLS1 per-tensor absmax/127 grid**, including
  the code table and all counted learned parameters, excluding only receiver-regenerated free banks;
- quantize in every treatment forward, keep an fp32 master/optimizer state, and save the EMA fake-quant-ready
  shadow at each declared stage checkpoint atomically;
- keep the loss and scorer authority unchanged; QAT acts on witness parameters, not frozen scorer authority;
- compare parsed receiver n600 `d_seg`, `d_pose`, exact archive bytes, and component score. Admission is by
  receiver cells and bytes, never training proxy improvement alone;
- preserve a terminal treatment checkpoint even on `NO_GO` and attach `verdict_scope` plus reformulation queue.

No live trainer flag exists today, so this memo intentionally does not invent one.
Because B found no positive aggregate d_seg gap, the QAT arm is a confirmatory/local-basin experiment rather
than a promised `0.000430` recovery. It should be lower priority than closing A3 or the ANE custom-VJP ticket
unless a joint Seg/Pose/bytes treatment supplies a new positive prize.

## Canonical equations, DAG, and DSL triality

- Equations: `src/tac/canonical_equations/int8_training_rungs_20260713.py`
  - `int8_teacher_w8a8_admission_v1` — conjunctive n600 cosine/speed gate;
  - `int8_witness_posthoc_gap_v1` — fail-closed terminal-receipt loader and signed QAT-prize equation;
  - `heterogeneous_overlap_seconds` — independent-forward overlap only, explicitly excluding a missing VJP.
- DAG: `.omx/research/sub015_DAG_int8_training_rungs_20260713.md`.
- DSL: `src/tac/witness_dsl/int8_training_rungs_policy.py`; teacher and witness-QAT stubs default OFF,
  `wired=false`, and `live_trainer_argv=[]`.
- Focused tests: six passing tests across equation receipt validation and default-OFF DSL invariants.

The equations are code-landed but not appended to the shared registry in this uncommitted main-review lane.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`.
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` §8 and
  `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`.
- `reports/latest.md`, `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, and current
  canonical pointer/ledger surfaces required by preflight.
- prior fp16 memo/receipt named above.
- #88/CoreML backend memos:
  `.omx/research/scorer_backend_benchmark_ours_vs_others_20260611.md` and
  `.omx/research/mlx_scorer_port_fidelity_speed_optimization_20260627.md`.
- PR95/QAT sources named above.
- sibling #336 memo `.omx/research/witness_sensitivity_bitalloc_336_20260713.md` read for boundary only;
  **no #336 file was edited, imported as evidence, or used to substitute a v7.5.2 measurement**.

## Pointer-delta honesty

Pointer delta: **ZERO**. Run directory mutation: **ZERO**. Training/evaluator/provider launch: **ZERO**.
The small local packet and JSON receipts live under
`experiments/results/int8_training_rungs_local_20260713/`. The first 48-pair B partial was preserved under
`rejected_source_drift/` and explicitly rejected after the probe source was formatted during execution; the
source-frozen n600 run restarted from pair 0 with a source-drift guard. No score or promotion claim is made.
Main advanced through unrelated commits during the long probe. The terminal receipt records the end head;
`b_posthoc_gap_n600.custody.json` derives the start head from reflog + measured elapsed time and verifies that
all imported measurement dependency paths were unchanged between heads and clean afterward. This distinction
is explicit rather than collapsing start/end custody into one SHA.
