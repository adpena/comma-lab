# Task #494 — maximum-throughput compute-substrate authority ladder

**UTC:** 2026-07-14T03:10:02Z  
**Lane:** `throughput_authority_ladder`  
**Task:** `#494`  
**Status:** `BUILT; FIXED+DYNAMIC_N600_MEASURED; INT64_CEILING_N600_RUNNING; HOST_GATES_OWED`  
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
   candidate**, but retain one-thread CPU-Torch as the automatic fallback;
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
| 2b — dynamic max-absolute forward | **MEASURED NO-EXACT-ARM THROUGH W24** | `INSTANCE`: same real n600 SegNet surface, W16/W18/W20/W22/W24, label-free runtime scale | full exact-row/hash receipt; W20 first training-tolerance arm; W24 leaves 19 flips | finite W25/W26 ceiling check is running; W26 is the last uniform precision with an exact single-int64 static bound |
| 3 — fixed-point verdict substrate | **BUILT; HOST MEASUREMENT OWED** | custom direct NHWC dense/grouped/depthwise Conv2d, exact int64 MAC, dynamic/fixed scale | all 125 Conv2d replacement, NumPy integer reference, cross-process n600 harness | exact argmax + zero uncertified + one candidate digest across 10 processes + speedup >1 |
| 3 — ANE | **PUBLIC-API FORMULATION BLOCKED** | CoreML 9 public activation compute exposes W8A8; settled W8A8 PTQ failed 45.836809% held-out flips | settled-state-aware ticket compiler; refuses duplicate W8A8 and unrepresentable higher-bit requests | a genuinely distinct W8 formulation, or a public higher-bit ANE compute surface with proved placement |
| 4 — integer render-R backend | **BUILT; HOST MEASUREMENT OWED** | four axes, Q15 weights, Q7/Q5 state, no atomics, exact int32 gather | default-off VJP backend, static overflow proof, exact NumPy-state hash gate, n600 matched benchmark | full-R source receipt + exact int-state parity + repeat identity + bounded fp32 error + speedup >1 |
| 4 — integer megakernel | **UNREFUTED DISTINCT FORMULATION** | integer/exact reductions only | the integer R kernel establishes the first exact component | #356 refuted fp32 reorder/fusion; no graph-wide integer lowering or speed receipt exists |

Every negative above is scoped. Fixed calibration W24 failing does not kill dynamic scaling, mixed
precision, integer convolution, Metal, ANE, or the authority-ladder paradigm. A direct-int64 Metal
speed loss would kill that kernel formulation only; exact limb/tensor decomposition and per-layer
precision allocation would remain open.

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
66 bits and exceeds signed int64. A separate resumable W25/W26 full-n600 receipt is therefore the
finite **single-int64 ceiling check**, not an unbounded resweep. Its final result will decide whether
the built uniform direct-int64 Metal formulation remains admissible or yields to mixed precision,
limb accumulation, or a correction ladder.

Pair-0 smoke (**MEASURED, INSTANCE only**) was:

| Arm | flips | uncertified pixels |
|---|---:|---:|
| W16 | 8 | 126 |
| W18 | 2 | 40 |
| W20 | 0 | 5 |
| W22 | 0 | 1 |
| W24 | 0 | 0 |

No native-integer latency claim follows from QDQ. With no exact QDQ arm yet, the custom Metal
command fails closed before speed measurement; it cannot launder a tolerance-only arm into authority.

## Op × substrate × precision authority-throughput assignment

| Operation | Current/default substrate | Precision | Authority grade | Final assignment / gate |
|---|---|---|---|---|
| witness forward/backward | MLX/Metal | fp32 | training signal | **ACTIVE**; keep portable NumPy/Torch parity surfaces |
| frozen teacher used for gradients | MLX/Metal | fp32 | training signal | **ACTIVE**; do not confuse with local verdict wall |
| render-R forward | MLX/Metal | fp32 | training signal | **ACTIVE**; NumPy-fp32 receiver remains reference |
| render-R adjoint | fixed-order custom Metal fp32 today | fp32 | training signal | candidate Q15/int32 gather only after exact n600/parity/speed gate |
| SegNet local verdict | CPU-Torch one thread | fp32 | local deterministic reference | automatic fallback; custom Metal may become a default-off candidate only after full conjunction |
| SegNet candidate verdict | custom Metal | receipt-selected dynamic WnAn, int64 MAC | local candidate filter | **HELD/OWED** until dynamic QDQ + exact/certified/cross-process/positive-speed receipt |
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
       SegNet = admitted custom-Metal dynamic fixedpoint
                OR one-thread CPU-Torch fp32 fallback
       PoseNet = live CPU canary at index 0 and every K while frozen;
                 labelled banked value otherwise;
                 always live after pose engagement
  -> controller consumes source-labelled telemetry only
  -> candidate archive / NumPy-fp32 byte-close
  -> exact contest CPU and separately contest CUDA replay on identical bytes
```

There is no silent fallback. The policy compiler binds custom Metal to the exact QDQ receipt
fingerprint, selected bit width, scale mode, exact n600 custody, interval certificate, one candidate
digest across processes, and positive speed. Any missing conjunct returns to CPU.

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
`PUBLIC_ANE_PRECISION_UNREPRESENTABLE`. That is a **FORMULATION/API** negative, not an ANE-family
negative. CoreML fp32 remains a forward-only advisory route; #490 established zero backward
selectors.

## Rung 4 and #356

Integer addition is reorder-independent only with exact partial sums, no overflow/saturation, and
one deterministic finalization. The integer-R build proves these conditions per stage and compares
the raw final int32 state against the NumPy authority hash for every real frame. It does not make
the score faster: render-R is negligible in the n24 single-call timer, while the slow local verdict
is forward-only CPU-Torch.

#356's megakernel negative was for fp32 whole-step reorder/fusion. An integer-lowered megakernel is
a distinct, unrefuted formulation, but it is not build-authorized by analogy. Every reduction and
nonlinearity in a claimed fusion domain needs an exact/bounded lowering plus a matched speed receipt.

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
  `interval_argmax_enclosure_certificate_v1`. Registration is append-only and deferred to MAIN
  after complete host receipts.
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
  admission bug, and Metal legacy-cache overwrite bug are durable typed guards/tests, not chat-only
  observations.

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
