# ANE-Unlock Ecosystem Survey — 2026-07-13

**Charter:** operator-directed online survey of the "ANE unlock" community (Anemll + broader ecosystem) to feed
the live `ane_unlock_correction` sibling arm (correction ladder R0→R5: error decomposition → precision-split
ANE trunk + fp32 head → calibrated corrector → band-tile w/ donated SE stats → W8A8).
**Our anchor numbers (MEASURED, main-local):** CoreML `CPU_AND_NE` fp16 EfficientNet-B2 SegNet forward =
**38.03× faster** than 1-thread CPU-torch (9.1 ms vs 346 ms @ 1×3×384×512); fidelity = **2.47% argmax flips**
vs fp32.
**Labels:** MEASURED = number from a cited external source or our own bench · DERIVED = follows from cited
mechanisms · INFERRED = plausible reading of cited material · ASSUMED = unverified.
**Status:** research memo, uncommitted, no code changes. `research_only=true` until the sibling arm consumes it.

---

## 0. Executive orientation

The community's "ANE unlock" has three distinct fronts, all relevant to us:

1. **CoreML-route optimization** (Anemll, WhisperKit, ml-stable-diffusion, coremltools guides) — everything we
   can adopt *today* on our existing CoreML teacher path.
2. **Hardware characterization** (maderix, Orion arXiv:2603.06728, arXiv:2606.22283) — the measured physics of
   the ANE: what is fast, what is bandwidth-bound, what fp16 does numerically.
3. **Private-API direct programming** (Orion, maderix, ANEForge) — bypassing CoreML entirely. High ceiling
   (delta weight reload, guaranteed ANE residency) but private-API fragility; NOT recommended for our frozen
   teacher (our weights never change → CoreML's bake-at-compile model costs us nothing).

Key hardware facts that frame every technique below (all MEASURED by maderix/Orion on M4 Max, INFERRED to
carry to M5-class):

| Fact | Value | Source |
|---|---|---|
| ANE fp16 peak (M4 Max, 16 cores) | ~19 TFLOPS (Apple's "38 TOPS" is INT8 marketing; INT8 weights are dequantized to fp16 before compute) | Orion §2.1 + maderix ([substack](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)) |
| On-chip SRAM | ~32 MB; exceeding it → **−30% throughput** (DRAM spill cliff) | Orion Table 1; maderix: 24 MB working set = peak, 96 MB = −30% |
| Dispatch overhead | ~0.095 ms per dispatch (XPC+IOKit) | Orion Table 1 |
| Utilization vs graph depth | single op ~30% util; deep graphs (16–64 ops) ~94% | Orion §3 / maderix |
| 1×1 conv vs matmul | conv formulation ~3× faster (ANE is "fundamentally a convolution engine") | Orion #17; maderix |
| fp16 range | ±65,504; overflow → NaN cascades; fix = clamp / weight pre-scaling | Orion §7.2; Anemll GEMMA3_FP16_SCALING |
| Batch mode | ~55% average gain from batching (ANE exploits parallelism across concurrent tensor ops) | arXiv:2606.22283 (via search abstract — MEASURED there, not independently re-verified by us) |
| ANE idle power | zero (hard power-gated, own power domain) | Orion Table 1 |

DERIVED for our teacher: EfficientNet-B2 @ 1×3×384×512 is a deep conv graph (hundreds of fused ops) → we are
already in the ~94%-utilization regime, and fp16 weights (~31 MB) sit at the edge of the 32 MB SRAM budget —
weight compression (palettization/W8A8) is therefore an *SRAM-fit lever*, not just a bandwidth one.

---

## 1. Per-technique table

Ladder legend: **R0** error decomposition · **R1** precision-split (ANE trunk + fp32 head) · **R2** calibrated
corrector · **R3** band-tile w/ donated SE stats · **R4** (reserved / band-tile variants) · **R5** W8A8.

| # | Technique | Source | Mechanism | Transferability to our conv teacher | Feeds rung |
|---|---|---|---|---|---|
| T1 | **Enumerated-shape multi-function compilation** (`infer_ctx{512,1024,…}` functions in ONE model, selected via `meta.yaml` template) | Anemll [examples/VARIABLE_CONTEXT.md](https://github.com/Anemll/Anemll/blob/main/examples/VARIABLE_CONTEXT.md); [coremltools multifunction docs](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html) | No dynamic shapes on ANE — compile one CoreML *function per discrete shape tier*; runtime picks the function; switching costs "a few milliseconds" (MEASURED: 45.2→42.1 tok/s across tiers, i.e. ~flat) | **DERIVED — direct.** We have no KV cache, but the same mechanism gives us *batch tiers*: compile `fwd_b1 / fwd_b8 / fwd_b32` functions over (B,3,384,512). ANE batching ≈ +55% throughput (§0) and amortizes the 0.095 ms dispatch. Our current 9.1 ms is batch-1. | NEW rung: **R-T (throughput)** |
| T2 | **ANEMLL-Dedup** — surgical weight deduplication across multifunction models (~50% size reduction) | Anemll releases/README ([repo](https://github.com/Anemll/Anemll)) | Multiple compiled functions share one weight blob instead of N copies | DERIVED — required companion of T1 so batch tiers don't triple the ~31 MB weight footprint (SRAM cliff, §0) | R-T |
| T3 | **Chunked multi-stage `.mlmodelc` pipelines** (embeddings / FFN / lm-head as separate CoreML models, composed at runtime) | Anemll conversion pipeline (`convert_model.sh`, `--chunk`, `calc_chunk_split.md`) | Each chunk is an independent CoreML model → each gets its OWN `computeUnits` + `compute_precision`. Anemll itself keeps all chunks fp16 (no fp32 chunks observed — INFERRED from repo docs), but the *mechanism* permits a per-chunk precision/device split | **DERIVED — this is exactly R1's architecture.** Split B2 at the last-stage boundary: trunk chunk → ANE fp16, head chunk (final conv + logits) → CPU/GPU fp32. Community-standard composition mechanics (IOSurface-backed zero-copy between chunks). | **R1** |
| T4 | **Per-op precision selector** — `ct.transform.FP16ComputePrecision(op_selector)` (a.k.a. typed-execution mixed precision) | [coremltools typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html); worked example [ailia/axinc Medium](https://medium.com/axinc-ai/using-mixed-precision-in-core-ml-77c2428ba728) | Callback returns fp16/fp32 per MIL op at conversion; fp32 ops route to CPU/GPU, rest stays on ANE. MEASURED (ailia): fp16 12.23 ms · fp32 35.57 ms (2.9×) · mixed 12.63 ms (**+3.3% cost for fp32 where it matters**) | **DERIVED — the cheapest possible R1.** No model surgery: keep ONE model, mark the final logit conv (+ any offender ops found by R0) fp32 via `op_selector`. Strictly easier than T3; T3 only needed if the op-selector granularity proves insufficient. | **R1** (primary), R0 |
| T5 | **fp16 preflight + weight pre-scaling** (`fp16_compatibility_check.py`, `fp16_preflight.sh`, GEMMA3_FP16_SCALING.md) | Anemll utils | Offline scan of weights/activations for fp16 range violations; rescale weight tensors (with compensating inverse scale downstream) so intermediates fit ±65,504 and use more of the mantissa | DERIVED — run against B2: BN-folded scales and SE-block global-pool→FC paths are the likely precision hot spots. A math-preserving rescale can shrink fp16 error *before* any corrector learns it. | **R0, R3** |
| T6 | **Activation clamping ±65,504 pre-softmax/norm** | Orion §7.2 (arXiv:2603.06728) | `clamp(x, −65504, +65504)` before overflow-prone ops kills NaN cascades without touching well-behaved activations | INFERRED — our flips are argmax jitter, not NaN, so likely inert for us; adopt only if R0 finds ±inf intermediates | R0 |
| T7 | **(B,C,1,S) channels-first layout + 64-byte last-axis alignment + chunked einsum** | Apple [ml-ane-transformers article](https://machinelearning.apple.com/research/neural-engine-transformers) | ANE buffers: last axis unpacked, must be 64-byte aligned; wrong layout = up to 32×(fp16)/64×(int8) memory waste; chunking improves L2 residency + multicore util. MEASURED: 10× speed / 14× memory on DistilBERT | INFERRED — mostly automatic for us: a conv net is already NCHW and coremltools handles layout. Value = *audit* rule: no stray reshapes/transposes in any corrector/head we add on the ANE side | R1, R2 hygiene |
| T8 | **W8A8 with calibration** (activation quantization; "optimized int8 compute on A17 Pro / M4 and later") | [coremltools opt-quantization guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html); maderix ANE repo README (W8A8 35.1 vs 18.6 TOPS ⇒ 1.88×) — **conflicts** with maderix substack + Orion ("no 2× INT8 speedup; INT8 dequantized to fp16") | Reconciliation (DERIVED): INT8 *compute* is dequantized to fp16 on ≤M4-class ANE datapaths, BUT int8 activations halve inter-tile L2/SRAM bandwidth → big wins exactly when bandwidth/SRAM-bound; coremltools claims a genuinely optimized int8 path on A17/M4+. Net: workload-dependent, must be A/B'd. | DERIVED — as designed for **R5**, with two honest caveats: (a) throughput gain on M5 is UNVERIFIED and possibly ~1× if we're compute-bound; the *reliable* win is SRAM fit (31 MB fp16 weights → ~16 MB W8) clearing the −30% cliff margin + batch headroom; (b) it ADDS quantization error on top of the 2.47% flips — must be gated per T10. | **R5** |
| T9 | **LUT4/LUT6 palettization, per-component flags** (`--lut-embeddings`, `--lut-lmhead`) | Anemll conversion recipes; coremltools palettization | Per-layer codebook compression; component-granular so accuracy-critical layers stay fp16 | DERIVED — weight-only sibling of T8: palettize trunk convs, keep head fp16/fp32. Zero activation-error added (weights only). Good SRAM lever if W8A8 flunks fidelity. | R5-alt |
| T10 | **QoI per-example no-regression gating** — fp16 CoreML model is THE reference; every compressed/optimized variant gated on per-example regressions, not dataset-average metric | WhisperKit ([arXiv:2507.10860](https://arxiv.org/html/2507.10860v1); HF model cards) | "per-example no-regressions (qoi), a stricter metric than dataset average WER"; compressed variants accepted within 1% of reference | **DERIVED — direct discipline transfer.** Our analogue: per-pair flip-rate vs fp32 (and vs the R-corrected output) on all 600 pairs, gate every rung on worst-pair regression, not mean. This IS our n600 discipline, community-corroborated. | Acceptance gate for **all rungs** |
| T11 | **Compile-time caching of constant inputs** (WhisperKit "silence caching": output of an all-zero 15 s block precomputed at compile time) | WhisperKit paper §arch | If part of the input is constant, bake its contribution | INFERRED — marginal for us (every frame differs); possible micro-use: static hood-region tiles in band-tile R3 | R3 (minor) |
| T12 | **ANE profiling without Xcode** (`ANE_PROFILER.md`); Xcode performance reports; `powermetrics` ANE counters | Anemll utils; Apple tooling | Per-op device-residency + latency attribution; powermetrics gives ANE power at 50 Hz sampling | DERIVED — required instrument: CoreML's `CPU_AND_NE` is a *request*, not a contract; residency proof + per-op fallback detection needs this. Also the instrument for the concurrency A/B (§3). | R0, R-T, concurrency test |
| T13 | **Delta compilation / weight reload without recompile** (`_ANEModel` unload → patch BLOBFILE → reload, 8.5× faster than recompile; 494 ms/step for 60 kernels) | Orion §5 (private APIs) | Bypasses `ANECCompile()` for weight updates | INFERRED — NOT needed: our teacher is frozen. Becomes relevant ONLY if we ever put a *learnable* module (corrector) on ANE. Private-API fragility. | (R2-future, low) |
| T14 | **LoRA adapter-as-IOSurface-input** (adapters as runtime inputs, not baked weights → hot-swap w/o recompile) | Orion §6 | Weights-as-inputs escape hatch for the bake-at-compile constraint | INFERRED — elegant pattern IF the R2 calibrated corrector lives on ANE and needs periodic refresh: express corrector weights as model *inputs*. Achievable in pure CoreML (extra input tensor), no private APIs needed for a small corrector. | **R2** (design option) |
| T15 | **Prefill/infer function splitting** (4-function rotation: prefill/infer × rotate) | Anemll | Separate compiled functions for throughput-shaped vs latency-shaped phases | INFERRED — LLM/KV-specific; our analogue is just T1's batch tiers | (subsumed by T1) |
| T16 | **SPLIT_EINSUM attention conversion** (`--attention-implementation SPLIT_EINSUM`) | [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion) | Reformulates attention so ANE can run it; ORIGINAL routes to GPU. Community: CPU+GPU/ORIGINAL sometimes *faster* than ANE — compute-unit choice is empirical per device | INFERRED — B2 has no attention (SE blocks only); the transferable lesson is "ANE-routable ≠ ANE-fastest; A/B compute units per stage" | R1 hygiene |
| T17 | **Heterogeneous ANE+GPU pipelines** (NPU prefill → GPU decode, zero-copy IOSurface handoff) | [SqueezeBits Yetter blog](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176); maderix (GPU prefill 6.7–9.7 ms → ANE decode 1.9–2.3 ms zero-copy) | Stage-level device split with shared-memory handoff; MEASURED +40% TTFT vs pure-MLX on iPhone 15 Pro | DERIVED — corroborates our architecture (ANE teacher feeding MLX-GPU training loop via unified memory); note their split is *sequential*, not concurrent | Architecture validation |

---

## 2. What the hardware characterization says about OUR teacher (DERIVED)

- **We are near the right operating point already.** Deep conv graph → ~94%-util regime; 9.1 ms for ~2×10⁹ MAC
  ≈ 0.4 TFLOPS effective… far below the 19 TFLOPS peak, which (per maderix's dispatch/SRAM analysis) means we
  are **latency/dispatch/IO-shaped at batch-1, not compute-bound** → T1 batch tiers are the #1 throughput lever
  (the +55% batch figure and dispatch amortization both point the same way).
- **SRAM budget is the hidden constraint.** B2 fp16 weights ~31 MB + activations already brush the 32 MB cliff;
  batching multiplies activation working set. W8/palettized weights (T8/T9) buy back SRAM before they buy speed.
- **fp16 flip mechanism.** ANE = fp16 multiply datapath with wide accumulator (arXiv:2606.22283 framing). Orion
  measured max logit error 0.073 on GPT-2 with **100% top-1 agreement** — because one token argmax over 32k
  well-separated logits is robust. Our 2.47% pixel flips are the same-magnitude logit noise hitting
  **millions of boundary pixels whose top-2 margin < fp16 error** — i.e. flips concentrate in the small-margin
  annulus (consistent with our margin-field/Fisher picture). DERIVED prediction for R0: flip locations ≈
  low-margin pixels; a margin-thresholded fp32 re-scoring (head-only or tile-only) should capture most of the
  2.47% cheaply. This is the community's per-op mixed-precision insight (T4) specialized by our own geometry.

---

## 3. The concurrency question (ANE teacher ∥ MLX-GPU training)

Evidence table:

| Evidence | Direction | Source |
|---|---|---|
| ANE is dedicated silicon, hard power-gated, own power domain; "ANE inference leaves the GPU and CPU entirely free for other workloads" | PRO | Orion §9 (MEASURED power table; the "free" claim is their framing, not a loaded-GPU test) |
| Single-stream ANE throughput unchanged from 68 GB/s (M1) to 154 GB/s (M5 base) memory bandwidth → single-stream ANE work is dispatch-bound, NOT bandwidth-bound | PRO | pradeep.md (MEASURED) |
| 4 concurrent ANE streams ran "with the GPU and CPU staying free for everything else" | PRO | pradeep.md (observational) |
| iPhone experiment: ANE inference + GPU inference together got **slower** — unified-memory bus contention | CON | [HackerNoon](https://hackernoon.com/i-made-my-iphones-neural-engine-and-gpu-run-inference-together-as-an-experiment-it-got-slower) (paywalled/403 for details; headline + summary only — iPhone-class bandwidth, two heavyweight inference workloads) |
| Yetter NPU+GPU split is sequential (prefill→decode), no concurrent-load measurements published | NEUTRAL | SqueezeBits |
| GPU sustained inference pulls 40–60 W on M3 Max GPU cluster; ANE order-of-magnitude less | PRO (thermal) | pradeep.md |

**Verdict (DERIVED): qualified GREEN.** No published M-series benchmark runs ANE inference concurrently with
GPU *training* under full load — the honest gap — but every mechanism points our way: separate silicon, separate
power domain, ANE demand for our teacher is a few GB/s (9 ms/forward, working set ≲32 MB, mostly SRAM-resident)
against an M5 Max-class ~0.5 TB/s bus, i.e. **single-digit-% bus occupancy**. The one negative result is two
heavyweight *inference* workloads on phone-class bandwidth. Required before adoption (cheap, local, $0):
run the witness trainer at full MLX-GPU load with the ANE teacher looping, and MEASURE (a) teacher ms/forward
solo vs concurrent, (b) MLX step-time solo vs concurrent, (c) `powermetrics` ANE+GPU power. Accept if both
degrade <5% (ASSUMED threshold, sibling arm may tighten).

---

## 4. Fidelity practices — what the community does about ANE fp16 drift

1. **Reference discipline:** WhisperKit treats the *fp16 CoreML model* as the reference and gates every further
   optimization per-example (QoI) — they do NOT chase fp32-parity; they contain regression. Our stricter need
   (fp32-parity of a *teacher signal*) justifies the correction ladder, but the *gating style* transfers (T10).
2. **Per-op fp32 carve-outs** via `op_selector` — the standard fix for overflow/precision-critical ops
   (LayerNorm pow/sqrt in the ailia example; for us: logit head, SE-pool paths). +3% latency, not 3× (T4).
3. **Weight pre-scaling** to fit/center fp16 ranges (Anemll Gemma-3 recipe) — proactive, model-level (T5).
4. **Activation clamping** at ±65,504 (Orion) — NaN insurance (T6).
5. **QAT-for-ANE:** coremltools supports calibration/QAT paths for W8A8; nobody in the surveyed corpus reports
   QAT *specifically to close fp16 argmax flips* on a segmentation model.
6. **Published flip-rate numbers for segmentation/conv on ANE: NONE FOUND.** The only quantitative parity datum
   is Orion's LLM result (max logit err 0.073, 100% top-1 over 64 tokens). Our 2.47%-flip measurement at
   384×512 appears to be **novel territory** — the community has not solved (or even publicly measured) dense
   per-pixel argmax fidelity on ANE. The correction ladder is original work, not a recipe we can import.

---

## 5. Ranked shortlist — 5 highest-EV adoptions

1. **T4 — per-op fp32 via `ct.transform.FP16ComputePrecision(op_selector)`** → R1 without model surgery: mark
   final logit conv (+ R0-identified offenders) fp32 inside the ONE existing model; expected ~3% latency cost vs
   our 38× win (ailia MEASURED analogue). First thing to try; may alone collapse the 2.47% flips into the noise.
2. **T1+T2 — enumerated batch-tier multifunction compile (`fwd_b1/b8/b32`) with weight dedup** → NEW throughput
   rung: ~+55% batch gain + dispatch amortization on top of 38×; teacher forwards are embarrassingly batchable
   across pairs in the training loop.
3. **T10 — WhisperKit-style QoI gate** → adopt per-pair no-regression vs fp32 as THE acceptance metric for every
   ladder rung (n600, worst-pair not mean) — turns the ladder into a gated, community-corroborated protocol.
4. **T5 — Anemll fp16-preflight + weight pre-scaling on B2** → free (offline, math-preserving) fp16-error
   reduction before any learned corrector; directly probes SE/BN-fold hot spots for R0's decomposition.
5. **T8/T9 — W8A8 (calibrated) with palettization fallback** → R5 as designed, reframed: the *dependable* win on
   M5 is SRAM-fit (clear the 32 MB cliff, buy batch headroom), the int8-compute speedup is contested between
   sources and must be A/B'd, and added quant error must pass the T10 gate.

Plus one **cheap prerequisite measurement**: the §3 concurrency A/B (teacher-on-ANE ∥ MLX-GPU training,
powermetrics receipts) — nobody has published it; we need it and it costs ~$0.

## 6. Honest gaps (what the community has NOT solved)

- **No published dense-argmax/segmentation fidelity numbers on ANE** (flip rates, boundary-pixel behavior).
  Our correction ladder is unprecedented in the surveyed corpus, not an import.
- **No ANE-inference ∥ GPU-training concurrency benchmark on M-series** — only phone-class negative anecdotes
  and unloaded-GPU positive framing. Must self-measure.
- **W8A8 compute-speedup on M4/M5 is contested** (maderix README 1.88× vs maderix substack + Orion "no INT8
  compute speedup") — reconciled only as a bandwidth/SRAM effect; M5-specific truth unmeasured.
- **CoreML gives no ANE-residency guarantee** (`CPU_AND_NE` is a request); silent CPU/GPU fallback per op is
  detectable only by profiling (T12). All our latency claims are contingent on residency staying stable across
  macOS updates (community-reported drift risk: "numeric drift aligned with version changes points to
  conversion-chain issues" — macgpu blog).
- **Direct-ANE (private API) route** solves residency + weight-reload but is version-fragile and undocumented;
  wrong risk profile for our frozen teacher today.

## Sources

- https://github.com/Anemll/Anemll (+ examples/VARIABLE_CONTEXT.md, examples/variable_context_demo.py, anemll/utils)
- https://machinelearning.apple.com/research/neural-engine-transformers
- arXiv:2603.06728 (Orion) — read in full, pp.1–18
- arXiv:2606.22283 (ANE: Architecture, Programming, and Performance) — abstract + search excerpts only
- https://github.com/maderix/ANE + https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615
- https://apple.github.io/coremltools/docs-guides/source/typed-execution.html · opt-quantization-overview.html · multifunction-models.html
- https://medium.com/axinc-ai/using-mixed-precision-in-core-ml-77c2428ba728
- WhisperKit arXiv:2507.10860 + argmaxinc/whisperkit-coreml HF cards
- https://github.com/apple/ml-stable-diffusion (SPLIT_EINSUM)
- https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176
- https://pradeep.md/2026/03/30/apple-neural-engine-local-llm-agents.html
- https://hackernoon.com/i-made-my-iphones-neural-engine-and-gpu-run-inference-together-as-an-experiment-it-got-slower (403; headline + secondary summaries only)
- https://macgpu.com/en/blog/2026-0415-mac-coreml-vs-mlx-production-inference-ane-remote-pool.html (search excerpts)
