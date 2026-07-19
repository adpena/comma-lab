# Neural self-compression SOTA crosswalk for the v10 generator payload — 2026-07-19

`research_only=true` · no launch · no score claim · no pointer mutation · donor read-only

## Authority and one-line verdict

**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**.

**Verdict:** keep canonical int8+Brotli for the present ep725 v10 generator; the receiver-closed
DeepCABAC base-weight archive is **+1,436 B** (the base stream itself is +1,409 B), while its
unassembled base+code streams are +16,349 B at section level; the
only high-EV neural move is a future, default-OFF, joint size-in-loss arm on a wider/new vehicle
whose measured quantization response clears the current rate-distortion knee.

`verdict_scope: instance/regime` — this rejects DeepCABAC `(lambda=0, interval=1)` as the lossless
coder for this donor and rejects sub-int8/entropy-shaping at the already-measured current operating
point. It does **not** reject learned conditional priors, REC, self-compression training, or new
witness vehicles. Every negative below narrows its own scope.

This pass follows `docs/operating_manual_craft_handoff.md`: actual counted bytes and realized-through-R
receipts outrank proxy entropy; quantities are labeled MEASURED/DERIVED/ESTIMATE; negative verdicts
carry scope; and the contest pointer moves only on exact contest-axis evidence.

## Question, rate object, and Rule 118 split

The object is the complete **video-derived v10 generator payload**, not only an isolated `.npz`:

1. witness base weights;
2. per-pair codes;
3. gauge-fixed packet scalars;
4. `dxi` and phase sections; and
5. any fitted entropy prior, hyperprior, topology mask, quantization exponent, or sampler state.

The measured ep725 archive currently has only items 1–2: 61,598 B base + 21,010 B codes in a
84,536 B blob (83,838 B `archive.zip`). Items 3–4 are absent, so no byte claim is made for them.

**Rule 118:** generic decoder code, a video-independent prior architecture, a fixed seeded random
network, and generic rejection/REC machinery are **FREE**. Every learned or fitted value that depends
on this video is **COUNTED**: weights, latents, masks, posterior/prior parameters, CDFs, seeds selected
from the video, tensor shape/step headers, packet scalars, `dxi`, phase, and side information. A
pretrained network unavailable in the frozen evaluator is not silently treated as free.

This is a literature/OSS survey, explicitly independent of contest-PR lineage. Source state was
refreshed through 2026-07-19; runnable means a public implementation exists, not that its receiver is
contest-ready.

Required-family map: **Family 1** is size-in-loss self-compression; **Family 2** is entropy modeling
over weights/INR state (NVRC, NeuroQuant, C3/Cool-Chic, NNCodec/DeepCABAC); **Family 3** is
REC/sample communication; **Family 4** is the newer 2025–26 implicit, diffusion and learned-wavelet
frontier. C3/Cool-Chic straddle Families 2 and 4 because their entropy model arrives inside a new
overfit representation rather than as a drop-in weight recoder.

## Ranked crosswalk

Rank is by expected value for this exact payload after custody, byte accounting, receiver portability,
and prior Pact evidence—not by paper headline.

| rank | family | paper + OSS snapshot | OUR bytes: MEASURED / DERIVED / ESTIMATE | Rule-118 free / counted split | consumer | falsifiable admission gate |
|---:|---|---|---|---|---|---|
| 1 | Size-in-loss self-compression | Csefalvay–Imber, [Self-Compressing Neural Networks](https://arxiv.org/abs/2301.13142); formulation spec below | **DERIVED:** deleting the entire 61,598 B base can save only 0.0410156 score, so pose-neutral `Delta d_seg` must be `<0.000410156`. Current #496/#242 evidence fails that knee; new arm bytes UNKNOWN. | FREE: quantizer/decoder implementation. COUNTED: learned `b`, exponent `e`, active topology, weight symbols, fitted headers and all packet fields. | task #154 weight-entropy; #242 MDL/flat minima; #496 rate-training; future #539/#553 vehicle | On a wider/new vehicle, default-OFF twin run must produce exact archive `Delta S < 0` at n24, then n600, with complete resume state and no cross-host receiver drift. |
| 2 | Learned/hierarchical entropy over network parameters | [NVRC](https://arxiv.org/abs/2409.07414), [official OSS](https://github.com/hmkx/NVRC) `ccc432d…`; [NVRC++](https://arxiv.org/abs/2606.28163) has no official runnable OSS found | **MEASURED control:** canonical base 61,598 B; **ESTIMATE:** an admitted learned prior must be `<61,598 B` including all fitted-prior overhead. NVRC donor bytes UNKNOWN; NVRC++ estimate-only. | FREE: fixed architecture/decoder. COUNTED: network latents, quantized parameters, entropy-model and hyperprior parameters, tables fitted to donor, syntax. | #154, #242, #539, #553 | Native receiver bitstream path must exist, parse back exact state, finish n600 `<30 min`, and beat 61,598 B after counted model overhead at matched n24 `d_seg`. NVRC README still marks input-bitstream evaluation code TODO: `verdict_scope: current OSS receiver surface`, family OPEN. |
| 3 | Sensitivity-calibrated mixed precision before entropy coding | [NeuroQuant](https://arxiv.org/abs/2502.11729), [OSS](https://github.com/Eric-qi/NeuroQuant) `4d787f6…` | **MEASURED prior:** WF mean-6 is 55,528 B but `Delta d_seg=+0.001319`, net `Delta S=+0.11353`; every tested tensor int8→int5 response was positive. NeuroQuant donor bytes UNKNOWN. | FREE: calibration/allocator code. COUNTED: mixed-precision symbols, bit-width map, scales, outlier values, any fitted calibration state required by receiver. | #496 primary; #242; #154 | Must improve exact n24 archive `Delta S`, not only reconstruction/calibration loss. Reactivate if quantization response falls about 50× or a wider vehicle moves its RD knee. `verdict_scope: current donor/regime`, allocator family OPEN. |
| 4 | Adaptive neural-weight syntax (best receiver-complete runnable Family-2 OSS) | [NNCodec](https://github.com/fraunhoferhhi/nncodec) `0e3dcd2…`; [DeepCABAC](https://github.com/fraunhoferhhi/DeepCABAC) `6a13468…` | **MEASURED actual full archive:** base-only 85,274 B vs canonical 83,838, **+1,436 B**; base stream 63,007 vs 61,598, +1,409 B. Freshly extracted selected-n24 archive emitted all 146,496,384 bytes exactly, matched `d_seg=0.0034556919`. Code-only stream loses +14,940 B but remains section-level. | FREE: generic decoder binary/inflate code outside archive. COUNTED: archive stream, 36-B codec-ID manifest delta, existing syntax/shape/step sideinfo. Fitted model overhead: 0 B. | sibling `arith_selfcomp_rate_coders`; #154/#242 control | Already fails bytes. Linux x86_64 decoder binary/parity remains owed; `verdict_scope: measured macOS DeepCABAC configuration`, NNCodec transforms and learned priors OPEN. |
| 5 | Relative entropy coding / sample communication | [COMBINER](https://arxiv.org/abs/2305.19185), [OSS](https://github.com/cambridge-mlg/combiner) `d54332b…`; [RECOMBINER](https://arxiv.org/abs/2309.17182), [OSS](https://github.com/cambridge-mlg/recombiner) `901588e…`; [GPRS](https://arxiv.org/abs/2305.15313), [OSS](https://github.com/Flamich/gps) `0d4622d…`; [GRC](https://arxiv.org/abs/2309.15746) | OUR bytes UNKNOWN. **DERIVED coding target:** about `KL(Q||P)/ln 2` bits plus framing/randomness overhead; proposal count can grow exponentially in block KL. | FREE: generic exact sampler, fixed public prior architecture, counter PRNG. COUNTED: donor posterior/prior parameters, selected seeds, indices, fallback data, all fit-dependent state. | #242 MDL; #154; low-sensitivity sections of #553 | $0 integer-lattice pilot below: bit-identical tensor SHA on macOS and Linux, p99 decode extrapolation `<30 min`, actual bytes below direct code, and n24 `Delta S<0`. Failure scopes only the chosen block/prior/sampler. |
| 6 | Shared random network + sparse fitted mask/modulation | [LotteryCodec](https://proceedings.mlr.press/v267/wu25e.html), [OSS](https://github.com/eedavidwu/LotteryCodec) `4abcc1a…` | OUR bytes UNKNOWN; runnable upstream but new-vehicle estimate, not a drop-in recode. | FREE: fixed seeded random network and generic decoder. COUNTED: selected mask, modulation/latent values, video-selected seed/index, syntax. | #539 witness vehicle; #553 packet; #242 | Deterministic NumPy-fp32 receiver must reproduce the fixed network cross-host; actual mask+modulation archive must beat current 83,838 B at matched Seg/Pose and finish n600 `<30 min`. |
| 7 | Families 2/4 overlap: overfit latent + small decoder / autoregressive latent prior | [C3](https://arxiv.org/abs/2312.02753), [OSS](https://github.com/Google-DeepMind/c3_neural_compression) `e63e75d…`; [Cool-Chic](https://github.com/Orange-OpenSource/Cool-Chic) `a6fe38f…` | OUR bytes UNKNOWN; **ESTIMATE/new vehicle**, not a weight-only drop-in. | FREE: generic multiresolution decoder and prior architecture. COUNTED: all overfit latents, decoder weights, fitted AR parameters/CDFs and headers. | #539/#553 new witness vehicle; #154 | Short receiver smoke first; then exact archive and n24 Seg/Pose. Reject if dependencies or n600 decode exceed 30 min. `verdict_scope: current drop-in applicability`, families OPEN as new vehicles. |
| 8 | Learned wavelet / per-instance latent coder | [WaLLoC](https://arxiv.org/abs/2412.09405), [OSS](https://github.com/ut-sysml/WaLLoC) `348095a…`; [Fitted Neural Lossless Image Compression](https://github.com/ZZ022/FNLIC) `4039d70…` | OUR bytes UNKNOWN; runnable upstream, but only **DERIVED architectural evidence** until a v10 rate object is defined. | FREE: CDF 9/7 or other fixed wavelet and generic network code. COUNTED: donor latents, fitted network/prior parameters, learned transforms not evaluator-provided, headers. | #539/#553 representation design; #154 | Define receiver-consumed payload first. Admit only actual complete archive bytes and realized-through-R score; no proxy image metric transfer. |
| 9 | Foundation/diffusion prior + communicated adaptation | [Compression as Adaptation / VOV](https://arxiv.org/abs/2603.07615) | **ESTIMATE-only:** no contest-ready OSS/payload measurement; reported sampling recipe is not mapped to the frozen evaluator or 30-minute receiver. | A generic architecture is free only if executable in the frozen receiver; pretrained/fitted model weights, LoRA/vector, selected samples and seeds are COUNTED unless evaluator-provided. | #539 future vehicle only | A dependency-complete portable receiver, counted foundation weights, fixed candidate budget, `<30 min` n600, and exact Seg/Pose gate. `verdict_scope: current runtime/dependency formulation`, diffusion family OPEN. |

Task numbers above are task consumers. In particular, task **#154 weight-entropy** is not the
unrelated Catalog #154 disk-hygiene rule.

## Family 1 — concrete default-OFF MLX size-in-loss specification

The Csefalvay–Imber quantizer is:

```text
q(w; b, e) = 2^e round(clip(2^(-e) w, -2^(b-1), 2^(b-1)-1))
z_l = I_l H_l W_l sum_i b_(l,i)
Lambda = Lambda_0 + gamma * (1/N) sum_l z_l
```

For Pact, use the contest-scaled form so the relaxed rate has the right units:

```text
B_relaxed_bits = sum_(active groups g) n_g * b_g + B_topology_bits
J_SC = J_witness + gamma * 25 * B_relaxed_bits / (8 * 37_545_489)
```

Exact packet-scalar/`dxi`/phase bytes remain stop-gradient constants until an honest differentiable
model exists; the final gate always substitutes actual compressed archive bytes. `J_witness` is the
same evaluator-facing training objective as its twin control—never a new visual proxy.

Typed DSL design (**specification only; do not build in this arm**):

```text
WitnessSelfCompressionGauge(
    mode = OFF | CHANNEL_BITDEPTH_STE_V1,       # default OFF
    gamma = 0.0,                                # default no-op
    grouping = OUTPUT_CHANNEL,
    b_min = 0, b_max = 8,
    exponent_mode = LEARNED_PER_OUTPUT_CHANNEL,
    activate_stage = <typed stage boundary>,
    prune_patience_stages = <typed positive integer>,
    exact_byteclose_gate = true,
)
```

MLX feasibility is **PLAUSIBLE**: elementwise clip/round/scale is native, and
[`mx.custom_function`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html)
or the repository's custom-VJP pattern can provide an STE. The VJP must return gradients for **weight,
bit depth, and exponent**. The existing Torch `src/tac/self_compress.py` returns no gradient for its
bit-depth input in the custom backward, so it is not evidence that accuracy pressure can learn bits.
Do not port that behavior.

Resumability is part of the lever contract: persist `b/e`, active topology, quantized EMA shadow,
optimizer state, stage and epoch; save atomically at every stage; prune only at stage boundaries; and
preserve every prior stage checkpoint. A `b=0` transition must reshape downstream consumers and
optimizer slots in one atomic stage transition or refuse to proceed.

**Admission law.** With pose unchanged, even deleting every base-weight byte buys only
`25*61,598/37,545,489 = 0.0410156` score. Therefore any arm with
`Delta d_seg >= 0.000410156` loses before topology/prior overhead. Prior #496 waterfill mean-6 costs
`+0.001319 d_seg`; task #242's tested entropy-penalty region was also net negative. This design is
therefore **HOLD for a wider/new vehicle**, not authority to modify or launch the current run.

`verdict_scope: current ep725 donor/regime` — the break-even calculation is exact for this base
section and pose-neutral assumption; it is not a family kill for train-big/compress-small.

## Family 2 — real donor measurement and the entropy-ceiling question

The donor remained sacred and read-only:

- ep725 `levelset_witness_ema_BEST.npz`, 460,448 B, SHA-256
  `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef` before and after;
- 17 base tensors / 72,695 int8 symbols; code `(1200,32)` / 38,400 symbols;
- canonical full n600 archive: **83,838 B MEASURED**, SHA-256
  `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3`;
- canonical n24 realized-through-R interaction row: `d_seg=0.003455691873872032`,
  `d_pose=63.0309157371521`.

DeepCABAC was selected as the best receiver-complete runnable Family-2 OSS for this exact transplant:
NVRC's current public receiver/evaluation path is incomplete, NeuroQuant is an allocator rather than
a codec, and C3/Cool-Chic require a new vehicle. DeepCABAC was built from commit `6a134687…` on the
SSD, then run at `lambda=0`, interval/step 1 over the canonical per-tensor symmetric-int8 arrays.
Syntax, tensor shape and step side information are in the measured streams; fitted learned-model
overhead is 0 B.

| surface | canonical | DeepCABAC | delta | authority |
|---|---:|---:|---:|---|
| base inner stream | 61,598 B | 63,007 B | **+1,409 B (+2.287%)** | MEASURED section |
| full n600 archive, base replaced / code still Brotli | 83,838 B | **85,274 B** | **+1,436 B** | MEASURED archive.zip |
| selected-n24 archive, base replaced / code still Brotli | 63,876 B | **65,310 B** | **+1,434 B** | MEASURED archive.zip |
| codes inner stream | 21,010 B | 35,950 B | **+14,940 B** | MEASURED section only |
| base+codes inner streams | 82,608 B | 98,957 B | **+16,349 B (+19.791%)** | MEASURED section only; no combined archive |

A deterministic receiver packet was then built with a compact counted codec ID in the manifest;
generic `inflate.py` and the 394,240-B macOS-arm64 decoder binary remain Rule-118 FREE outside
`archive.zip`. The selected-n24 archive was freshly extracted and decoded three times. Every complete
146,496,384-B output had SHA-256 `3ffe6e11…` and was byte-identical to the separately scored canonical
n24 raw, so all 24 frozen-scorer inputs and the quoted `d_seg/d_pose` are identical. The timed run was
14.89 s with 18 workers on Apple M5 Max. This closes the requested macOS advisory byte-close; the
platform-specific decoder is not Linux/contest authority.

Replacing only the base makes actual full-archive rate score **+0.000956173 worse**, derived from
`25*1,436/37,545,489`. The +16,349-B base+code figure is deliberately **not** converted to score:
no combined DeepCABAC receiver archive was assembled.

The base's single global-iid entropy is **6.838867 bits/symbol MEASURED**, or **62,143.934 B
DERIVED**. Canonical per-tensor Brotli is 545.934 B *below* that global-iid estimate, whereas
DeepCABAC is 863.066 B above it. Thus:

- the tested object is **not merely at a global-iid coder ceiling**; local/tensor structure matters;
- the incumbent already exploits enough structure that a generic coder swap is not the lever;
- a learned conditional prior remains scientifically OPEN, but must encode the base in `<61,598 B`
  **after every fitted-prior byte**. Proxy cross-entropy is not admission evidence.

The unmeasured Family-2 candidates divide cleanly: NVRC is the direct hierarchical
quantization+entropy-model transplant; NeuroQuant is a sensitivity/bit-width allocator that still
requires an external codec; and C3/Cool-Chic couple their latent prior to a different overfit INR
vehicle. None supplies OUR donor bytes yet, so each is labeled UNKNOWN/ESTIMATE rather than borrowing
its paper bitrate.

The sibling `arith_selfcomp_rate_coders` owns classical coders and the PR56 block-FP baseline. Its
landing manifest was not available at memo write time, so this arm does not duplicate it. MAIN landing
review must place its final int8+coder row beside this receipt before consuming a “best coder” claim.

Full machine-readable receipt:
`.omx/research/neural_selfcomp_deepcabac_measurement_20260719_codex.json`.

## Family 3 — deterministic REC/sample-communication pilot

COMBINER/RECOMBINER communicate posterior samples relative to a prior, avoiding ordinary scalar
quantization. In the ideal coding model, expected message length tracks `KL(Q||P)/ln 2` plus overhead.
The contest blocker is not the identity alone: proposal count can grow exponentially in per-block KL,
and standard floating Gaussian samplers/CDFs need not be bit-identical across macOS and Linux.

The sharpest **$0 pilot** is deliberately small:

1. choose `out_sdf.weight` (480 symbols), the lowest measured sensitivity surface; prior n16
   int8→int5 `Delta d_seg=+0.000042` only identifies the pilot target and is not a transfer claim;
2. define both `P` and `Q` on an exact integer lattice—no host libm Gaussian CDF;
3. use a counter-based fixed PRNG with a public, video-independent seed schedule;
4. cap block KL at 8–10 bits and make fallback literal coding explicit and counted;
5. run 256 seed trials; record mean/p95/p99 proposals, bytes and decode time;
6. compare decoded tensor SHA across macOS and Linux x86_64;
7. parse back into the canonical packet and run the n24 realized-through-R gate;
8. refuse if p99 extrapolated n600 decode exceeds 30 minutes or any host SHA differs.

All donor-fit posterior/prior parameters, selected indices/seeds and fallback symbols are COUNTED.
The generic sampler/rejection decoder and video-independent prior architecture are FREE. This pilot
is estimated to be minutes on CPU and $0, but it is **not launched by this research-only arm**.

`verdict_scope: deterministic portability constraint` — a floating/uncapped implementation that
fails cross-host SHA or the runtime budget is closed; exact discrete/blockwise REC remains OPEN.

## Family 4 — newer priors, implicit vehicles, diffusion, and learned wavelets

- **LotteryCodec (ICML 2025)** is the most Rule-118-native new-vehicle idea: a fixed seeded random
  network can be FREE, but its video-selected binary mask/modulations are COUNTED. OSS is runnable;
  OUR exact bytes and Seg/Pose are UNKNOWN.
- **C3 and Cool-Chic** show that overfit multiresolution latents plus a small decoder and learned
  latent prior can beat a monolithic INR representation. They are vehicle candidates, not evidence
  that the current weight blob can be recoded losslessly.
- **WaLLoC** and **FNLIC** supply runnable wavelet/latent and per-instance-prior mechanisms. A fixed
  CDF 9/7 wavelet can be FREE; all fitted latents and prior/network values remain COUNTED. OUR bytes
  are UNKNOWN.
- **NVRC++ (2026)** is the closest newer entropy-model paper, but no official runnable OSS was found
  as of 2026-07-19. It is **ESTIMATE-ONLY**, explicitly not a donor-byte claim.
- **Compression as Adaptation / VOV (2026)** couples a foundation/diffusion decoder to communicated
  adaptation and large candidate search. No contest-ready OSS/payload measurement was found, and
  pretrained weights cannot be assumed evaluator-free. It is **ESTIMATE-ONLY** under the present
  receiver/runtime contract.

`verdict_scope: drop-in applicability and current OSS` — none of these negative/unknown labels kills
the representation families. Their admission unit is a complete receiver-closed archive, not a
paper metric or latent bitrate that omits decoder/fitted-prior bytes.

## Decision and reactivation queue

1. **KEEP now:** canonical per-tensor int8+Brotli on this donor. DeepCABAC loses exactly; no coder
   launch is owed.
2. **HOLD for the next wider/new vehicle:** `WitnessSelfCompressionGauge` default-OFF twin arm,
   only after a short quantization-response probe predicts `Delta d_seg < 0.000410156` for the full
   base-deletion bound.
3. **$0 disambiguator:** exact discrete REC on `out_sdf.weight`; this is the cheapest way to decide
   whether sample communication is portable and runtime-bounded before touching the full payload.
4. **Vehicle shortlist:** LotteryCodec first, then C3/Cool-Chic; each must count mask/latent/decoder
   parameters and pass a deterministic NumPy receiver smoke before training spend.
5. **Learned-prior reopener:** transplant NVRC/NVRC++ only when the public receiver path is complete
   or a local deterministic bitstream decoder is independently specified; threshold is `<61,598 B`
   including fitted prior overhead at exact matched state.

No proposal changes the score pointer, launch state, current run, or archive selection.

## Triality and system wire-in

- **DSL:** proposed typed `WitnessSelfCompressionGauge`, default OFF; existing
  `WeightEntropyPenaltyMLX` remains the separate entropy-shaping control. No CLI flag was invented.
- **DAG:** `donor checkpoint -> canonical int8 grid -> {Brotli control, DeepCABAC measured control,
  future learned prior/REC} -> exact parse-back -> NumPy receiver -> n24 Seg/Pose -> n600 ->
  contest CPU/CUDA`. Every edge before exact parse-back is non-authorizing.
- **Equations:** contest action `S=100*d_seg+sqrt(10*d_pose)+25*B/37,545,489`; self-compression
  quantizer/loss and REC length law are stated above. Actual archive bytes replace differentiable
  rate estimates at admission.
- **Sensitivity map / bit allocator:** consume existing tensor response ordering; do not repeat
  #496/#336. REC pilot starts at `out_sdf.weight`; size-in-loss groups by output channel.
- **Pareto constraint:** accept only `Delta S<0` with Seg, Pose and rate separately recorded. No
  aggregate may hide a facet regression.
- **Cathedral/autopilot:** research-only; no dispatch hook is enabled. A future build becomes eligible
  only after the named n24 gate and canonical lane claim.
- **Continual learning:** the durable measured row and scoped reactivation criteria are this memo plus
  the JSON receipt. No new empirical equation anchor is registered because exact lossless recoding
  preserves the already-registered receiver row and the DeepCABAC result is an implementation row,
  not a universal law.
- **Probe disambiguator:** exact discrete REC pilot arbitrates REC portability; learned-prior gate
  arbitrates coder ceiling vs entropy-model ceiling.

## STORES CONSULTED and provenance

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`;
- `reports/latest.md` and `tac.frontier_scan.build_frontier_scan_payload` surfaces;
- `.omx/state/lane_registry.json`, `canonical_task_status.jsonl`,
  `subagent_progress.jsonl`, `master_gradient_anchors.jsonl`, `modal_call_id_ledger.jsonl`,
  `cost_band_posterior.jsonl`, `continual_learning_posterior.jsonl`;
- council continual-learning queries for self-compression/entropy/MDL/rate and blocking-outcome
  query (no controlling hit);
- `.omx/research/p0_recovery_rate_probes_20260715.md`,
  `.omx/research/sensitivity_bitalloc_witness_n96_20260707.md`,
  `.omx/research/weight_entropy_penalty_balle_adversarial_review_byteclose_20260620.md`,
  `.omx/research/yhat_rd_ladder_witness_prepare_20260719_codex.json`, and
  `.omx/research/yhat_rd_ladder_20260719_codex.json`;
- latest sister findings/session/council/design memos named in session preflight;
- public paper/OSS links in the crosswalk, pinned commits below.

Pinned public snapshots: NVRC `ccc432d…`; NeuroQuant `4d787f6…`; NNCodec `0e3dcd2…`;
DeepCABAC `6a13468…`; COMBINER `d54332b…`; RECOMBINER `901588e…`; GPRS `0d4622d…`;
relative-entropy-coding `a0ba55e…`; C3 `e63e75d…`; Cool-Chic `a6fe38f…`;
LotteryCodec `4abcc1a…`; WaLLoC `348095a…`; FNLIC `4039d70…`.

Bulk OSS/evidence lives under
`/Volumes/VertigoDataTier/pact/evidence/neural_selfcomp_20260719/`; no local bulky artifact or
`/tmp` authority surface was created. Sacred donor SHA is unchanged. **MAIN landing review is required**
before these artifacts can be merged or used as canonical authority.
