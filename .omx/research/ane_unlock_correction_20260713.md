# ANE frozen-SegNet forward correction ladder — 2026-07-13

`lane_id: lane_ane_unlock_correction_20260713`  
`checkpoint_id: ane_unlock`  
`research_only: true`  
`training_launched: false`  
`score_claim: false`  
`promotion_eligible: false`  
`pointer_moved: false`  
`review_status: measured-complete-no-joint-bar`

## Preregistration — frozen before correction-ladder measurement

**LITERAL ASK:** determine whether the measured CoreML/ANE frozen-SegNet forward can be corrected
within its measured CPU-time surplus, and report the complete fidelity/latency trade curve.

**PRIMARY ADMISSION BAR (operator-specified):** on the held-out real-frame split, final SegNet
argmax flip rate versus one-thread CPU-Torch fp32 must be `<= 0.0033%` (`<= 3.3e-5` as a
fraction), while the measured end-to-end corrected forward remains `>= 10x` faster than the
matched one-thread CPU-Torch fp32 reference. A rung below either threshold is not label-grade and
will not be silently promoted.

**MEASURED STARTING RECEIPTS (inputs, not re-derived):** CoreML fp16 `CPU_AND_NE` median
`9.098124981 ms` versus one-thread CPU-Torch median `346.006374981 ms` at
`(1,3,384,512)`, a `38.0305146x` forward speedup. On 24 real cached frames with real frozen
weights, fp16 CoreML produced `116,764 / 4,718,592 = 0.0247455173` argmax flips and median logit
cosine `0.9943149984`. Sources:
`experiments/results/precision_backend_matrix_20260713/main_local/a2_ane_latency.json` and
`a2_ane_fidelity.json`.

**DERIVED CORRECTION BUDGET:** the matched starting medians leave
`346.006374981 - 9.098124981 = 336.90825 ms` per forward before simple forward-only break-even.
The primary `>=10x` bar allows total corrected latency at most `34.600637498 ms`, so the usable
correction budget for that bar is only `25.502512517 ms` beyond the measured ANE pass. Both
figures will be superseded by matched timing in the terminal receipt.

### Fixed rungs and falsifiers

1. **R0 error decomposition:** compare Torch-fp32 with CoreML FLOAT32 on `CPU_ONLY` and
   `CPU_AND_GPU`; compare CoreML fp16 against CoreML-fp32; localize flip sites by Torch margin,
   ordered class pair, and adjacency-boundary distance. Falsifier: if FLOAT32 CoreML already has
   substantial flips, precision-only remedies cannot close the full gap.
2. **R1 precision split:** sweep final conv head, last decoder block, and last two decoder blocks,
   using CoreML selective precision when placement is viable and/or a two-model ANE-trunk plus
   Torch-fp32 tail. Report prediction plus transfer/tail/total timing for each.
3. **R2 calibrated logit correction:** fit per-channel affine correction on calibration frames
   only; evaluate once on held-out frames. If affine misses the bar, fit a tiny at-most-three-layer
   logit-space corrector with deterministic seed and preserved calibration checkpoint. No held-out
   selection or training is permitted.
4. **R3 approximate band correction:** choose thresholds on calibration frames, report band area
   and flip coverage, and recompute only selected band tiles with a modest halo. Global/SE values
   may be donated by the ANE pass. This is an approximate corrective formulation outside the
   earlier bit-exact tile-halo verdict; it does not re-open exact tiling.
5. **R4 variants:** CoreML linear W8A8 where supported and FLOAT32 on `CPU_AND_GPU`, each with
   real-frame flips and matched latency.
6. **R5 composition:** compose only calibration-chosen rungs, remeasure on held-out frames, and
   report forward speedup plus teacher Amdahl speedup with forward share `p=0.78`:
   `S_teacher = 1 / ((1-p) + p / S_forward)`.

**FIXED SPLIT:** frame indices `0..11` are calibration and `12..23` are held out in the existing
`gt_n24.npz['gt_f1']` order. Results on the calibration split are fitting evidence only. Held-out
metrics carry the rung verdict. If a larger real cache is used later, it is a separately labeled
confirmation and does not change this split after seeing results.

**CONSUMER TIERS:** `training-gradient tolerance` may admit a sub-label-grade rung only under a
separate gradient/tie gate; `verdict-advisory` may use a measured but non-label-grade backend with
explicit drift; `label-grade` requires the primary held-out bar. No local row is contest score
authority.

**VERDICT SCOPE:** every negative is at most the measured correction formulation on this host,
CoreML build, frozen SegNet, input geometry, and real-frame split. The prior exact tile-halo kill
binds bit-exact tiling only; this memo's band correction is explicitly approximate and may use
donated global statistics. The killed cheap-feature prediction surrogate is a different family
from post-forward correction and is not reused as a verdict.

**STORES CONSULTED:** unified corpus query over `research(5805)`, `equations(664)`,
`memory(1894)`, `dag(520)`, `council(277)`, `tasks(135)`, and `docs(92)`; `CLAUDE.md`;
`AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`; v7.5/v8 specifications;
latest Codex findings/session summary, T3 council, V9 design, current directive files, canonical
lane/subagent/task/frontier surfaces, precision-backend receipts, and the existing A2 scripts.

## Measurements

## Outcome first

**NO label-grade ANE correction stack was unlocked.** The only held-out fidelity pass is the full
CoreML FLOAT32 graph on `CPU_AND_GPU`: `28/2,359,296 = 1.1867947e-5` flip fraction
(`0.0011868%`, below the `0.0033%` bar). Its matched median is `70.768813 ms` versus
`255.428583 ms` one-thread CPU-Torch, only `3.609338x`, below the `10x` bar. With teacher forward
share `p=0.78`, the measured full-teacher Amdahl speedup is `2.293020x`.

The cross-receipt sensitivity using the operator's starting `346.006375 ms` CPU timing is
`4.889249x` forward and `2.634812x` teacher, still below `10x`; it is **not admission evidence**
because its random-init latency custody is not matched to the real-weight FLOAT32 run. The true
starting ANE receipt remains `9.098125 ms`, `38.030515x` forward, `4.157834x` teacher, but its
real-weight fidelity is `2.4745517%`, so it fails the label gate by approximately `750x`.

The answer to the operator's P0 is therefore precise: **FLOAT32 CoreML can correct the labels, but
no measured correction preserves a >=10x matched speedup, and this pass cannot claim that its
FLOAT32 graph ran on ANE.** Partial precision, output calibration, approximate band tiles, and W8A8
all fail on fidelity. The frontier pointer remains unmoved.

### Timing-custody blocker

The dense ladder requested `CPU_AND_NE`, but E5RT repeatedly failed to create its cache under
`~/Library/Caches/...e5bundlecache`. The same graph then measured approximately `75 ms`, not the clean
main-local `9.098 ms`; placement is consequently **UNCONFIRMED** and the ladder timings are local
fallback/cache-contaminated evidence. Computer-use was attempted to run the command in Terminal outside
the restricted execution context, but the app safety policy forbade Terminal control. The operator-provided
scratch `.venv` also became externally absent after R0/R1/R2/R4; it was not rebuilt. R3 reused the
already-existing `.venv_executorch_spike` and records Torch `2.12.0` separately.

## R0 — decomposition and WHERE

On the fixed held-out frames under the Torch-2.13 dense receipt:

| Comparison | Flips / 2,359,296 | Flip fraction | Interpretation |
|---|---:|---:|---|
| Torch-fp32 vs CoreML-fp32 CPU_ONLY | 28 | `1.1867947e-5` | op-substitution floor; passes flip bar |
| CoreML-fp32 vs CoreML-fp16 | 65,648 | `0.0278252496` | precision/placement delta |
| Torch-fp32 vs CoreML-fp16 | 65,657 | `0.0278290643` | final fallback-placement error |

The pairwise precision delta is `99.9863%` of the final-flip magnitude. Pairwise distances are not
strictly additive when all three labels differ, so this ratio is a routing statistic, not an exclusive
partition. A separate Torch-2.12 three-way re-derivation corroborates the route with `65,636/65,653 =
99.9741%` held-out flips classified as precision-only, but differs by four final flips and is retained
as a runtime-sensitivity receipt rather than substituted into R0.

The all-n24 fallback graph produced `163,580/4,718,592 = 3.4667121%`, worse than the canonical
main-local starting receipt's `116,764/4,718,592 = 2.4745517%`. That drift is another reason not to
equate the sandbox placement with the clean ANE placement.

**WHERE:** Torch margin at fp16 flip sites has median `0.130278`, q90 `0.509296`, q95 `0.803307`, and
q99 `1.495245`; the error is not confined to razor ties. A 2-pixel reference-boundary annulus occupies
`6.1257%` of pixels but covers only `50.1314%` of flips; 4 px occupies `9.9916%` and covers `66.1279%`;
8 px occupies `16.6639%` and covers `81.3755%`; 16 px is needed for `91.7600%` coverage. The four
dominant ordered confusions are `2->0: 66,074`, `4->0: 36,041`, `0->2: 26,054`, and
`0->4: 18,658`. This broad spatial/margin support predicts why an approximately 5% correction band
cannot close the error.

## Full held-out trade curve

Every speedup below uses the matched CPU timing inside that rung's receipt. `flip pass` means only
the preregistered fidelity half; `speed pass` means only `>=10x`.

| Rung / variant | Held-out flips | Flip % | Median ms | Forward x | Teacher x | Flip pass | Speed pass |
|---|---:|---:|---:|---:|---:|---|---|
| R0 fp16 `CPU_AND_NE` request | 65,657 | 2.782906 | 75.271 | 3.393 | 2.223 | no | no |
| R0 fp32 CPU_ONLY | 28 | 0.001187 | 83.071 | 3.075 | 2.111 | **yes** | no |
| R0 fp32 CPU_AND_GPU | 28 | 0.001187 | 70.769 | 3.609 | 2.293 | **yes** | no |
| R1 fp32 final head | 65,702 | 2.784814 | 67.788 | 3.768 | 2.342 | no | no |
| R1 fp32 last decoder block + head | 65,724 | 2.785746 | 74.016 | 3.451 | 2.242 | no | no |
| R1 fp32 last two decoder blocks + head | 65,637 | 2.782059 | 71.061 | 3.594 | 2.288 | no | no |
| R2 per-channel affine | 81,790 | 3.466712 | 75.483 | 3.384 | 2.220 | no | no |
| R2 one-layer 3x3 residual corrector | 97,988 | 4.153273 | 73.190 | 3.490 | 2.255 | no | no |
| R3 5%-area band, donated SE | 98,149 | 4.160097 | 1,091.628 | 0.292 | 0.346 | no | no |
| R3 50%-flip-cover threshold | 100,623 | 4.264959 | 1,091.212 | 0.292 | 0.346 | no | no |
| R3 75%-flip-cover threshold | 155,338 | 6.584083 | 1,126.146 | 0.283 | 0.336 | no | no |
| R3 90%-flip-cover threshold | 233,890 | 9.913550 | 1,155.948 | 0.276 | 0.328 | no | no |
| R3 95%-flip-cover threshold | 285,409 | 12.097210 | 1,204.285 | 0.265 | 0.316 | no | no |
| R3 99%-flip-cover threshold | 371,811 | 15.759405 | 1,231.857 | 0.259 | 0.310 | no | no |
| R4 W8A8 PTQ | 1,081,426 | 45.836809 | 144.095 | 1.773 | 1.515 | no | no |

### R1 — precision split

Keeping only the head, last block, or last two decoder blocks in FLOAT32 changes the held-out rate by
at most `8.88e-6` absolute around the fp16 baseline and never approaches the `3.3e-5` bar. This scopes
the negative to CoreML's per-op precision-selector formulation on this converted graph: the damaging
precision is distributed earlier than these tails, or the planner does not realize the intended split.
It is not a kill of every physical two-model partition.

### R2 — calibrated logits

The five-channel affine fit used calibration frames `0..11` only and worsened held-out flips to
`3.466712%`. The deterministic 230-parameter, one-layer 3x3 residual corrector also used only the
calibration split and worsened held-out flips to `4.153273%`. The fit checkpoint is preserved. These
results kill only these output-space fits; they show that the error is not a stationary channel bias or
one-layer local logit residual.

### R3 — approximate donated-SE band tiles

The CoreML donor emitted logits plus all **23** full-frame EfficientNet-B2 SE gates. CPU-Torch injected
those gates and recomputed stride-32-aligned `64x64` cores with a `32 px` halo. This is explicitly
approximate and outside the earlier exact tile-halo verdict.

The calibration 5%-area threshold covers only `48.4360%` of calibration flips and activates a median
`22.5/48` held-out tiles because the band is spatially dispersed. It worsens held-out flips from
`2.782906%` to `4.160097%` and costs `1.091628 s`. Covering `99.0023%` of calibration flips requires a
`24.1934%` band, activates a median `27/48` tiles, and worsens held-out flips to `15.759405%`.

**Scoped verdict:** `NO_GO` for full-network `core=64, halo=32`, donated-gate band recomputation. The
finite halo changes local activations too much even with global gates, and tile occupancy destroys the
economics. This says nothing terminal about a dense encoder/SE pass followed by a sparse decoder cut;
that reformulation is staged. All 24 frame checkpoints are preserved and the complete R3 footprint is
below 1 MiB.

### R4 — variants

True calibration-driven CoreML W8A8 (linear-symmetric per-channel int8 weights plus per-tensor int8
activations) yields `45.836809%` held-out flips and `144.095 ms`. This is a terminal negative only for
this post-training-quantized formulation/calibration. FLOAT32 CPU_AND_GPU is the best fidelity rung.

## R5 — composition and consumers

Every additive R1/R2/R3 correction worsens or fails to materially improve fidelity, so the optimum
measured stack degenerates to the full FLOAT32 CPU_AND_GPU graph. No row satisfies both bars.

The existing canonical measured-only equation closes without a new registration:

```text
S_forward = T_cpu / T_corrected
S_teacher = 1 / ((1 - p) + p / S_forward),  p = 0.78
ADMIT_label = 1[flip_rate <= 3.3e-5] * 1[S_forward >= 10]
```

For the selected row, `S_forward=3.609338`, `S_teacher=2.293020`, and `ADMIT_label=0`. The reused
equation is `amdahl_measured_disjoint_wall_split_with_async_cpu_verdict_v2`; a duplicate equation was
not registered.

| Consumer tier | Status | Honest disposition |
|---|---|---|
| training-gradient tolerance | **BLOCKED_NOT_MEASURED** | Forward logits do not measure input cotangents or parameter gradients. #456 permits scoped razor-tie reduction drift, not unknown placement/arbitrary flips. |
| verdict-advisory | **UNLOCKED_LOCAL_ONLY** | Full FLOAT32 CoreML clears the held-out flip bar for explicitly local advisory labels; no contest or score authority. |
| label-grade | **NOT_UNLOCKED** | No rung clears flip and matched `>=10x` simultaneously. |

## Triality, apparatus, and successor queue

- **Equation:** existing measured disjoint Amdahl law, instantiated in `r5_composition.json`.
- **DSL:** N/A-with-reason. No trainer/backend flag was invented; trainer wire-in is a later ticket.
- **DAG:** `.omx/research/ane_unlock_correction_DAG_FEED_20260713.md` is the isolated FEED.
- **Pool:** three rows were staged through the real `record_candidate` helper in
  `.omx/research/ane_unlock_correction_candidate_rows_20260713.jsonl`: clean FLOAT32 placement
  remeasure, n600 input-cotangent parity, and dense-encoder/sparse-decoder donated-SE reformulation.
  They were not silently injected into the shared controller pool.
- **Six hooks:** R0's margin/class/boundary surface is the research-only sensitivity contribution;
  flip+wall is the binding Pareto constraint; bit allocator is N/A because the corrector ships nowhere;
  autopilot remains blocked; memo/receipts/pool rows are the continual-learning update; the measured
  ladder is the probe-disambiguator.

## Receipts, verification, and pointer delta

Primary receipt directory: `experiments/results/ane_unlock_correction_20260713/`.

- `r0_decomposition.json` SHA-256 `c1c53caee7a99f3c26d85c17ef4a79c18c95a5557abe433c5e0e961de5ce569c`
- `r1_precision_split.json` SHA-256 `efee2774cbcf16db7284203c7b9abbcab0c2abf3e08a8e3634b2ef9152622f07`
- `r2_logit_correction.json` SHA-256 `5fb4407db20598fee371a8b9a53df7c402c2f827b5dc41f7a5de85d71c90a287`
- `r3_band_tile_donated_se.json` SHA-256 `4bec525fc0052ec0d7ef5a6f6d5fbc290e5b28b6df9ec4c93cb76c962c54bc23`
- `r4_variants.json` SHA-256 `bc9efd81f86894ace0dcf9cadd8a6b14721d8b8f56609bf60df111c200adbfd4`
- `r5_composition.json` SHA-256 `e5ffad62050c641771306ebe7b580220ff58fff939065e9bdd16334fac65cf73`
- `artifact_manifest.json` SHA-256 `e5660fb6da2406311a8bd775c929eb80ed6dd95d56680ca57251459785ec479b`

Source weight SHA-256 is `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`;
GT-cache SHA-256 is `684a2e043f5637eec6f2bc837b845588a7ec759c2a5c4b3b75dcb9f872541ede`.
Scratch scripts compile under their recorded runtimes. R3 emitted 24 atomic per-frame checkpoint pairs;
the complete lane is `652 KiB`, so no bulky artifact cleanup was required. No run directory, trainer,
archive, evaluator, contest axis, or frontier pointer was mutated. **Pointer delta: ZERO.** Files remain
uncommitted as requested.
