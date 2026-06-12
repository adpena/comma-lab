<!-- SPDX-License-Identifier: MIT -->
# NeRV-family SYNERGY survey vs the HNeRV Track-A carrier — under the invariant-floor verdict (2026-06-12)

**UTC:** 2026-06-12T17:45:28Z · **Subagent:** `nerv-family-synergy-survey-20260612` · **Mode:** DESIGN/survey/research only
(no production code, no GPU, no dispatch, no /tmp, $0 spend, no collision: did NOT touch the MPS basin out-dir
`experiments/results/torch_vehicle_full_mps_basin_bc20_n600`, `src/tac/torch_vehicle/**`, or
`src/tac/substrates/cool_chic/**`).
**Evidence grade:** every quantified claim is tagged either **[external:<paper>]** (a published number — NEVER our
score), **[MEASURED:<artifact>]** (an exact number from a cited in-repo smoke), or **[PREDICTION:<basis>]** (a
derivation). NO contest score is claimed. `promotable=false`, `score_claim=false`, `score_roadmap_update_eligible=false`,
`mechanism_update_eligible=true`. Per Catalog #307: nothing here KILLS a paradigm; verdicts are
IMPLEMENTATION/EV-LEVEL classifications that direct the next $0 probe.
**Frontier (pointer, NOT hardcoded):** `.omx/state/canonical_frontier_pointer.json` → contest-CPU **0.19109982**
(177,169 B, sha `b46897267d…`, `lane_pr110_payload_entropy_recode`); contest-CUDA **0.20533** (186,876 B).
**Frontier UNMOVED.** Ladder: T_3 = sub-0.15 (the aim), T_1 = sub-0.19 (floor of acceptable), T_floor ≈ 0.118.

> **THE OPERATING CONSTRAINT (the verdict this survey runs UNDER).** The Layer-1 carrier first-principles memo
> (`layer1_carrier_first_principles_20260612T171912Z.md`, commit `2350b6b2e`) establishes — and the
> `capacity_verdict_…_20260611.md` MEASURED-refutation corroborates — that **the rate floor is a scorer-conditional
> MDL INVARIANT, not carrier-dependent. No carrier swap lowers it.** Therefore a NeRV variant can help toward sub-0.15
> ONLY via one or more of: **(1) distortion control at equal bytes** (finer spatial detail → fewer d_seg argmax flips
> on the thin boundary band; better temporal/flow modeling → lower d_pose); **(2) slack reduction** (closer to the
> floor at equal bytes — bounded, because HNeRV's overfit weights are already near-MDL); **(3) trainability** (faster /
> deeper basin convergence on the single video, which buys more distortion descent per fixed compute); **(4)
> dashcam-structure fit** (inductive bias matching ego-motion flow + patch recurrence of ONE drive); **(5) lever
> synergy** (composes with the Layer-2 pose-FiLM + score-domain levers, doesn't fight them). **A variant cannot lower
> the floor.** This survey scores every variant on exactly these axes and ignores rate-floor claims from the papers
> (which are PSNR-RD, the wrong objective — §C.5 below).

---

## HEADLINE — the three answers

1. **The two structurally synergistic candidates are HiNeRV (d_seg-spatial) and FFNeRV/DNeRV (d_pose-temporal).** Under
   the invariant floor, the contest is a DISTORTION problem (the 0.07314 distortion residual is the sub-0.15 lever,
   per the carrier memo §A.4), and these two attack the two distortion axes from opposite ends:
   **HiNeRV's hierarchical bilinear-grid encoding raises spatial capacity** (its raison d'être: it gives 72.3% bit-rate
   saving over HNeRV at equal PSNR `[external:arXiv 2306.09818]` — i.e. **much more spatial fidelity at equal bytes**,
   which is exactly "lower d_seg at equal rate"); **FFNeRV's flow-guided aggregation and DNeRV's frame-difference stream
   model inter-frame motion** — the same frame_0→frame_1 delta structure PoseNet reads — i.e. the d_pose axis.
2. **Exploration status: we have built ADAPTERS/SCAFFOLDS for nearly the entire family, but the two highest-EV mechanisms
   are PARTIAL, not complete.** `hi_nerv/` implements the hierarchical-latent INJECTION (real, wired) but the
   **HiNeRV-defining hierarchical feature-grid / bilinear-grid upsampling path is an L0 SCAFFOLD, OFF by default, and
   `official_core_forward_parity_proven=False`** (`src/tac/substrates/hi_nerv/architecture.py:82,136`). `ff_nerv/` is an
   L0 SKETCH (DCT band-limited grid, real forward, never trained to an anchor). This is **READY ∧ high-EV orphaned
   signal** per the anti-signal-loss non-negotiable: the mechanism is the single most-cited reason HiNeRV beats HNeRV,
   it is half-built, and it has never been head-to-head measured against our HNeRV decoder on the score axis.
3. **The honest EV bound is SMALL and bounded by the distortion residual, not the rate floor.** HNeRV sits near
   R\*_scorer; a better spatial/temporal bias can only shave the d_seg / d_pose residual at equal bytes, and that
   residual is `0.07314` of the `0.19110` total `[MEASURED: capacity_verdict_…]`. The MAXIMUM a carrier-bias swap can
   buy is driving that residual toward 0 — and even that is mostly reachable by the Layer-2 levers on the EXISTING HNeRV
   bank (carrier memo §D.4). **So a variant is worth building ONLY if a $0 head-to-head shows it lowers d_seg (or d_pose)
   at MATCHED bytes by a margin the levers cannot already capture in-training.** The first $0 step (§D) measures exactly
   that, on the frozen basin checkpoint, with zero GPU and zero basin contention.

---

## A. THE FAMILY ENUMERATION (paper + OSS + framework + maturity in-repo)

All papers' headline numbers are **PSNR-RD on UVG/MCL-JCV/Bunny** — the WRONG objective for this contest (the scorer
ignores texture inside an argmax cell and the pose-null space, §C.5). They are kept `[external]` and used ONLY to read
off the *mechanism* and its *spatial-vs-temporal* character, never as our score.

| Variant | Paper | OSS repo | Framework | Core mechanism (what it adds over HNeRV) | In-repo maturity |
|---|---|---|---|---|---|
| **HNeRV** (OURS, reference) | [arXiv 2304.02633](https://arxiv.org/abs/2304.02633) (CVPR'23) | github.com/haochen-rye/HNeRV | torch | content-adaptive per-frame embedding + conv/PixelShuffle/sin decoder | **L1 ALPHA** — `sane_hnerv/` is the primary shipped substrate; `pr95_hnerv.py` is the proven 0.193-class bank; the live MPS basin IS an HNeRV-class carrier |
| **HiNeRV** ⭐ | [arXiv 2306.09818](https://arxiv.org/abs/2306.09818) (NeurIPS'23) | github.com/hmkx/HiNeRV | torch | **hierarchical encoding: bilinear-interpolated multi-resolution feature GRIDS + depthwise-conv/MLP** → much higher capacity at equal params (72.3% BD-rate over HNeRV) `[external]` | **PARTIAL** — `hi_nerv/architecture.py` (634 LOC) has hierarchical-latent injection WIRED but the feature-grid/bilinear-grid path is `use_hierarchical_feature_grid=False` + `official_core_forward_parity_proven=False` (L0 SCAFFOLD); `hinerv_as_renderer.py` is L0 LEGACY |
| **FFNeRV** ⭐ | [arXiv 2212.12294](https://arxiv.org/abs/2212.12294) (ACM-MM'23) | [maincold2.github.io/ffnerv](https://maincold2.github.io/ffnerv/) → github.com/maincold2/FFNeRV | torch | **optical-flow-guided frame aggregation** (reuse pixels from neighbor frames via learned flow) + 1D temporal grids → fully-conv; "don't waste params memorizing the same pixel across frames" `[external]` | **L0 SKETCH** — `ff_nerv/architecture.py` (290 LOC, DCT band-limited grid, real forward, never anchored); `ffnerv_as_renderer.py` (566 LOC, L1 FULL but Fourier-feature flavor, not the flow-warp) |
| **DNeRV** ⭐ | [arXiv 2304.06544](https://arxiv.org/abs/2304.06544) (CVPR'23) | github.com/QiZhao-NJU/Neural-Representation-for-Video-via-Differential-Input-and-Pyramidal-Architecture | torch | **two streams: content + FRAME-DIFFERENCE**; collaborative content unit fuses them; models inter-frame dynamics for large-motion scenes `[external]` | **NOT TOUCHED** as a substrate (no `dnerv/` dir); the diff-stream idea overlaps `nervdc_as_renderer.py` (decoder-conditioning, L1) + `ego_nerv_as_renderer.py` (pose-FiLM) |
| **SNeRV** | [arXiv 2501.01681](https://arxiv.org/abs/2501.01681) (ECCV'24) | github.com/qwertja/SNeRV | torch | **2D-DWT LF/HF decomposition**: encode only LF, GENERATE HF via decoder (MFU + HFR); temporal extension via TUBs `[external]` | **PARTIAL/CARRIER** — `snerv_inverse_steg_carrier/` (carrier, no standalone arch); the official MFU/HFR/TUB conv path is REAL-AND-COMPLETE ~70% per `snerv_fullstack_extreme_scrutiny_…` but trained scorer-blind (d_seg 0.71) + has a B3 export-binding wall |
| **BoostNeRV** | [arXiv 2402.18152](https://arxiv.org/abs/2402.18152) (CVPR'24 Highlight) | [github.com/Xinjie-Q/Boosting-NeRV](https://github.com/Xinjie-Q/Boosting-NeRV) | torch | **universal booster for ANY INR**: conditional decoder + temporal-aware affine transform (frame-index prior) + sinusoidal NeRV block + entropy-min `[external]` | **L0 SCAFFOLD** — `boost_nerv/` (287 LOC, iterative-boosting residual chain, real forward); `boost_nerv_pr110_residual/` (177 LOC, L0, residual-vs-PR110, not integrated); `boostnerv_*_sweep_results_*` dirs exist (PR110 variant sweeps, advisory) |
| **NIRVANA** | [arXiv 2212.14593](https://arxiv.org/abs/2212.14593) (CVPR'23) | github.com/UMD/snap (UMD + Snap) | torch | **autoregressive PATCH-WISE**: fit separate small nets per frame-group, init from previous group's weights; quantize during training; 12× faster encode `[external]` | **L0 SKETCH/SCAFFOLD** — `nirvana/` (303 LOC, patch-grid adaptive scheduler, real forward); `nirvana_cascading_nerv/` (design-stage, DEFERRED pending symposium); `codex_findings_nirvana_numpy_portable_inflate_*` explored the inflate path |
| **E-NeRV** | [arXiv 2207.08132](https://arxiv.org/abs/2207.08132) (ECCV'22) | [github.com/kyleleey/E-NeRV](https://github.com/kyleleey/E-NeRV) | torch | **disentangle spatial + temporal context** (decouple the coupled NeRV) → 8× faster convergence + fewer params `[external]` | **L1 FULL** — `e_nerv_as_renderer.py` (605 LOC, real encoder-decoder, Lagrangian, score-aware wired); substrate lane L0/L1 SKETCH (`_full_main` NotImplementedError) |
| **PNeRV** | [arXiv 2404.08921](https://arxiv.org/abs/2404.08921) (CVPR'24) | github.com/QiZhao-NJU/…Pyramidal-Architecture (shared w/ DNeRV) | torch | **pyramidal multi-scale**: Kronecker FC (KFc) low-cost rescaling + Benign Selective Memory (BSM) merges coarse+fine → spatial consistency `[external]` | **NOT TOUCHED** (no `pnerv/` dir); the pyramidal/multi-scale idea overlaps `mnerv_as_renderer.py` (Mallat 3-scale cascade, L1) |
| **SRNeRV** (scale-wise recursive) | [arXiv 2603.08227](https://arxiv.org/abs/2603.08227) (2026, NEW) | (repo not yet found; arXiv-only at survey time) | torch (implied) | **scale-wise recursive**: shared scale-invariant channel-mixing module recursively applied across scales + scale-specific spatial mixing → "significantly reduce model size while preserving capacity" `[external]` | **NOT TOUCHED** (brand-new 2026 paper; closest in-repo is `tcnerv_as_renderer.py` temporal-conv + `mnerv` multi-scale) |
| **SR-NeRV** (super-resolution) | [arXiv 2505.00046](https://arxiv.org/abs/2505.00046) (2025) | (arXiv-only at survey time) | torch (implied) | **super-resolution embedding efficiency**: train at low-res, SR-upsample at decode to improve embedding efficiency `[external]` | **NOT TOUCHED**; `sr_nerv_resolution_axis_enhancer_*` codex finding explored the resolution axis |
| **VQ-NeRV** | [arXiv 2403.12401](https://arxiv.org/abs/2403.12401) | github (U-shape + codebook) | torch | U-shape + codebook discretizes shallow + inter-frame RESIDUAL `(f_e−f_d)`; per-frame tokens | **L1 MLX-LOCAL** — `pact_nerv_vq/` (287 LOC, VQ-VAE van-den-Oord, EMA codebook, real); `capstone_vq_nerv` is the base_ch=20 substrate; **rate solved (22–34 KB) but distortion-blocked (d_seg 0.506)** per `pact_nerv_vq_maturity_audit_20260609` |
| **NeRV** (vanilla) | [arXiv 2110.13903](https://arxiv.org/abs/2110.13903) (NeurIPS'21) | github.com/haochen-rye/NeRV | torch | the origin: frame-index → conv/PixelShuffle/sin frame; NO content embedding | (ancestor; superseded by HNeRV's content-adaptive embedding) |
| **Block-NeRV / TC-NeRV / DS-NeRV / CNeRV / EGO-NeRV / NERVDC / M-NeRV** | various | various | torch | tile-decomposed / temporal-conv / depthwise-separable / conv-stem / pose-FiLM / decoder-conditioning / Mallat-multiscale | all **L0–L1** as_renderer adapters (424–595 LOC each, real forwards) — our internal variant zoo |
| **COIN++ / SIREN / FINER / WIRE** | [SIREN](https://arxiv.org/abs/2006.09661) / [FINER](https://liuzhen0212.github.io/finer/) | various | torch | coordinate-MLP w/ periodic/wavelet/modulation activations; NO content embedding | **L0 SKETCH** (`siren/` 284 LOC, `coin_plus_plus/` 246 LOC) — **DOMINATED** for compression (carrier memo §B.4): no per-frame embedding ⟹ bytes blow up |

**The crisp reading:** the four families that matter for THIS task (given the floor invariance) are **(spatial)** HiNeRV +
SNeRV + PNeRV/SRNeRV, **(temporal/pose)** FFNeRV + DNeRV, **(efficiency/convergence)** E-NeRV + NIRVANA, and **(universal
enhancer)** BoostNeRV. The coordinate-MLP family (SIREN/COIN++) is dominated. VQ-NeRV is our own container, not a separate
lever. SR-NeRV (super-resolution) is a poor fit for a FIXED-resolution contest (§B note).

---

## B. THE SYNERGY-AXIS SCORING (the heart — under the invariant floor)

Scoring legend per axis: **+++/++/+/0/−** (− = actively dominated/fights). The five synergy axes are the ONLY admissible
EV channels (rate-floor is excluded by the verdict). "Export feas." = full-RGB renderer (L5) + numpy-portable inflate
≤100 LOC + score-aware-trainable.

| Variant | (a) d_seg / spatial control | (b) d_pose / temporal control | (c) dashcam-structure fit | (d) slack reduction | (e) lever synergy (pose-FiLM + score-domain) | (f) trainability | (g) export feas. | **Net synergy verdict** |
|---|---|---|---|---|---|---|---|---|
| **HiNeRV** | **+++** — bilinear-grid multi-res feature grids = much finer spatial detail at equal bytes ⟹ fewer argmax flips on the thin boundary band; the canonical "more spatial fidelity per byte" mechanism | **+** — frame+patch unified; some temporal via patch coords, but NOT a flow model | **++** — multi-res grids fit recurring road/sky/building patches of ONE drive | **++** — its whole point is more capacity/byte; could shave HNeRV's distortion slack | **++** — it IS an HNeRV-class decoder; pose-FiLM + margin-weighted seg plug in unchanged | **+** — deeper/wider net is HARDER to fit on one video; risk of the same mean-field collapse if skip/grad-conditioning is wrong | **++** — full RGB renderer (L5); decoder is conv+grid-sample (numpy-portable); grid bitstream/QAT is the unbuilt part | **TOP d_seg-SPATIAL candidate.** The single most-cited reason a NeRV beats HNeRV is spatial capacity; that maps directly to the d_seg residual. PARTIAL in-repo (grid path OFF). |
| **FFNeRV** | **+** — fully-conv spatial continuity helps a little | **+++** — flow-guided aggregation explicitly models inter-frame motion = the frame_0→frame_1 delta PoseNet reads; cheap per-pair deltas | **+++** — dashcam = mostly smooth global ego-motion flow; FFNeRV's flow-warp is BUILT for exactly this temporal redundancy | **++** — flow ⊂ temporal grid is GENERATED not stored ⟹ cheap per-pair, frees slack | **++** — the flow IS the pose signal; composes with pose-FiLM (flow gives the motion, FiLM injects the readout) — potentially additive, not redundant | **+** — flow estimation adds a sub-net to fit on one drive; tractable | **+** — full renderer; the flow-warp is a numpy-portable bilinear-sample; flow regularization at compress-time | **TOP d_pose-TEMPORAL candidate.** Attacks the pose axis from the motion-model side. L0 SKETCH in-repo. |
| **DNeRV** | **+** — content stream | **+++** — explicit FRAME-DIFFERENCE stream = the inter-frame dynamics; designed for large-motion (ego-motion qualifies) | **++** — frame-diff captures the drive's motion; collaborative fusion | **+** | **++** — the diff-stream is a natural carrier for "what changed between the pair" — directly feeds the pose readout | **+** — two-stream adds params | **+** — full renderer; two-stream is numpy-portable | **STRONG d_pose alternative to FFNeRV.** Diff-stream is a cheaper-to-build pose lever than flow-warp; NOT TOUCHED in-repo (overlaps `nervdc`/`ego_nerv`). |
| **SNeRV** | **++** — DWT LF/HF: HF restorer recovers fine textures spectral-bias usually misses ⟹ sharper boundaries ⟹ fewer flips | **+** — TUB temporal extension | **+** — wavelet multiscale fits scene structure; but the official conv path is heavy | **+** — HF generated not stored; but measured d_seg 0.71 when trained scorer-blind | **+** — composes; but the LF/HF split fights the score-domain Lagrangian if HF is the boundary signal and gets generated wrong | **0** — REAL but trained scorer-blind in-repo (d_seg 0.71); B3 export-binding wall (`carrier.py:312`) | **MEDIUM (spatial, spectral-bias angle).** Real and ~70% complete but has BOTH the inert-loop bug AND a real export wall; behind HiNeRV. |
| **BoostNeRV** | **+** (booster — improves whatever base) | **+** (temporal-affine prior) | **+** | **+** — entropy-min term + better param distribution → marginal slack | **+++** — UNIVERSAL booster: composes ONTO HiNeRV/FFNeRV/the HNeRV bank as a quality+convergence multiplier, NOT a competing base | **+++** — its explicit goal is faster convergence + balanced params (buys distortion descent/compute) | **+** — conditional decoder is numpy-portable | **BEST ENHANCER (compose-last).** Not a base; a multiplier on the §D winner. Highest trainability synergy. L0 in-repo. |
| **E-NeRV** | **+** | **+** (disentangled temporal context) | **+** | **+** — fewer params at equal fidelity = some slack | **+** | **+++** — 8× faster convergence = MORE distortion descent per fixed local-MPS compute (the binding resource) | **++** — full renderer; L1 FULL adapter already exists | **BEST CONVERGENCE BACKBONE (trainability lever).** Pairs with HiNeRV's capacity: E-NeRV gives fast-fit, HiNeRV gives spatial capacity. L1 adapter ready. |
| **NIRVANA** | **+** (patch-wise local detail) | **0/+** (autoregressive group init = weak temporal) | **+** — patch recurrence of one drive | **+** — per-group small nets quantized-in-training | **0** — patch-wise autoregression is orthogonal to the global score-domain Lagrangian; awkward to FiLM | **++** — 12× faster encode; quantize-in-training is export-friendly | **+** — numpy-portable inflate explored (`codex_findings_nirvana_…`) | **MEDIUM (encode-speed, not score).** Its win is encode SPEED (PSNR-neutral), which is not the bottleneck; lower priority. |
| **PNeRV / SRNeRV** | **++** (pyramidal/scale-recursive spatial consistency) | **+** | **+** | **+** (SRNeRV: smaller model at equal capacity) | **+** | **+** (SRNeRV: recursive sharing = fewer params/faster) | **+** | **MEDIUM (spatial, behind HiNeRV).** Same spatial-capacity goal as HiNeRV but less validated for compression; SRNeRV is brand-new (2026, no repo yet). Watch, don't build first. |
| **SR-NeRV (super-res)** | **0/−** | **0** | **−** — designed for low-res-train/high-res-decode; the contest is FIXED 384×512 with a bicubic↑874→bilinear↓384→uint8 roundtrip that already does the resampling | **+** (embedding efficiency) | **0** | **+** | **0** — the SR upsampler fights the eval roundtrip | **DOMINATED for this contest.** Fixed-resolution + existing roundtrip ⟹ the SR axis adds risk, not fidelity. |
| **VQ-NeRV** | **0** — quantizing features can FLIP a SegNet argmax (needs straight-through boundary protection) | **0** | **+** | **+** — index carrier compresses; but = HNeRV floor, different container (rate solved 22–34 KB, distortion blocked d_seg 0.506) `[MEASURED]` | **+** — composes as the codec LAYER on a fixed base | **+** | **++** — production codec grammar exists (PVQ archive) | **CONTAINER, not a distortion lever.** Reuse its codebook machinery for rate; it does NOT lower d_seg. |
| **SIREN / COIN++ / FINER / WIRE** | **+** (spectral knob) | **0** | **−** | **−** — NO content embedding ⟹ bytes blow up for 1200 frames (carrier memo §B.4) | **0** — no native score path (only ω₀) | **+** | **−** — global field, no local argmax-boundary handle | **DOMINATED.** Borrow the FINER activation as a Layer-2 decoder tweak; never a standalone carrier. |

### B.1 The two-axis split (the actionable conclusion)

The distortion residual `0.07314` decomposes (carrier memo §A.4) as `100·d_seg ≈ 0.056` (the dominant distortion debt)
+ `sqrt(10·d_pose) ≈ 0.017`. So the **d_seg-spatial axis is ~3× the prize of the d_pose-temporal axis** in the residual,
and the marginal value of pose is high near d_pose→0 (CLAUDE.md operating-point analysis). The mapping is clean:

- **d_seg-SPATIAL axis (the bigger prize) → HiNeRV.** A finer spatial representation at equal bytes is, definitionally,
  fewer pixels in the wrong place — and on a SegNet whose `d_seg` lives on a thin top-2-margin boundary band, "fewer
  wrong pixels near class edges" = "fewer argmax flips." HiNeRV's hierarchical bilinear-grid encoding is the family's
  canonical spatial-capacity mechanism. **But:** the Layer-2 margin-weighted-seg lever (carrier memo Component-2 folded
  into training) already targets the boundary band on the HNeRV decoder — so HiNeRV only WINS if its capacity lowers the
  boundary-flip rate by MORE than margin-weighting already extracts from HNeRV at the same bytes. That is the open
  question the §D probe answers.
- **d_pose-TEMPORAL axis (the smaller but high-marginal prize) → FFNeRV (flow) or DNeRV (frame-diff).** Both model the
  inter-frame motion PoseNet reads. **But:** the Layer-2 pose-FiLM lever (carrier memo Component-3, MEASURED-GO) ALREADY
  collapses d_pose to the stored-pose quant floor at ~1–3 KB by INJECTING the GT pose as side-info. FFNeRV/DNeRV would
  have to lower d_pose BELOW the stored-pose floor (unlikely — side-info beats modeling near d_pose→0) OR lower it for
  FREE (no stored bytes) by modeling motion well enough that no pose codec is needed. The latter is the only real EV,
  and it is bounded by how cheaply the flow/diff sub-net fits the single drive.

---

## C. EXPLORATION-STATUS AUDIT (fully / partial / not-touched + orphaned-ready flags)

Per the anti-signal-loss non-negotiable: a node that is BOTH READY (code/scaffold exists OR first step is a $0 smoke)
AND high-EV is **ACTIONABLE-NOW**, not parked. Flags below.

| Variant | Status | Evidence (file:line) | Orphaned-ready flag |
|---|---|---|---|
| **HiNeRV** | **PARTIAL** — hierarchical-latent injection WIRED + real; **the feature-grid / bilinear-grid path is L0 SCAFFOLD, OFF, parity-unproven** | `hi_nerv/architecture.py:82` (`official_core_forward_parity_proven=False`), `:136` (`use_hierarchical_feature_grid=False`), `:556-567` (toggleable grid path + wired mid/fine injection); `hinerv_as_renderer.py` L0 LEGACY | **🟢 READY ∧ HIGH-EV ORPHAN.** The single highest-EV distortion mechanism in the family is HALF-built and never score-axis-measured. The $0 probe (§D) is the first step. |
| **FFNeRV** | **L0 SKETCH** — `ff_nerv/architecture.py` (290 LOC, DCT band-limited grid, real forward, never anchored); the FLOW-WARP mechanism itself is NOT the implemented variant (the adapter is Fourier-feature flavored) | `ff_nerv/architecture.py` (L0 SKETCH, `_full_main` NotImplementedError per registry); `ffnerv_as_renderer.py:247-257` (Fourier, not flow-warp) | **🟡 READY ∧ MED-HIGH-EV.** The flow-warp (the actual FFNeRV mechanism) is NOT built — the in-repo file is a different mechanism under the name. Building real flow-warp is a from-scratch sub-net, not a drop-in. |
| **DNeRV** | **NOT TOUCHED** (no substrate) — overlaps `nervdc_as_renderer.py` (L1, decoder-conditioning) + `ego_nerv_as_renderer.py` (L1, pose-FiLM) | no `dnerv/` dir; `nervdc_as_renderer.py:207` (prev-frame conditioning ≈ a weak diff stream) | **🟡 MED-EV.** The frame-diff idea is partly present via nervdc; a true two-stream DNeRV is unbuilt. |
| **SNeRV** | **PARTIAL/CARRIER** — official MFU/HFR/TUB conv path REAL ~70%, but trained scorer-blind (d_seg 0.71) + B3 export wall | `snerv_inverse_steg_carrier/`; `snerv_fullstack_extreme_scrutiny_vs_evaluate_py_20260609` (B1 wrong-loss, B3 export `carrier.py:312-316`) | **🟠 BLOCKED-WALL.** Has BOTH the shared inert-loop bug (fixable) AND a real export-binding wall (separate landing). Behind HiNeRV. |
| **BoostNeRV** | **L0 SCAFFOLD** — `boost_nerv/` (287 LOC real) + `boost_nerv_pr110_residual/` (177 LOC, not integrated) + advisory PR110 sweep dirs | `boost_nerv/architecture.py` (real iterative-boosting); `boostnerv_pr110_*_sweep_results_20260526/` (advisory) | **🟢 READY ∧ MED-EV (enhancer).** Composes onto any base; cheap to bolt-on once a winner exists. |
| **E-NeRV** | **L1 FULL adapter** (`e_nerv_as_renderer.py` 605 LOC real); substrate lane L0/L1 SKETCH | `e_nerv_as_renderer.py` (L1 FULL, score-aware wired); `lane_e_nerv_l0_scaffold_20260520` | **🟢 READY ∧ MED-EV (convergence backbone).** Most-complete of the non-HNeRV variants; the trainability lever. |
| **NIRVANA** | **L0 SKETCH** — `nirvana/` (303 LOC real); `nirvana_cascading_nerv/` design-stage DEFERRED | `nirvana/architecture.py`; `codex_findings_nirvana_numpy_portable_inflate_20260527` | **🟡 LOW-EV.** Win is encode-speed (not the bottleneck). |
| **PNeRV** | **NOT TOUCHED** (overlaps `mnerv_as_renderer.py` Mallat 3-scale, L1) | no `pnerv/` dir; `mnerv_as_renderer.py` (multi-scale present) | **⚪ LOW-EV** (spatial, behind HiNeRV). |
| **SRNeRV** (2026) | **NOT TOUCHED** — brand-new paper, no repo found | arXiv 2603.08227 only | **⚪ WATCH.** New 2026; closest is `tcnerv`/`mnerv`. Monitor for repo. |
| **SR-NeRV** (super-res) | **NOT TOUCHED** — `sr_nerv_resolution_axis_enhancer_*` codex finding only | `codex_findings_sr_nerv_resolution_axis_enhancer_20260601` | **⚪ DOMINATED** (fixed-resolution contest). |
| **VQ-NeRV** | **L1 MLX-LOCAL** — `pact_nerv_vq/` (287 LOC real, EMA codebook); rate solved, distortion blocked | `pact_nerv_vq_maturity_audit_for_codebook_investment_20260609` (d_seg 0.506, "dark mean-field") | **🟢 READY as the CODEC LAYER** (rate), not a distortion lever. |
| **SIREN/COIN++** | **L0 SKETCH** (`siren/` 284, `coin_plus_plus/` 246, real) | `siren/`, `coin_plus_plus/` | **⚪ DOMINATED** (no content embedding). |

**Audit headline:** we have built ADAPTERS for essentially the entire family (`e/hi/ff/m/ds/c/block/tc/ego/nervdc-NeRV`
as_renderer.py, 424–1251 LOC each — all REAL forwards, not shims), AND substrate scaffolds for `hi_nerv / ff_nerv /
boost_nerv / ds_nerv / nirvana / tc_nerv / block_nerv / siren / coin_plus_plus / pact_nerv_vq / sane_hnerv`. The fleet is
EXTENSIVELY scaffolded. The gap is NOT breadth — it is that **the two highest-EV mechanisms (HiNeRV's bilinear feature
grid and FFNeRV's flow-warp) are the PARTIAL/SKETCH ones**, and per the Task-#68 fleet-reactivation memo
(`nerv_fleet_reactivation_and_arch_selection_20260610`), the WHOLE fleet shares the d_seg≈0.5 inert-loop bug (M-loss:
harness scorer-weights default 0.0; M-arch: skip-free decoder) — so prior "this variant doesn't work" verdicts are
IMPLEMENTATION-LEVEL falsified (Catalog #307), PARADIGMS INTACT, and the variants reactivate on the loop fix. **Nothing
here is a paradigm kill; the high-EV mechanisms are simply unfinished + unmeasured on the score axis.**

---

## D. RANKED RECOMMENDATION + FIRST $0 STEP + HONEST EV BOUND

### D.1 The ranking (EV toward sub-0.15 UNDER the invariant floor — on distortion-control + dashcam-fit + lever-synergy + trainability, NOT rate)

| Rank | Candidate | Synergy axis | EV verdict | Why this rank |
|---|---|---|---|---|
| **1** | **HiNeRV hierarchical feature-grid** (finish the OFF grid path) | **d_seg-SPATIAL** (the bigger prize, ~0.056 of the residual) | **HIGHEST-but-BOUNDED** | The family's canonical spatial-capacity mechanism, mapped to the dominant distortion debt; PARTIAL in-repo (READY-orphan). **Bounded by:** must beat what margin-weighted-seg already extracts from HNeRV at equal bytes. The §D.3 probe decides. |
| **2** | **FFNeRV flow / DNeRV frame-diff** (build the motion model) | **d_pose-TEMPORAL** (smaller prize, high marginal near d_pose→0) | **MEDIUM-HIGH-but-BOUNDED** | Attacks pose from the motion-model side. **Bounded by:** pose-FiLM (MEASURED-GO) already collapses d_pose at ~1–3 KB stored side-info — FFNeRV/DNeRV only win if they lower d_pose for FREE (no stored bytes) by modeling motion. Real EV only if the flow/diff sub-net fits the drive cheaply. |
| **3** | **BoostNeRV booster** ⊕ **E-NeRV backbone** (compose onto the rank-1/2 winner) | **trainability + slack** | **MEDIUM (multiplier)** | Not bases. E-NeRV's 8× convergence buys more distortion descent per fixed local-MPS compute (the binding resource); BoostNeRV's conditional-decoder + entropy-min is a quality+convergence multiplier on whatever base wins. Compose LAST. |
| **4** | **SNeRV DWT LF/HF** (spectral-bias angle for d_seg) | d_seg-spatial (alt) | **MEDIUM, behind #1** | Real ~70% but carries BOTH the inert-loop bug AND a real B3 export-binding wall — two landings vs HiNeRV's one. Only if HiNeRV's grid path falsifies. |
| **5** | **VQ-NeRV codec layer** | rate container | **LOW (rate, not distortion)** | Reuse its codebook machinery for the rate term on the §D winner's weights; does NOT move d_seg. Behind the distortion work. |
| — | **NIRVANA / PNeRV / SRNeRV / SR-NeRV / SIREN/COIN++** | — | **LOW / WATCH / DOMINATED** | encode-speed-not-score / behind-HiNeRV / new-no-repo / fixed-res-mismatch / no-content-embedding. Borrow parts, don't build first. |

### D.2 The honest EV bound (NO-FAKE — this is the load-bearing caveat)

**The EV of ANY carrier-bias swap is bounded above by the distortion residual `0.07314`, and in practice much smaller.**
HNeRV sits NEAR the scorer-conditional floor (its overfit weights are near-MDL, carrier memo §C.2). A better spatial bias
(HiNeRV) or temporal bias (FFNeRV/DNeRV) **cannot lower the rate floor** — it can ONLY shave the d_seg / d_pose residual
at equal bytes. AND the Layer-2 levers already attack that residual on the EXISTING HNeRV bank (margin-weighted seg for
d_seg; pose-FiLM for d_pose, MEASURED-GO). So the marginal EV of a variant is **only the d_seg/d_pose it shaves BEYOND
what the levers already capture, at MATCHED bytes.** That margin is plausibly small and could be zero (the levers might
already saturate the boundary-flip reduction the extra capacity would buy). **A variant is worth its from-scratch build
cost ONLY if a $0 head-to-head proves a non-trivial matched-byte d_seg (or d_pose) gap.** Stated as a falsifiable claim:

> **N1 (falsifiable):** At matched archive bytes on the basin task, HiNeRV's hierarchical feature-grid decoder lowers the
> exact d_seg (boundary-flip rate) by ≥20% vs the HNeRV decoder WITH margin-weighted-seg already applied. *Falsified by:*
> a head-to-head where the grid decoder's matched-byte d_seg is within noise of (or worse than) margin-weighted HNeRV —
> in which case the d_seg win is already captured in-training and HiNeRV adds risk, not EV (rank-1 collapses to "don't
> build; the lever is the ceiling").

> **N2 (falsifiable):** FFNeRV/DNeRV motion-modeling lowers d_pose for FREE (zero stored pose bytes) below the
> stored-pose-FiLM floor at equal total bytes. *Falsified by:* pose-FiLM's ~1–3 KB stored side-info beating the
> free-modeling d_pose — in which case the pose axis is already solved by side-info and the temporal variant adds no EV.

### D.3 The first $0 step (MVP-first, the decisive matched-byte distortion measurement)

**The single highest-value $0 probe is a HiNeRV-grid-vs-HNeRV matched-byte d_seg head-to-head on the FROZEN basin
checkpoint** — because it answers the one question the whole ranking reduces to (N1): *does extra spatial capacity lower
the boundary-flip rate at equal bytes by more than margin-weighting already does?* Concretely (NO GPU, NO basin
contention — read the frozen fork-point checkpoint exactly as the pose-FiLM disambiguator did):

1. **Read the frozen basin checkpoint** (`experiments/results/torch_vehicle_full_mps_basin_bc20_n600` fork-point;
   read-only, no daemon touch) → its HNeRV decoder + per-pair latents + the `seg_out` the eval already produces.
2. **Build a param-matched HiNeRV-grid decoder head** (CPU/numpy reference; finish the OFF `use_hierarchical_feature_grid`
   path in `hi_nerv/architecture.py` as a *reference forward only*, NOT a trainer) at the SAME byte budget as the basin
   HNeRV decoder. Overfit BOTH on a small N (e.g. 8–16 pairs) with identical compute.
3. **Measure exact d_seg at MATCHED bytes** for: (a) HNeRV-decoder baseline, (b) HNeRV-decoder + margin-weighted-seg,
   (c) HiNeRV-grid decoder, (d) HiNeRV-grid + margin-weighted-seg. The decisive comparison is (c)/(d) vs (b): does the
   grid's spatial capacity lower the boundary-flip rate BEYOND margin-weighting?
4. **Round-trip check:** push the best render through bicubic↑874 → bilinear↓384 → uint8 and re-measure d_seg — confirm
   the spatial-capacity gain SURVIVES the eval roundtrip (a finer detail erased by the resize blur is no gain).

**Falsifiable threshold:** N1's ≥20% matched-byte d_seg reduction (survived) = GO for finishing HiNeRV as the rank-1
distortion carrier; <20% (or noise) = the margin-weighted lever is the ceiling, HiNeRV adds risk-not-EV, and the program
stays on the HNeRV-bank + Layer-2-levers path (carrier memo §F rank-1). A sister cheaper variant of this probe for the
pose axis (N2): on the same checkpoint, A/B a tiny flow-warp/frame-diff sub-net vs the stored-pose-FiLM floor and measure
free-modeling d_pose vs side-info d_pose at equal bytes.

### D.4 How it composes (full-stack synergy)

The carrier (Layer 1) feeds the Layer-2 levers and Layer-3 bolt-ons as ONE co-designed system:
- **HiNeRV (if GO) is a DROP-IN-ISH decoder swap** into the existing `torch_vehicle`/PR95 driver — the encoder/latent
  split, the score-aware loss, the EMA/eval-roundtrip/diff-YUV6 plumbing, and ALL five Layer-2 levers (score-domain
  Lagrangian, pose-FiLM, margin-weighted seg, score-aware QAT, lossless recodes) plug into an HNeRV-class decoder
  unchanged. The grid path is a decoder-internal change, not a new substrate. **This is the cheap-to-try property** that
  makes HiNeRV rank-1 over a from-scratch FFNeRV flow-warp (which is a new sub-net).
- **FFNeRV/DNeRV (if GO) compose with pose-FiLM:** the flow/diff gives the motion structure; pose-FiLM injects the
  readout — potentially additive (motion-model + side-info) rather than redundant, IF the free-modeling d_pose clears
  the N2 bar.
- **BoostNeRV ⊕ E-NeRV compose LAST** as a convergence/quality multiplier on the rank-1/2 winner — never as a competing
  base (per the canonical leaderboard binding-depth discipline: bind all ingredients into ONE coherent carrier).
- **VQ-NeRV's codebook is the Layer-3 rate container** on the winner's weights (rate solved 22–34 KB), applied after the
  distortion is in the cell.

The binding constraint order is **distortion first** (HiNeRV grid + pose-FiLM + margin-seg drive the 0.07314 residual →
~0, crossing T_3), **then rate** (VQ codebook + lossless recodes push toward T_floor). This matches the carrier memo §D.4
phasing exactly — the survey CONFIRMS no carrier swap is needed for the rate term, and identifies HiNeRV (spatial) +
FFNeRV/DNeRV (temporal) as the only two variants whose *distortion-control* bias could shave the residual beyond the
levers, with the $0 probe as the decide-don't-defer gate.

---

## Wire-in hooks (CLAUDE.md 6-hook per Catalog #125)

1. **Sensitivity-map — ACTIVE (design):** new prior — the family's distortion-control EV concentrates on TWO axes
   (HiNeRV→d_seg-spatial, FFNeRV/DNeRV→d_pose-temporal), and d_seg is ~3× the residual prize; the grid path is the
   READY-orphan to probe first.
2. **Pareto — ACTIVE:** adds the constraint that a variant's EV is bounded by the distortion residual (0.07314) AND
   capped by what the Layer-2 levers already extract at matched bytes — a variant must clear the matched-byte d_seg/d_pose
   gap, not a PSNR-RD claim.
3. **Bit-allocator — N/A (no archive emitted); the §D probe's matched-byte protocol IS a future bit-allocator prior.**
4. **Cathedral autopilot — N/A** (survey; no archive-deployable artifact).
5. **Continual-learning — ACTIVE:** reseeds the planner with (a) carrier-swap-for-rate EV ≈ 0 (floor invariant,
   confirmed); (b) the two distortion-axis variants (HiNeRV/FFNeRV-DNeRV) + their EV BOUND (must beat the levers at
   matched bytes); (c) the READY-orphan flag on HiNeRV's grid path; (d) N1/N2 as falsifiable anchors for the $0 probe.
6. **Probe-disambiguator — ACTIVE:** the §D.3 HiNeRV-grid-vs-margin-HNeRV matched-byte d_seg probe IS the disambiguator
   between "HiNeRV is the rank-1 distortion carrier" and "the margin-weighted lever on the HNeRV bank is the ceiling."

**Mission contribution:** `frontier_breaking_enabler` (a survey that ranks the family's distortion-control synergy under
the invariant floor, flags the highest-EV READY-orphan, names the single $0 probe that decides build-vs-defer, and bounds
the EV honestly). **Frontier UNMOVED 0.19109982.** No score asserted. No GPU launched. No paid spend. No collision with
running agents.

---

## Sources (WebSearch, cited inline)

- HNeRV — [arXiv 2304.02633](https://arxiv.org/abs/2304.02633) (CVPR'23, content-adaptive embeddings).
- **HiNeRV** — [arXiv 2306.09818](https://arxiv.org/abs/2306.09818) (NeurIPS'23; bilinear-interp hierarchical positional
  encoding + depthwise-conv/MLP; 72.3% BD-rate over HNeRV on UVG). Repo: github.com/hmkx/HiNeRV.
- **FFNeRV** — [arXiv 2212.12294](https://arxiv.org/abs/2212.12294) (ACM-MM'23; flow-guided frame aggregation + 1D
  temporal grids). Repo: [maincold2.github.io/ffnerv](https://maincold2.github.io/ffnerv/).
- **BoostNeRV** — [arXiv 2402.18152](https://arxiv.org/abs/2402.18152) (CVPR'24 Highlight; universal booster, conditional
  decoder + temporal-affine + sinusoidal block + entropy-min). Repo: [github.com/Xinjie-Q/Boosting-NeRV](https://github.com/Xinjie-Q/Boosting-NeRV).
- **NIRVANA** — [arXiv 2212.14593](https://arxiv.org/abs/2212.14593) (CVPR'23; autoregressive patch-wise, group-init,
  quantize-in-training, 12× faster encode).
- **E-NeRV** — [arXiv 2207.08132](https://arxiv.org/abs/2207.08132) (ECCV'22; disentangled spatial/temporal, 8× faster
  convergence). Repo: [github.com/kyleleey/E-NeRV](https://github.com/kyleleey/E-NeRV).
- **DNeRV** — [arXiv 2304.06544](https://arxiv.org/abs/2304.06544) (CVPR'23; content + frame-difference two streams).
  Repo: github.com/QiZhao-NJU/Neural-Representation-for-Video-via-Differential-Input-and-Pyramidal-Architecture (shared w/ PNeRV).
- **PNeRV** — [arXiv 2404.08921](https://arxiv.org/abs/2404.08921) (CVPR'24; pyramidal, KFc + BSM, spatial consistency).
- **SNeRV** — [arXiv 2501.01681](https://arxiv.org/abs/2501.01681) (ECCV'24; 2D-DWT LF/HF, encode-LF-generate-HF, MFU +
  HFR + TUB). Repo: github.com/qwertja/SNeRV.
- **SRNeRV** (scale-wise recursive) — [arXiv 2603.08227](https://arxiv.org/abs/2603.08227) (2026; scale-invariant
  recursive channel-mixing, parameter-efficient). NEW; repo not yet found at survey time.
- **SR-NeRV** (super-resolution) — [arXiv 2505.00046](https://arxiv.org/abs/2505.00046) (2025; low-res-train + SR-decode
  embedding efficiency). DISTINCT from SRNeRV.
- **VQ-NeRV** — [arXiv 2403.12401](https://arxiv.org/abs/2403.12401) (U-shape + codebook residual discretization).
- **NeRV** (vanilla) — [arXiv 2110.13903](https://arxiv.org/abs/2110.13903) (NeurIPS'21).
- **SIREN** — [arXiv 2006.09661](https://arxiv.org/abs/2006.09661); **FINER** — [liuzhen0212.github.io/finer](https://liuzhen0212.github.io/finer/).

## Cross-references (in-repo)

- `layer1_carrier_first_principles_20260612T171912Z.md` (commit `2350b6b2e`) — the invariant-floor verdict this survey
  runs under; §A.4 distortion residual = 0.07314; §F ranks the HNeRV-bank + Layer-2 levers path.
- `nerv_fleet_reactivation_and_arch_selection_20260610T192434Z.md` (Task #68) — the fleet-wide inert-loop bug (M-loss +
  M-arch), the smaller-basis fusion recommendation, the reactivation list; the maturity backbone for §C.
- `capacity_verdict_smaller_basis_by_rate_REFUTED_pivot_to_waterfiller_20260611.md` — the MEASURED refutation of the
  carrier-as-rate-lever thesis (D.2 corroboration).
- `pact_nerv_vq_maturity_audit_for_codebook_investment_20260609.md` — VQ rate-solved/distortion-blocked; the codec-layer
  role.
- `snerv_fullstack_extreme_scrutiny_vs_evaluate_py_20260609.md` — SNeRV REAL ~70% but inert-loop + B3 export wall.
- `src/tac/substrates/hi_nerv/architecture.py:82,136,556-567` — the HiNeRV grid path L0 SCAFFOLD (the READY-orphan).
- `docs/vehicle_operating_system.md` — the maturity ladder + the inactive-objective Mistake-B fleet-wide.
