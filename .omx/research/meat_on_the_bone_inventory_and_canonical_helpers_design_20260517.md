# Meat-on-the-bone inventory + canonical-helper namespace design

**Date:** 2026-05-17
**Status:** active research memo; consumed by subagents implementing §7.5 + §7.6 build queue
**Lane:** `lane_meat_on_the_bone_inventory_and_canonical_helpers_design_20260517`
**Frontier baseline:** fec6 archive `6bae0201` at `0.19205 [contest-CPU GHA Linux x86_64]` / `0.22621 [contest-CUDA T4]`

## Provenance

Originated as §7.5 + §7.6 in `docs/pr_writeups/cpu_frontier_fec6_20260517.md`. Moved here to keep the contest PR writeup focused on the submission narrative. The writeup now carries a 1-paragraph summary + pointer back to this memo.

This memo is the canonical inventory of (a) dimensions held at a single config across the video in the current fec6 stack and (b) the design for the missing `tac.*` canonical helper namespaces that would make subsequent substrate work compose all the optimizations for free.

## §1 — Operator framing

Verbatim 2026-05-17: *"What we are learning is that even within PR95 and PR101 and PR 106 and all leaderboard families is that anything that applies as single config to each pixel or frame or pair or bit or byte across the entire video and uses arbitrariness even if asymmetric are leaving meat on the bone"*.

Verbatim 2026-05-17 (continued): *"All techniques should be available through helpers even if not applicable immediately because we have several other things to build and implement and iterate on so each is uniquely and individually fully and completely extreme and absolute optimization"*.

Verbatim 2026-05-17 (continued): *"There are likely other meat on the bone categories i haven't thought of like master gradient and others you should think about, maybe in concert with or unlocked by multipass or other optimizations and Boston's [boosting] and layers and layers of stacks and such"*.

Verbatim 2026-05-17 (continued): *"You can use decorators and other pythonic techniques to allow for beautiful and elegant creative expressive composable abstractions and production hardened world class OSS"*.

## §2 — Definitions

- **Global config:** one parameter setting applied uniformly across the video for a given dimension. Example: a single fp16 scale per tensor for ALL of that tensor's int8 mantissa bytes; a single Brotli quality for ALL streams; a single per-frame `crf` for the mask video.
- **Per-element conditional config:** the parameter is allowed to vary along that dimension based on local content. Example: per-pair palette selection (fec6 K=16 per pair); per-stream codec selection (*magic codec*); per-pair latent additive correction (PR106 `format0d` two-pass).

The cross-family empirical anchor (the wins observed):

| substrate | global config (over-charging) | per-element conditional (the win) | empirical effect |
|---|---|---|---|
| PR101 GOLD | uniform Brotli on `decoder.bin` | n/a (baseline) | `~0.193 [contest-CPU]` |
| fec6 (this submission) | uniform K=8 palette per pair | K=16 modes selected per pair from a fixed palette | `-0.001 [contest-CPU]` over PR101 |
| PR106 `format0d` | uniform single-correction grammar | two-pass additive corrections selected per pair | `-0.024 [contest-CUDA]` over PR101 CUDA |
| *magic codec* (PR106 r2 grammar) | uniform Brotli on all streams | per-stream codec selection across ~16 primitives | `-5 to -10%` rate across mixed-entropy streams |

## §3 — Inventory of global configs (categories A–J)

### A. Quantization (post-training-vs-QAT gap)

fec6 currently uses post-training int8 mantissa + per-tensor fp16 scale on `decoder.bin`. Quantizr's published 5-stage staircase (anchor → finetune → joint → QAT → final) ends with FP4 codebook QAT.

| dimension | current (global) | per-element / better alternative | exists in repo? |
|---|---|---|---|
| Quantization scheme | post-training int8 | QAT to FP4 (Quantizr 5-stage canonical) | `src/tac/training_curriculum/quantizr_5_stage_staircase.py` ✓; not applied to fec6 |
| Per-tensor fp16 scale | uniform across all int8 bytes in a tensor | per-block-within-tensor (block-FP, AbsMax-block, Hessian-weighted block) | `tac.codec.block_fp_codec` ✓; not applied to fec6 |
| Per-tensor scale assignment | one scale per tensor | per-output-channel (standard modern QAT) | partial in `tac.quantization`; not wired for fec6 |
| Quantization precision per layer | uniform int8 across all tensors | mixed-precision per-layer | `src/tac/sensitivity_map.py` (task #275) exists; not consumed by fec6 quantizer |
| Quantizer shape | uniform scalar | Lloyd-Max optimal scalar (or k-means weight clustering) | `tac.codec.weight_clustering` ✓; not applied to fec6 |
| BN folding | not done on fec6 export | fold BN into preceding conv before quantization | not implemented in fec6 export path |
| Hessian-weighted precision | not used | Ω-Hessian + block-FP | `tac.codec.hessian_block_fp_codec` ✓ (Selfcomp uses); not applied to fec6 |

### B. Entropy coding on every payload

Each archive member uses a single coder selected at build time. The *magic codec* per-stream auto-selector (task AA) exists but isn't applied to fec6 per-stream.

| payload | current coder | better coder(s) | exists? |
|---|---|---|---|
| `decoder.bin` (int8 mantissa) | none (raw int8) | per-tensor arithmetic coding on the int8 distribution | `tac.packet_compiler.single_tensor_ac` ✓; not applied |
| `decoder.bin` (overall stream) | fixed wrapper, no compressor | *magic codec* per-stream selection | `tac.packet_compiler.magic_codec_auto_select` ✓; applied to PR106 r2, not fec6 |
| `latents.bin` (uint8 temporal deltas) | raw uint8 deltas | hyperprior on latents (Ballé 2018) or per-dim AC | `tac.codec.balle_hyperprior` ✓ (nscs03 substrate uses); not applied to fec6 |
| `selector.bin` (K=16 codes) | fixed-Huffman uniform across 600 pairs | context-adaptive AC (PPM/CABAC) | not implemented |
| `poses.pt` | fp16 raw | arithmetic-coded pose deltas (PD-V2) | `tac.codec.pd_v2` ✓; not applied to fec6 |
| `masks.mkv` | AV1 single CRF | entropy-coded lossless mask format | `src/tac/mask_entropy_coder.py` ✓; not applied to fec6 |

### C. Per-element-conditional codes (extension of the operator's §7.4 framework)

Even where we use a non-trivial coder, the coder's parameters are uniform across the video.

| dimension | current (uniform) | per-element-conditional | route |
|---|---|---|---|
| K=16 palette | shared across all 600 pairs | per-frame-class palette (highway / urban / parking, switched on a SegNet-class summary) | offline clustering on SegNet outputs; ~$0 build |
| Selector Huffman code | fixed across all 600 selector indices | per-region-class code (partitioned by motion magnitude or SegNet class) | offline distribution analysis; ~$0 build |
| `masks.mkv` CRF | one CRF for the whole video | per-frame CRF based on SegNet sensitivity | per-frame SegNet analysis pass; ~$0 build |
| `masks.mkv` per-class | one CRF across all 5 SegNet classes | per-class CRF (boundary classes need more precision) | `masks` codec extension; ~$0 build |
| Mask resolution | uniform 384×512 | per-class adaptive resolution | `masks` codec + inflate-time upsample variants; ~$0 build |
| Pose precision | fp16 across all 600 pairs | per-pair adaptive precision (still pairs fp8, fast-motion fp16) | per-pair motion-magnitude analysis; ~$0 build |
| Per-tensor scale | uniform across bytes within a tensor | per-block-within-tensor scale (block size 64–256 elements) | direct extension of `tac.codec.block_fp_codec`; ~$0 build |

### D. Training-time global configs

| dimension | current (uniform) | per-element-conditional | route |
|---|---|---|---|
| Rate-distortion λ_R | one value across all pairs | per-pair adaptive (more rate to high-distortion pairs) | per-pair training-loop scheduler; small refactor |
| Loss weight ratio (PoseNet / SegNet) | uniform across pairs | sensitivity-weighted per-pair (master gradient §H) | requires master-gradient anchor first |
| EMA decay | uniform 0.997 across all parameters | per-tensor adaptive based on observed stability | per-tensor EMA shadow; small refactor |
| Learning-rate schedule | cosine across all epochs and tensors | Muon for matrix tensors + AdamW for vector tensors (PR95 Phase 2-4) | `experiments/train_substrate_pr95_phase_2_4.py` queued (task #608); not applied to fec6 |
| Eval-roundtrip noise σ | one σ across all pairs | per-pair adaptive | small refactor |
| Scorer-input preprocessing | uniform across the video | LoRA adapter per-video / per-pair fine-tune | LoRA harness; partial in `tac.training` |

### E. Designed-but-not-applied-to-fec6 primitives

| primitive | applied to | not applied to fec6 because | how to wire |
|---|---|---|---|
| Ω-W water-filling | Selfcomp / Lane Ω-W-V2 | fec6's decoder coder is int8 + fp16, not water-filling-eligible format | needs fec6 export-format adapter |
| Hessian-block-FP | Selfcomp | same as above | `tac.codec.hessian_block_fp_codec` exists; needs adapter |
| DP1 stacking (composition) | none yet | composition helpers exist but zero empirical anchors | subagent `afcada2203ab5b774` landed PATH 1 dispatch-ready; sha256 `507d2a000e...`, $1.90 envelope |
| DP1 as training-time prior | none yet | reformulated PATH 2 per PV-6: DP1 is frame-space (not weight-space); applies to `pr101_lc_v2_clone` decoder RGB output | scaffold landed (research_only=true); Phase 2 council deliberation pending |
| Master gradient → score-aware selector training | none yet | extractor just unblocked at protocol layer | downstream of one Modal CPU extraction (~$1) or local M5 Max (~$0) |
| NSCS01 nullspace-split (frame_0 in SegNet nullspace per `upstream/modules.py:108`) | NSCS01 substrate | fec6's selector optimizes pose-axis only; needs frame_0-only SegNet-orthogonal byte slot in grammar | the SA02 floor-unlocker from grand-council T3; ~$0 design + ~$5-15 dispatch |
| Sensitivity-map axis weights | `src/tac/sensitivity_map.py` exists | not consumed by fec6 selector-discovery loop | consumer wire-in into `tools/build_pr101_frame_exploit_selector_packet.py` |
| Substrate composition matrix (~7,834 cells) | matrix exists | only ~dozens empirically validated | fec6 × {DP1, NSCS01, wavelet residual, magic codec, SABOR} cells each ~$1-5 |
| Per-tensor magic codec | magic codec runs per-stream | not extended to per-tensor-within-stream | trivial extension to `tac.packet_compiler.magic_codec_auto_select`; grand council T3 #1 |
| Engineered corrections (per-pair / per-pixel hand-designed residuals — LANE GE / HM / CG / DI / SI-V3 family) | lane scaffolds exist | fec6 grammar has no engineered-correction slot | grammar extension for per-pair correction stream; ~$0 + ~$2-8 dispatch |
| Post-filter at inflate time | `experiments/train_postfilter_on_renderer.py` exists | fec6 inflate.py is pure decoder pass | train + bake postfilter into inflate.py per Catalog #146; ~$5-20 retrain + ~$0 wiring |
| Wavelet residual coding | wavelet substrate scaffolds exist | fec6 has no wavelet residual slot | grammar extension; ~$0 build + ~$2-8 paired-axis dispatch |

### F. Search-method coverage

| dimension | current sweep depth | full sweep | tool/route |
|---|---|---|---|
| Selector palette K | `K ∈ {4, 8, 16}`; chose 16 | exhaustive `K ∈ {2, …, 64}` × per-pair palette assignment | local macOS-CPU fan-out (M5 Max); §4.4 |
| Per-pair mode selection | greedy per-pair vs frozen palette | joint optimization of palette + per-pair assignment (alternating projections / k-means / E-M) | sister of `tac.codec.weight_clustering`; CPU-bound |
| Fixed-Huffman code | one Huffman from the global K=16 distribution | per-region-class Huffman codes | offline distribution analysis; ~$0 |
| AV1 CRF | `crf ∈ {28, 32, 36, 40, 44}`; hand-picked | full `crf ∈ [20, 63]` per-frame per-class sweep + Lagrangian dual | `tools/mask_encoding_sweep.py` (planned); CPU + ffmpeg |
| Latent quantization step | one step per dim chosen by min-MSE | per-dim per-frame-class step + cross-dim correlation | sister of `tac.codec.balle_hyperprior` |
| Bayesian / CMA-ES over the joint manifold | not used | `optuna` / `cmaes` search with master gradient as side info | tasks #400 + #401 + #405 wired bridges; not yet targeted at fec6 |

### G. Compress-time and inflate-time compute budgets

Compress-time compute is effectively unbounded. We have not exploited this:

| compress-time technique | current | unlimited-compute alternative | route |
|---|---|---|---|
| TTO at compress time | pose only (PD-V2) | TTO on every learnable parameter (selector indices, palette entries, latent scales, mask CRF schedule) with arbitrarily many iterations | extend `optimize_poses.py` template into generic TTO harness |
| Multipass refinement | single-pass build | `Lane 8 multi-pass` (iterative: quantize → measure → re-quantize residual → re-measure) | `Lane 8` Level 3 on different substrate; not yet applied to fec6 |
| Per-pair coordinate search | partial (task #466 + #470) | exhaustive across mode × palette-entry × pose-delta product | M5 Max fan-out 12-core parallel; CPU-bound |
| Simulated annealing on discrete codes | not used | SA on selector indices using master gradient as energy landscape | requires `tac.optimization` SA primitive; not built |
| Per-pair λ_R bisection to rate budget | not used | bisection per pair to maximize per-pair distortion budget while respecting total rate | small scheduler refactor |
| Iterated bisection on quantization knee | not used | per-tensor bisection on int8 scale + per-block-within-tensor scale | trivial CPU fan-out |
| LoRA per-pair fine-tune at compress time | not used | per-pair LoRA adapter trained at compress, baked into per-pair selector at inflate | requires per-pair LoRA harness |

Inflate-time budget is **30 minutes on T4**. Current fec6 inflate runs in <5 min on Modal T4. The unused ~25 min is a free resource subject to: (i) Catalog "strict scorer rule" forbids loading PoseNet/SegNet at inflate, (ii) inflate must produce same bytes deterministically per run.

| inflate-time technique | current | 30-min-budget alternative | route |
|---|---|---|---|
| Per-frame post-processing | none beyond decoder forward | deterministic denoiser / sharpener / wavelet post-filter (no scorer access; image-domain priors only) | per-frame post-filter network baked into inflate.py per Catalog #146; ~$0 |
| Per-pair pose refinement using motion-only prior | none | gradient descent on per-pair pose using ONLY temporal-coherence prior (SE(3) smoothness across pairs) | requires per-pair refinement loop in inflate.py; ~$0 |
| Multi-archive ensemble at inflate | none | carry multiple decoder variants; inflate-time-select per-frame using deterministic image-domain criterion (no scorer) | per-variant slot in grammar + selection rule; ~$2-5 |
| Wavelet residual stacking at inflate | none | multiple wavelet residual variants in archive, summed selectively at inflate per deterministic rule | grammar extension for stacked-residual slots |
| Mask post-processing temporal smoothing | none | per-pair mask temporal-coherence smoothing using only the previous pair's mask (no scorer) | mask-codec extension; ~$0 |
| Inflate-time TTO without scorer | none | optimize latents using frozen image-domain prior baked into the archive | design memo needed |

### H. Master-gradient-as-meta-axis

The master gradient `G[byte_i, term_j] = (∂S/∂byte_i)_{term_j}` for `term_j ∈ {seg, pose, rate}` is a per-byte score sensitivity. Once it exists for fec6 (subagent `a2aa62263236144ee` just unblocked the extractor), **every existing optimization becomes per-byte sensitivity-weighted** rather than uniform. This is meta to A-G — it re-parameterizes the per-element dimensions by their per-byte score contribution.

| meta-optimization | uniform form | sensitivity-weighted form | unlocks |
|---|---|---|---|
| Per-byte quantization precision | uniform int8 across all weight bytes | more bits where `|G[byte]|` is large (Lagrangian: bits ∝ sensitivity) | A meta-extension |
| Per-byte coding budget | uniform entropy budget | per-byte target rate `r(b) ∝ |G[byte]|^α` (water-filling with sensitivity weighting) | B meta-extension |
| Per-byte selector freedom | K=16 modes per pair, uniform | more modes where `Σ |G[byte]|` for that pair is large | C meta-extension |
| Per-byte loss weighting at training | uniform pixel-MSE | per-byte weighted MSE = `Σ G[byte]² · (byte_pred − byte_target)²` | D meta-extension |
| Sensitivity-aware multipass | uniform refinement | Pass N+1 refines only bytes where Pass N's residual sensitivity exceeds threshold | G meta-extension |

Master gradient is the principled answer to "which bytes deserve more bits / modes / passes." Without it, every per-element code in A-G uses a hand-chosen proxy (Hessian diag, magnitude, position) instead of the actual `∂S/∂byte`. Symposium §3.6 use-1 (score-aware loss at byte-grain) is the canonical form of NSCS01's manual per-frame-head gradient routing.

### I. Boosting / ensemble / stack-of-stacks

| ensemble form | current | refined form | route |
|---|---|---|---|
| Residual cascade | single decoder, no cascade | decoder_2 corrects decoder_1's residual; decoder_3 corrects decoder_2's; each stage adds bytes ∝ remaining distortion | extension of PR106 `format0d`'s 2-pass additive correction to N passes |
| Per-pair decoder ensemble | one decoder for all 600 pairs | M decoder variants in archive; per-pair selector picks the best at inflate via deterministic image-domain criterion | grammar extension; cost = M × decoder size |
| Mode ensemble | K=16 modes per pair (one decoder evaluated K ways) | K=16 modes × M=8 decoders = 128 per-pair candidates | extends fec6 selector grammar |
| Composition matrix (stack-of-stacks) | ~dozens of cells validated | substrate composition matrix (Catalog #528 / #566) enumerates ~7,834 (substrate × bolt-on × bolt-on × bolt-on); ≤1% validated | per cell ~$1-5; priority-rank by master-gradient predicted ΔS |
| Pareto-frontier-aware stack growth | greedy stack-by-largest-predicted-ΔS | each stack increment must Pareto-improve `(rate, distortion)`; reject increments that improve one at the cost of the other | Pareto-front tracker per Catalog #259 + #275 |
| Cross-paradigm wrapper-stacking | not used | NSCS01 nullspace-split as outer wrapper around (DP1 composition × magic codec × fec6) | high-design-cost; requires class-shift research |

### J. Side information / pre-processing / per-pair input conditioning

The scorer can't be loaded at inflate (strict-scorer rule), but we can compute scorer-relevant features at compress and bake into the archive as Wyner-Ziv side information.

| dimension | current (uniform) | per-element-conditional | route |
|---|---|---|---|
| Input pre-processing | uniform across the video | per-pair temporal denoise + gamma/exposure normalization tuned to that pair's content | per-pair pre-processor; CPU-bound at compress |
| Optical-flow side-information | not used in fec6 | RAFT-derived per-pair optical flow baked into archive as side info; decoder uses to predict frame_1 from frame_0 (Wyner-Ziv) | `train_substrate_*_lapose` / RAFT lanes exist; not composed with fec6 |
| openpilot ego-motion features | not used | openpilot calibration / ego-motion as per-pair side info (Catalog #461) | requires openpilot feature-extractor at compress; sub-KB per pair |
| Per-pair SegNet class summary | not used | low-bit SegNet class summary baked into archive; decoder allocates detail per region | SegNet pass at compress; ~1KB per pair |
| Per-pair conditional-entropy bound | not computed | per-pair MDL lower bound on achievable rate; allocate per-pair λ_R to hit it | per-pair entropy estimator; CPU at compress |
| Decoder architecture per pair | one decoder for all pairs | per-pair-class architecture (different decoder for highway / urban / parking) | per-class decoder + class-selector in grammar |
| Knowledge distillation chain | single teacher → single student | multi-teacher ensemble distillation; cross-axis distillation (CUDA teacher → CPU-axis student) to pre-emptively close §4.1 gap | distillation harness extension; `train_distill.py` template |

## §4 — Tally

- ~25 single-config dimensions (A-D)
- ~12 already-implemented primitives unwired for fec6 (E)
- ~6 search-method coverage gaps (F)
- ~13 compress-time / inflate-time budget gaps (G)
- ~5 master-gradient meta-axis amplifications (H)
- ~6 boosting / stack-of-stacks compositions (I)
- ~7 side-information / pre-processing / per-pair architecture rows (J)
- ~5 canonical-helper namespaces missing (§5 below)

**Total: ~75 rows.** Roughly half need only existing repo tools wired in (~$0-20 each); half require new helper namespaces (§5; one-time build then amortized across every future substrate).

## §5 — Canonical-helper namespace design

Per the operator standing directive `feedback_consolidate_everything_into_meta_layer_or_canonical_helpers_standing_directive_20260515.md` + CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": every technique enumerated in §3 should land as a canonical `tac.*` helper even when not immediately applicable to fec6, so the next substrate inherits the optimization automatically rather than re-implementing.

### §5.1 — Existing namespaces (coverage map)

| namespace | covers | missing |
|---|---|---|
| `tac.packet_compiler.*` | per-stream codec auto-selection (~16 primitives) | per-tensor-within-stream extension; per-byte sensitivity weighting (consumer of `tac.master_gradient`) |
| `tac.codec.*` | block-FP, hessian-block-FP, weight-clustering, balle-hyperprior, single-tensor AC, water-filling | per-pair adaptive precision; per-class CRF; per-block-within-tensor scale; LoRA per-pair compress-time adapter |
| `tac.optimization.*` | macOS-CPU advisory, MPS research signal, substrate composition matrix | simulated annealing on discrete codes; CMA-ES / `optuna` over joint manifold (F); per-pair coordinate search; iterated bisection on quantization knee |
| `tac.sensitivity_map.*` | axis-level reweighting API (Catalog #586) | per-byte master-gradient consumer that converts `G[byte, term]` into per-byte quant precision / coding budget / selector freedom (H) |
| `tac.master_gradient.*` | per-byte score gradient with operating-point cache | the 5 H-row consumers — each a wrapper that takes `(G, archive_bytes, policy)` and returns the parameterized output |
| `tac.training_curriculum.*` | Quantizr 5-stage staircase | PR95 8-stage curriculum; Muon + AdamW per-tensor selector; per-pair λ_R scheduler |
| posterior + observability infra (`tac.continual_learning` / `tac.council_continual_learning` / `tac.probe_outcomes_ledger` / `tac.deploy.modal.call_id_ledger` / `tac.frontier_scan`) | operator-side observability + decision posterior | wire-in to cathedral autopilot (5 of 15 orphans closed by subagent `a2ce5edd2d297d91f`; 10 remain) |

### §5.2 — Missing namespaces (build queue)

- `tac.compress_time_optimization.*` — generic TTO harness, multipass refinement loop, simulated-annealing on discrete codes, per-pair coordinate search, iterated bisection on rate-distortion knee (G compress-time)
- `tac.inflate_time_post_processing.*` — per-frame deterministic denoiser, wavelet post-filter, temporal mask smoothing, motion-only pose refinement; all scorer-free; all bake into inflate.py per Catalog #146 (G inflate-time)
- `tac.boosting.*` — residual cascade builder, per-pair decoder ensemble selector, mode-ensemble dispatch, Pareto-front-aware stack-growth (I)
- `tac.side_information.*` — RAFT optical-flow side-info baker, openpilot ego-motion feature extractor at compress, per-pair SegNet class summary baker, per-pair conditional-entropy estimator (J)
- `tac.search.*` — Bayesian / CMA-ES / `optuna` over the joint substrate-parameter manifold; consumed by the cathedral autopilot ranker (F)

### §5.3 — API design (decorator-based, composable)

Each namespace exposes a decorator + dataclass-backed contract mirroring the existing `@register_substrate(SubstrateContract(...))` pattern (Catalog #241). Taste anchor:

```python
# tac.compress_time_optimization
from tac.compress_time_optimization import (
    compress_time_pass, CompressTimePassContract, ComposableCompressPipeline,
)

@compress_time_pass(CompressTimePassContract(
    id="sensitivity_weighted_quant_refinement",
    stage_order=2,                  # runs after raw quant, before entropy coding
    consumes={"master_gradient", "archive_bytes_v0"},
    emits={"archive_bytes_v1"},
    sensitivity_weighted=True,      # auto-consumes tac.master_gradient anchor
    deterministic=True,             # required for byte-stable archives
    max_wallclock_seconds=None,     # compress-time is unbounded
))
def refine_quant_by_sensitivity(state, *, gradient, policy):
    """One pass of per-byte quant refinement weighted by ∂S/∂byte."""
    ...
    return state.replace(archive_bytes_v1=...)

# composition: pipelines stack with `|` and re-rank automatically via cathedral autopilot
pipeline = (
    ComposableCompressPipeline()
    | "raw_quant"                                # stage 1
    | "sensitivity_weighted_quant_refinement"    # stage 2 (above)
    | "iterated_bisection_rate_knee"             # stage 3
    | "magic_codec_per_tensor"                   # stage 4
    | "boosted_residual_cascade(depth=3)"        # stage 5
)
result = pipeline.run(seed_archive, master_gradient=mg_anchor)
```

The same shape applies to the other four namespaces — `@inflate_time_post_filter` / `@boost_stage` / `@side_info_baker` / `@search_strategy`. Each decorator stamps an immutable contract (frozen dataclass, type-hinted, schema-validated at import time per Catalog #168 AST discipline), wires automatically into the cathedral autopilot ranker (Catalog #125 6-hook contract), persists outcomes via fcntl-locked JSONL (Catalog #128/#131 sister discipline), and refuses ambiguous composition (e.g. two passes that emit the same key without explicit ordering raise at pipeline-build time, not at runtime).

### §5.4 — Composition primitives

- `|` (sequential) — chains stages: `A | B` runs A then B
- `&` (parallel-merge) — runs stages in parallel and merges by per-byte policy
- `@` (attach search) — attaches a search strategy that sweeps a stage's hyperparameters via `tac.search.*`

Each compose operation returns a new immutable pipeline (no mutation, no surprise side-effects, easy to test). Pipeline objects are JSON-serializable so the cathedral autopilot can rank candidate pipelines without instantiating them and the operator can audit each ranked candidate as plain text.

### §5.5 — Production-hardening contracts (enforced at decoration time)

- Frozen dataclass contracts, type-hinted throughout (mypy-strict / pyright-strict where applicable)
- Schema validation at import time per Catalog #168 (both `ast.Assign` and `ast.AnnAssign` handled)
- `deterministic=True` required for any stage that emits archive bytes (Catalog #158 deterministic-compiler discipline)
- fcntl-locked JSONL persistence for stage outcomes (Catalog #128 / #131 sister)
- `scorer_free=True` required for any stage that runs at inflate time (Catalog "strict scorer rule" non-negotiable)
- 6-hook wire-in declared at decoration (Catalog #125)
- Malformed stages fail at decoration time (import error), not at dispatch — composition errors surface in the IDE / preflight, not in paid GPU dispatch

## §6 — Build sequence priority

Order proposed for the next-session subagent wave (each item is one focused subagent slot):

1. `tac.master_gradient` consumers (5 H-row wrappers) — unlocks every sensitivity-weighting in A-G
2. `tac.compress_time_optimization` namespace (generic TTO + multipass + SA + per-pair coordinate search + iterated bisection)
3. `tac.boosting` namespace (residual cascade + per-pair decoder ensemble + Pareto-front-aware stack growth)
4. `tac.inflate_time_post_processing` namespace (per-frame denoiser + wavelet post-filter + temporal smoothing + motion-only pose refinement)
5. `tac.side_information` namespace (RAFT optical-flow baker + openpilot ego-motion + per-pair SegNet class summary + per-pair conditional-entropy estimator)
6. `tac.search` namespace (Bayesian / CMA-ES / `optuna` joint-manifold search)

Each namespace is one subagent + ~1-2 days of build. Once all five land, every future substrate (including the §11.5 class-shift candidates) inherits the toolbox via single-decorator opt-in.

## §7 — Cross-references

- Grand council T3 $50-budget symposium memo: `.omx/research/grand_council_t3_strategic_symposium_50_dollar_budget_20260517.md` (top 5 ranked for $40-90 envelope)
- Comprehensive codebase audit: `.omx/research/comprehensive_codebase_distillation_synthesis_20260517.md` (801 lanes; META-INSIGHT)
- META-ASSUMPTION ADVERSARIAL REVIEW anchor: `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` (the 18 shared assumptions producing the 0.196-0.199 plateau)
- DP1 dual stacking landing: `feedback_dp1_plus_fec6_dual_stacking_build_landed_20260517.md`
- Cathedral autopilot orphan-signal wire-in: `feedback_cathedral_autopilot_orphan_signal_wire_in_landed_20260517.md`
- §7.5 + §7.6 origin in PR writeup commit history; now collapsed to a 1-paragraph summary + pointer to this memo

## §8 — Reactivation criteria

Re-read this memo when:
- A new substrate is being designed (apply the §3 inventory to that substrate's archive grammar; cite which rows are addressed and which remain global)
- A new helper namespace is being scaffolded (use §5.3-5.5 contracts; cite this memo as the design provenance)
- The cathedral autopilot ranker needs new candidate sources (consult §5 namespace coverage map for what's wired vs missing)
- A council deliberation needs the canonical list of "what's leaving meat on the bone" (any substrate, any axis)

## §6 — Provider cost-driver breakdown (full table, dropped from PR writeup §11.5)

Preserved verbatim from the writeup pre-tightening pass. Subagents implementing the cathedral autopilot's `tac.cost_band_calibration` posterior + provider routing decisions should consult this table.

| category | cost basis | why |
|---|---|---|
| **Contest auth evaluation (the scorer itself)** | **essentially free** | One `inflate.sh` + `upstream/evaluate.py --device cpu` run is ~60-90 min on the M5 Max MacBook Pro at ~$0 marginal (electricity-only); on Modal Linux x86_64 CPU ~$0.06-0.12; on Modal T4 CUDA ~$0.30-1.50 wall-clock + dispatch overhead. The rigorous local-iteration loop: edit archive grammar → macOS-CPU eval (within `6e-6` of GHA Linux x86_64) → if promising, run paired `[contest-CPU GHA Linux x86_64]` + `[contest-CUDA T4]`. Verification round-trip `<$1` per candidate. |
| **End-to-end NEW substrate retraining** | **~$10-100 per substrate per attempt** | Training a Ballé-2018 joint codec, NeRV-family from-scratch retrain at contest resolution, or coordinate-MLP / SIREN / Cool-Chic substrate end-to-end costs ~10-40h on Modal T4 (~$0.59/hr) / Modal A10G (~$1.10/hr) / Vast.ai 4090 (~$0.25-0.40/hr spot) / Lightning A100 (subscription) / AWS g4dn.xlarge (~$0.22/hr spot). Trained substrate weights become reusable as anchors for cheap grammar/codec exploration per row 1. |
| **Complex curriculum / multi-stage training** | **2-5× retrain cost multiplier** | Substrates that need a staircase (Quantizr 5-stage canonical: anchor → finetune → joint → QAT → final), the PR95 8-stage curriculum, score-aware distillation, EMA shadow co-training, or LoRA TTO chains multiply the base retrain cost. PR95 Phase 2-4 alone runs ~$25-50 on Modal A100; the Quantizr staircase replication is ~$15-30 on Modal T4. |
| **Family sweeps for curve / optimum identification** | **$20-200 per substrate-family pareto map** | Identifying the rate-distortion Pareto frontier for a substrate family across `λ_R ∈ [0.001, 0.1]` and `sigma ∈ [0.5, 5.0]` costs N substrate-variant trainings (typically 8-16 grid points) at the per-retrain cost above. Identifying the optimum of a curve costs more than confirming a single point. |
| **Cross-paradigm composition validation** | **$30-150 per composition cell** | A stacking decision (e.g. NeRV mask codec × Ballé hyperprior × magic codec selector on PR101) requires joint training validation to confirm the composition is additive vs. saturating vs. antagonistic — at minimum one full retrain of the composed architecture. The substrate-composition matrix has thousands of theoretically-valid cells; we have empirically validated dozens. |
| **Hinton-distilled scorer surrogates** | **$10-40 per surrogate** | Training a CPU-trained Hinton-distilled SegNet/PoseNet surrogate (per `feedback_WW_cpu_trained_hinton_surrogate_bootstrap`) so a proxy-faithful loss is available for non-contest-resolution training. One-time cost per surrogate-version; amortized across many substrate retrains. |
| **Long-burn deep retrains** | **$50-200 per full-resolution job** | Some substrates (HiNeRV, FFNeRV, the more ambitious world-model architectures) want 24-100 epochs at full 1200-frame contest resolution. At those wall-clocks even Modal T4 / Vast.ai 4090 spot prices add up; the budget question becomes whether a 24h vs 100h run is the deciding factor for whether the substrate exits its plateau. |
| **Provider-rate variance + queue overhead** | **5-30% effective markup** | Modal T4 vs A10G vs A100 vs H100, Vast.ai 4090 spot vs dedicated, Lightning A100 subscription vs on-demand, AWS g4dn vs g5/g6 — rates vary 3-5×; queue wait + image-build overhead adds 5-30 min per dispatch (~$0.10-1.00 hidden cost). `tac.cost_band_calibration` posterior tracks per-provider effective rates from `.omx/state/modal_call_id_ledger.jsonl`. |
| **Dispatch-machinery infrastructure + replay buffers** | **<5% of total spend** | Provider tracker, custody validation, paired-axis re-evaluation, ledger maintenance, OSS release prep — each costs cents to dollars; cumulative across many sessions remains a rounding error vs. substrate-retrain bills. |


# OBSERVABILITY_SURFACE_SECTION_WAIVED:historical_design_memo_predates_catalog_305_section_header_requirement_or_is_namespace_design_not_substrate_specific_observability_per_catalog_110_113_HISTORICAL_PROVENANCE_APPEND_ONLY_discipline_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
