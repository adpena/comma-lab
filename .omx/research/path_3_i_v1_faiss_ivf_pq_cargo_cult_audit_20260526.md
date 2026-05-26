# SPDX-License-Identifier: MIT

---
schema: substrate_design_memo_v1
council_tier: T2
council_attendees: [Shannon, Dykstra, Jegou, Schmid, Atick, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent:
  - member: AssumptionAdversary
    verbatim: "Eight assumptions classified CARGO-CULTED-EMPIRICALLY-FALSIFIED-PARTIAL on the prior V1/V4/V8 work; the most important is that *the per-pair side-info channel reducer methodology* was the canonical-target rather than *per-pair residual quantization*. Path 3 I is a NEW substrate-design question (per-pair residual codec stacking on PR110) not a sub-extension of the side-info channel question."
council_assumption_adversary_verdict:
  - assumption: "V1 Faiss-IVF-PQ for per-pair side-info channel applies directly to per-pair residual codec stacking on PR110"
    classification: CARGO-CULTED
    rationale: "Side-info channel (V1-V8) compresses SegNet softmax outputs; residual codec stacks on PR110 fec6 reconstructs frame residuals (PR110_decoder_output - frame_gt). Different signal shape, different entropy structure, different stacking semantics. Per-pair side-info MI optimization does NOT translate to per-pair residual quantization."
  - assumption: "Faiss IVF-PQ is the right codec primitive for sub-2KB per-pair residual budget"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "V4 anchor empirically falsified <2KB budget at family level for side-info channel; residual codec has DIFFERENT byte budget envelope (PR110 fec6 is 178559 bytes total so per-pair residual overhead must be much smaller for net Δ<0)."
  - assumption: "The PQ codebook training set assumptions transfer cleanly between Faiss-IVF-PQ-for-side-info-channel and Faiss-IVF-PQ-for-residual-codec"
    classification: HARD-EARNED-CROSS-POLLINATION
    rationale: "PQ codebook training is signal-agnostic in the Jégou-Douze-Schmid 2011 sense; the codebook fits whatever K-means clusters exist in the residual distribution. The transfer is mathematically clean but operationally distinct."
council_decisions_recorded:
  - "op-routable #1: PHASE 2 path decision = Path (b) substrate-design REDIRECT — the per-pair residual codec is a NEW substrate question requiring V8-like learned-compression posture, NOT a sub-extension of V1-V7 side-info channel family"
  - "op-routable #2: distinguish from sister E=BoostNeRV via decomposition principle — Faiss-IVF-PQ residual codec uses PQ codebook decomposition (vector quantization); BoostNeRV uses iterative-boosting gradient-residual learning"
  - "op-routable #3: distinguish from sister G=NIRVANA via decomposition principle — NIRVANA uses hierarchical wavelet-pyramid scale-cascade; Faiss-IVF-PQ residual is flat per-pair PQ encoding"
  - "op-routable #4: NEW design memo MUST declare canonical-vs-unique decision per Catalog #290 for FAISS-PQ canonical helper adoption vs FORK because residual signal has DIFFERENT entropy structure"
  - "op-routable #5: NEW design memo MUST declare 3-axis evidence per directive #3 (math/sci/eng rigor + MLX drift minimization + numpy portability)"
deferred_substrate_retrospective_due_utc: 2026-06-25T07:52:00Z
deferred_substrate_id: v1_faiss_ivf_pq_residual_codec
related_deliberation_ids:
  - v1_faiss_v4_probe_plus_v8_design_landed_20260519
  - v1_faiss_v8_learned_compression_faiss_design_20260519
  - council_t3_cargo_cult_resurrection_v1_faiss_20260519
  - atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518
  - council_per_substrate_symposium_v1_dense_faiss_ivf_pq_reactivation_20260518
lane_id: lane_path_3_i_v1_faiss_ivf_pq_residual_cargo_cult_first_20260526
phase: phase_1_cargo_cult_audit
operator_directive_anchor: "Never simply extend unless a rigorous adversarial cargo cult pass has been done first" (2026-05-26)
horizon_class: frontier_pursuit
research_only: true
dispatch_enabled: false
---

# Path 3 I — V1 Faiss IVF-PQ residual codec PHASE 1 cargo-cult audit

> **Status**: PHASE 1 of 3-phase methodology per operator binding directive 2026-05-26 *"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"*. This memo enumerates V1+V4+V8 prior-work assumptions, classifies each HARD-EARNED-vs-CARGO-CULTED per the canonical addendum, and produces operator-routable PHASE 2 substrate-design decision input. Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" Catalog #325 + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" Catalog #220 non-negotiables.

## 0. Methodology

Per the 3-phase EXTENSION methodology binding for path #I:

- **PHASE 1 (THIS memo)**: cargo-cult pass on V1+V4+V8 priors
- **PHASE 2**: substrate-design decision per Catalog #290 (paths a/b/c)
- **PHASE 3**: L0 SCAFFOLD per chosen path

Per the Catalog #303 cargo-cult audit framework + the HARD-EARNED-vs-CARGO-CULTED addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`): every prior-work assumption is interrogated per the 4-category taxonomy with unwind path. **Crucially per CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW"** Catalog #291: the Assumption-Adversary MUST surface the BACKDROP shared by V1+V4+V8 (the assumption-framing all three operate within), not just the per-assumption questions.

## 1. Prior-work surface enumeration

| Document | Date | Surface | Verdict baseline |
|---|---|---|---|
| `atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md` | 2026-05-18 | V1/V2/V3 design memo (canonical Faiss-IVF-PQ for ATW V2-1 side-info channel) | Side-info channel reducer methodology |
| `council_per_substrate_symposium_v1_dense_faiss_ivf_pq_reactivation_20260518.md` | 2026-05-18 | V1 dense reactivation symposium | V1 dense FALSIFIED at 386× <2KB budget; reactivation criteria queued |
| `council_t3_cargo_cult_resurrection_v1_faiss_20260519.md` | 2026-05-19 | T3 cargo-cult resurrection symposium | V4 hand-rolled + V8 learned-compression op-routables surfaced |
| `v1_faiss_v4_probe_plus_v8_design_landed_20260519.md` | 2026-05-19 | V4 probe outcome (MI=2.37) + V8 design memo landing | per-pair class HISTOGRAM reducer EMPIRICALLY VALIDATED for side-info channel |
| `v1_faiss_v8_learned_compression_faiss_design_20260519.md` | 2026-05-19 | V8 learned-compression Faiss extension design memo (predicted band [0.187, 0.193]) | Predicted band CARGO-CULTED-PENDING-EMPIRICAL per Catalog #324 |
| `cargo_cult_resurrection_top_3_symposiums_v1_faiss_c6_ibps_v2_nscs06_v8_variant_c_20260518.md` | 2026-05-18 | Cross-pollination canonical | Discrete-posterior strategy bridging V8/C6/NSCS06 |
| `src/tac/substrates/v8_learned_compression_faiss/` (existing scaffold) | 2026-05-20 | L0 scaffold artifacts (architecture + archive + score_aware_loss + smoke) | Local-smoke deterministic; NO trained-score claim |
| `src/tac/optimization/faiss_ivf_pq_atw_channel.py` (28.6KB canonical helper) | 2026-05-18 | Canonical helper for build_pq_codebook + encode_per_region_histogram + serialize_codebook + estimate_pq_encoding_budget | Side-info channel canonical |

## 2. The META-ASSUMPTION (per Catalog #291 + #292)

**Per the Assumption-Adversary** seat (sextet-pact mandatory per "Council conduct" amendment Fix-7):

> The shared assumption framing ALL V1+V4+V8 prior work is operating within: **"V1 Faiss-IVF-PQ is a SIDE-INFO CHANNEL codec for ATW V2-1 cooperative-receiver loss substrate."**
>
> Per the operator's binding directive 2026-05-26 *"the MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it"*: the BACKDROP question of "what is the substrate I=V1 Faiss IVF-PQ residual codec is SERVING?" must be answered explicitly.
>
> The brief (`.omx/research/path_3_candidate_inventory_for_next_wave_spawning_20260526.md` §Tier 2 I) says: *"Faiss inverted-file + product-quantization (Jégou-Douze-Schmid 2011) for high-recall approximate nearest neighbor; applied to per-pair residual quantization for stacking on PR110 fec6 frontier."*
>
> **The mismatch**: V1-V8 prior work operates at the **side-info channel** for **ATW V2-1 cooperative-receiver loss substrate**. Path 3 I per the inventory operates at the **per-pair residual codec** for **PR110 fec6 frontier stacking**. **These are STRUCTURALLY DIFFERENT substrate surfaces.**
>
> Classification: **CARGO-CULTED-MISFRAMED-EXTENSION**. The "extension of V1 Faiss" framing implies inherited V1-V8 design priors apply directly. They do NOT — the surface, signal shape, byte budget, and stacking semantics are all different.

**This is the dominant cargo-cult to UNWIND first.** Treating path 3 I as a sub-extension of V1-V8 inherits MIS-FRAMED design priors. Per the operator's binding directive #1 (substrate-design-from-first-principles), the correct PHASE 2 decision is **REDIRECT** to a substrate-design-FRESH posture (path b) per Catalog #290, with V1-V8 prior work as **research INPUT only** (not as design extension target).

## 3. Per-assumption cargo-cult audit table

Per Catalog #303 + the HARD-EARNED-vs-CARGO-CULTED addendum + the 3-axis rigor amendment (math + scientific + engineering per layer). Each assumption gets unwind path + 3-axis evidence per directive #3.

### Inherited V1-V8 assumptions (interrogated for path-I applicability)

| # | Assumption | Classification | Math rigor | Scientific rigor | Engineering rigor | Unwind path |
|---|------------|----------------|-----------|------------------|-------------------|-------------|
| 1 | Faiss-IVF-PQ for the substrate at issue is a side-info channel codec for ATW V2-1 cooperative-receiver loss | CARGO-CULTED-MISFRAMED | n/a — assumption-framing not theorem | Side-info channel is Wyner-Ziv canonical; residual codec is rate-distortion canonical (DIFFERENT framework) | Reusing 28.6KB canonical helper `tac.optimization.faiss_ivf_pq_atw_channel` IS technically possible but its tests/observability all assume side-info channel | DECLARE path 3 I substrate is **per-pair residual codec for PR110 fec6 stacking** explicitly; FORK new canonical helper module `tac.optimization.faiss_ivf_pq_residual_codec` (NEW) or canonical-vs-unique decision (Catalog #290) |
| 2 | The <2KB byte budget per V1-V7 side-info channel applies to per-pair residual codec | CARGO-CULTED — EMPIRICALLY FALSIFIED at side-info family + DIFFERENT envelope for residual codec | PR110 fec6 = 178559 bytes total; per-pair residual overhead budget = ~(target_Δbytes/600 pairs) ≈ 5-30 bytes/pair for ΔS-positive | Need first-principles R(D) bound on per-pair residual entropy from PR110 reconstruction (Mallat wavelet-residual lens; Atick-Redlich retinal-residual lens) | V4 anchor showed Faiss-IVF-PQ at 5.4KB cannot meet <2KB; same constraint applies AND IS TIGHTER for residual codec | DECLARE per-pair residual budget envelope explicitly via Dykstra-feasibility per Catalog #296; ~10-30 byte/pair upper bound likely |
| 3 | k_topk=3 (per-pair class HISTOGRAM reducer) preserves MI for side-info channel applies to per-pair residual codec | CARGO-CULTED-MIS-TRANSLATED | per-pair class HISTOGRAM is a SOURCE-SIDE reducer for downstream conditioning; per-pair residual is OUTPUT-SIDE quantization of RGB error tensor | Different mathematical surfaces: histogram is information-theoretic; residual quantization is rate-distortion | Different signal: histogram operates on softmax bin probabilities; residual operates on pixel-level RGB error | NO direct translation; design memo MUST re-derive PQ codebook structure from PR110 residual statistics (NOT inherit V4 k_topk=3 anchor) |
| 4 | The V8 predicted band [0.187, 0.193] for side-info channel learned-compression applies to per-pair residual codec | CARGO-CULTED — DIFFERENT SURFACE | V8 band derived from Wyner-Ziv side-info R_WZ(D); residual codec needs MSE-distortion R(D) for output frames | V8 paper anchors Balle 2018 entropy-bottleneck for spatial latent; residual is fundamentally different signal shape | V8 archive grammar reserves slot for `learned_compression_encoder_weights_int8 + categorical_posterior_codeword_stream_brotli`; residual archive needs `per_pair_residual_pq_codebook + per_pair_pq_codeword_stream` | NEW design memo MUST derive predicted band per Catalog #324 post-training Tier-C density on the EMPIRICAL PR110 residual distribution; NO inheritance from V8 |
| 5 | V8 cross-pollination canonical with C6 IBPS Path B2 + NSCS06 v8 Variant C applies to per-pair residual codec | CARGO-CULTED-PARTIAL | V8/C6/NSCS06 all discrete-posterior at substrate-latent or chroma-residual surfaces; residual codec is a NEW surface | Cross-pollination canonical *concept* applies; specific surface analogy does NOT | Residual codec may cross-pollinate with E=BoostNeRV (iterative residual) + G=NIRVANA (hierarchical residual) DIFFERENTLY than with V8's substrate-latent cluster | PHASE 2 design memo MUST cite sister coordination from E + G + Path 3 brief explicitly, NOT inherit V8 cross-pollination map |
| 6 | The Faiss canonical 28.6KB helper `tac.optimization.faiss_ivf_pq_atw_channel` is directly reusable for residual codec | CARGO-CULTED-PARTIAL | Underlying Faiss IVF-PQ primitives (build_pq_codebook, serialize_codebook) ARE signal-agnostic | Helper *primitives* are canonical; helper *application* (encode_per_region_histogram) is side-info-specific | The 28.6KB helper provides primitives; residual codec needs NEW helper module wrapping Faiss for residual signal | Catalog #290 canonical-vs-unique decision per layer: ADOPT primitives (build_pq_codebook + serialize_codebook); FORK application layer (NEW `encode_per_pair_residual` / `decode_per_pair_residual` per residual signal shape) |
| 7 | The OMP_NUM_THREADS=1 workaround for Faiss on Apple Silicon is the right MLX-portability path | CARGO-CULTED-FOR-AXIS-3 | Faiss is C++/OMP; MLX is Apple Silicon Metal — fundamentally different execution model | Faiss-cpu PyPI wheel uses libomp; MLX Metal uses GPU shaders | OMP_NUM_THREADS=1 is a workaround for CPU thread contention; MLX-Faiss adapter requires either (a) MLX-native PQ encoder OR (b) numpy reference path for portability | Per directive #3 axis 3 numpy portability: NEW design memo MUST decide MLX-native vs numpy-reference for PQ primitives; numpy-reference is portable + MLX-Faiss adapter is OPTIONAL accelerator |
| 8 | The MLX-first gate at threshold 0.001 contest-units (Catalog #1265) applies cleanly to Faiss-based codec | CARGO-CULTED-PENDING-EMPIRICAL | Threshold 0.001 = 90× margin over empirical anchor 0.000011 for MLX↔PyTorch parity | Faiss PQ encoder is quantization-step-deterministic if seed + codebook pinned; MLX↔Faiss-CPU parity should be tighter than 0.001 | Need empirical MLX↔numpy↔PyTorch parity on the actual residual codec primitives | NEW design memo MUST declare MLX↔numpy↔PyTorch parity test plan per directive #3 axis 2 + 3 |

### New path-I-specific assumptions (NEVER tested in V1-V8 prior work)

| # | Assumption | Classification | Math rigor | Scientific rigor | Engineering rigor | Unwind path |
|---|------------|----------------|-----------|------------------|-------------------|-------------|
| 9 | Per-pair RGB residual (PR110_decoded - frame_gt) has enough structure for PQ codebook to achieve meaningful distortion-saving at <30 byte/pair budget | CARGO-CULTED-PENDING-EMPIRICAL | Per-pair residual entropy ≈ 1-3 bits/pixel typical natural-image residual statistics (Mallat wavelet-residual lens) | Per Daubechies wavelet theory: residuals at successive decoder levels have sparse structure; PQ on wavelet-domain residuals more efficient than direct-pixel | Need MLX-local probe: extract PR110 fec6 residuals on 600 contest pairs; fit PQ codebook (M=4, ksub=256); measure per-pair distortion at fixed byte budget | PHASE 3 L0 SCAFFOLD with numpy reference + MLX-Faiss adapter probe |
| 10 | Single per-pair PQ codebook (shared across all 600 pairs) is the right granularity vs per-class-conditioned codebook (per SegNet class) | CARGO-CULTED-PENDING-EMPIRICAL | Per-class-conditioned: more codebooks, smaller per-codebook entropy, larger total codebook bytes | per-class-conditioned codebook is Atick-Redlich cooperative-receiver canonical (decoder has scorer side-info per Wyner-Ziv R_WZ(D)) | Need probe with both: shared-codebook vs per-class-conditioned | NEW design memo declares both as variants; smoke-fork in PHASE 3 sister probes |
| 11 | Residual codec stacks ADDITIVELY on PR110 fec6 frontier (orthogonal axes per Stack-of-Stacks discipline) | CARGO-CULTED-PENDING-EMPIRICAL | Residual codec quantizes (PR110_decoded - frame_gt) so adds back to PR110_decoded at inflate; stacking is additive by construction | Per CLAUDE.md "Stack-of-Stacks discipline" Catalog #319 sister deliverability proof | Need empirical composition smoke after PR110 fec6 + I=Faiss residual codec individually verified | PHASE 5 composition smoke conditional on PHASE 3 + PHASE 4 individual anchors |
| 12 | The MLX-first gate at threshold 0.001 contest-units is achievable for residual codec primitives | CARGO-CULTED-PENDING-EMPIRICAL | Codebook quantization is deterministic; PQ encoding is deterministic; bilinear interpolation between residual decode and pixel addition is MLX-drift-sensitive per Catalog #1255 + sister DreamerV3 max_abs=24.34 anchor | Per directive #3 axis 2: known-drift patterns AVOID; per-axis tolerance characterize | Need MLX↔PyTorch parity test on the residual decode + addition path; FOLLOW canonical `bilinear_resize2x_align_corners_false_nhwc` per sister A=DreamerV3 forensic | PHASE 3 L0 SCAFFOLD includes MLX↔PyTorch parity test per axis 2 |

## 4. The KEY MISFRAMING (Assumption #1 deep dive)

The brief calls path 3 I *"V1 Faiss IVF-PQ residual codec MLX-local L0 SCAFFOLD via cargo-cult-pass-first methodology"* and *"V4 hand-rolled probe done + V8 learned-compression Faiss design memo exists"*. The V4 + V8 priors are PRESERVED AS INPUT RESEARCH not bolt-on target.

**The structural correction**: V1-V8 priors are the canonical FAISS-IVF-PQ FOR ATW V2-1 SIDE-INFO CHANNEL design history. Path 3 I per the inventory brief targets *per-pair residual quantization for stacking on PR110 fec6 frontier* — a substrate-DIFFERENT application of the SAME underlying Jégou-Douze-Schmid 2011 PQ primitives.

The cargo-cult to unwind: **inheriting V8's predicted band, cross-pollination canonical, archive grammar, and council attendee roster as if they applied to the residual codec surface.** They do NOT. The shared element is the underlying Faiss primitives (build_pq_codebook, serialize_codebook). The non-shared elements are signal shape (RGB residual vs SegNet softmax bins), byte budget (~10-30 byte/pair vs ~3-5 byte/pair for side-info), entropy structure (natural-image residual vs categorical posterior), and stacking semantics (additive frame correction vs side-info channel conditioning).

**Per the operator's substrate-design-from-first-principles directive #1**: Path 3 I MUST design the substrate FRESH (per Catalog #290 path b: substrate-design REDIRECT) with V1-V8 prior work as input research not extension target.

## 5. Distinguishing from sister candidates

Per the inventory brief sister citation + the 3-phase cargo-cult-first methodology binding:

| Sister | Decomposition principle | Surface | Path 3 I distinction |
|---|---|---|---|
| **E=BoostNeRV PR110 residual** (in-flight `a35f9f86781aaaa4f`) | Iterative boosting (gradient-residual learning across iterations) | per-pair RGB residual against PR110 fec6 frontier | I=Faiss-IVF-PQ uses **vector quantization** (PQ codebook + index lookup) NOT gradient-residual iterative learning |
| **G=NIRVANA cascading NeRV** (in-flight `ae952528954e27bef`) | Hierarchical wavelet-pyramid (multi-scale residual cascade) | per-pair RGB residual cascade across levels (48×64 → 96×128 → 192×256 → 384×512) | I=Faiss-IVF-PQ is **FLAT per-pair PQ** (single-level codebook quantization) NOT hierarchical scale cascade |
| **A=DreamerV3 RSSM** (landed `69253a1cc`) | Categorical latent dynamics (G×K group-categorical alphabet) | per-pair latent representation | I=Faiss-IVF-PQ targets RGB residual codec NOT latent dynamics |
| **F=Z8 hierarchical predictive coding** (in-flight) | Canonical quadruple (Rao-Ballard + Mallat + Hafner + Wyner-Ziv) | per-pair predictive-coding bottleneck | I=Faiss-IVF-PQ is paradigm-orthogonal codec primitive NOT predictive-coding paradigm |
| **K=COIN++ INR** (in-flight `a7977f23a7f0f0573`) | Meta-learned MLP per coordinate | per-pair coordinate-MLP encoding | I=Faiss-IVF-PQ is per-pair residual quantization NOT meta-learned implicit representation |

**Path 3 I unique decomposition principle**: per-pair RGB residual against PR110 fec6 frontier, quantized via Jégou-Douze-Schmid 2011 Product Quantization (NOT iterative boosting; NOT hierarchical cascade; NOT meta-learning; NOT predictive coding). The codec primitive is OPQ-aware PQ + IVF (inverted-file index for codebook nearest-neighbor lookup) per the canonical 2011 paper.

## 6. MLX-Faiss adapter feasibility (axis 3 portability)

Per directive #3 axis 3 (numpy portability):

- **Faiss is C++ with libomp + Apple Silicon arm64 bindings via faiss-cpu PyPI wheel.** V4 hand-rolled probe at `tools/probe_atw_v2_1_faiss_pq_v4_hand_rolled.py` ran successfully on M5 Max with `KMP_DUPLICATE_LIB_OK=TRUE + OMP_NUM_THREADS=1` workaround.
- **MLX is Apple Silicon Metal-native; no direct Faiss binding exists.** Path 3 I cannot use MLX-native Faiss directly.
- **The portability path**: split the codec into (a) PQ codebook training (numpy reference using sklearn KMeans OR Faiss CPU as accelerator) + (b) PQ encoding (numpy reference + MLX-native vectorized index lookup) + (c) PQ decoding (numpy reference + MLX-native vectorized gather).

**Per directive #3 axis 3**: every primitive should have sister numpy reference implementation. This unlocks (a) GHA CPU CI testing without Faiss install, (b) cross-validation between MLX-native and Faiss-CPU paths, (c) operator-portable diagnostic on non-Apple-Silicon hardware.

**Per directive #3 axis 2 (MLX drift minimization)**: PQ encoding is integer-deterministic (codebook lookup is exact integer arithmetic); PQ decoding is float-deterministic (codebook gather is exact float copy). MLX drift surface limited to (i) bilinear interpolation if residual is upsampled to native resolution before addition (USE canonical `bilinear_resize2x_align_corners_false_nhwc` per sister A forensic), (ii) uint8 cast at final output (USE canonical Catalog #205 sister rounding).

## 7. PHASE 2 input — substrate-design decision

Per Catalog #290 paths (a/b/c):

- **Path (a) DIRECT EXTENSION**: inherit V1-V8 side-info channel design priors directly. **REJECT**: misframes substrate per Assumption #1.
- **Path (b) SUBSTRATE-DESIGN REDIRECT**: declare path 3 I as NEW substrate-design question for *per-pair RGB residual codec stacking on PR110 fec6 frontier*; V1-V8 prior work is research INPUT only; canonical 28.6KB helper primitives reusable for build_pq_codebook + serialize_codebook; NEW canonical helper module FORKED for `encode_per_pair_residual` + `decode_per_pair_residual`. **RECOMMENDED PATH** per operator's substrate-design-from-first-principles directive #1 + Assumption-Adversary verdict.
- **Path (c) HYBRID — V8 SIDE-INFO + RESIDUAL CODEC**: implement BOTH V8 side-info channel learned-compression + Faiss-IVF-PQ residual codec as composable axes per Stack-of-Stacks. **DEFER**: V8 is currently sister scaffold (`src/tac/substrates/v8_learned_compression_faiss/`) not landed full-substrate; composing two unverified substrates is the kitchen_sink anti-pattern.

**PHASE 2 verdict**: Path (b) SUBSTRATE-DESIGN REDIRECT.

## 8. 6-hook wire-in declaration per Catalog #125

| # | Hook | Status | Rationale |
|---|------|--------|-----------|
| 1 | Sensitivity-map contribution (`tac.sensitivity_map.*`) | N/A — cargo-cult audit only | Sensitivity-map contribution lands at PHASE 3 L0 SCAFFOLD via canonical `tac.sensitivity_map.per_byte_leverage` |
| 2 | Pareto constraint (`tac.pareto_*`) | N/A — cargo-cult audit only | Pareto constraint lands at PHASE 3 via canonical `tac.pareto_solver.add_constraint` |
| 3 | Bit-allocator hook | N/A — cargo-cult audit only | Bit-allocator hook lands at PHASE 3 byte budget allocation |
| 4 | Cathedral autopilot dispatch hook | **ACTIVE (DESIGN-LEVEL)** | This cargo-cult audit memo IS consumed by cathedral autopilot ranker via `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #344; PHASE 2 substrate-design decision IS Tier A observability-only routing recommendation per Catalog #341 (NOT score-mutating) |
| 5 | Continual-learning posterior update | **ACTIVE (DESIGN-LEVEL)** | This cargo-cult audit memo IS continual-learning artifact per CLAUDE.md "Subagent coherence-by-default" non-negotiable; canonical posterior anchor via Catalog #313 probe-outcomes ledger NOT YET appended (PHASE 3 outcome registration) |
| 6 | Probe-disambiguator | **ACTIVE (DESIGN-LEVEL)** | This memo IS canonical disambiguator between V1-V8 side-info channel application vs per-pair residual codec application of Faiss-IVF-PQ primitives; canonical disambiguator question: "is V1 Faiss IVF-PQ research lineage applicable to PR110 fec6 residual codec without re-derivation?" — answered EMPIRICALLY-FALSIFIED-PARTIAL via per-assumption audit table |

## 9. 3-axis rigor evidence per directive #3

### Axis 1: Math + scientific + engineering rigor per layer

| Layer | Math | Scientific | Engineering | Verdict |
|---|---|---|---|---|
| Cargo-cult unwind methodology | Shannon information-theory + Wyner-Ziv R_WZ(D) | `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` | Catalog #303 cargo-cult audit + Catalog #292 per-deliberation assumption surfacing + Catalog #325 per-substrate symposium | HARD-EARNED |
| Sister substrate distinguishing | Decomposition principle taxonomy (vector quantization vs iterative boosting vs hierarchical cascade vs meta-learning vs predictive coding) | Jégou-Douze-Schmid 2011 PQ paper + Friedman boosting + Mallat wavelet theory + Dupont COIN++ 2021 + Rao-Ballard 1999 | sister coordination citation per Catalog #230 ownership map | HARD-EARNED |
| Predicted band derivation | Per-pair residual R(D) bound + Catalog #324 post-training Tier-C density validation discipline | per Catalog #324 anchor: predicted_band derived BEFORE training IS structurally invalid; post-training Tier-C re-measurement REQUIRED | NO predicted band claimed in this PHASE 1 memo (defer to PHASE 2 design memo with Dykstra-feasibility per Catalog #296) | HARD-EARNED (no premature claim) |
| MLX-Faiss adapter feasibility | Integer codebook + float gather both deterministic | Faiss-cpu PyPI wheel + libomp Apple Silicon arm64 + MLX Metal Apple Silicon | V4 hand-rolled probe empirically demonstrated faiss-cpu workable on M5 Max with OMP_NUM_THREADS=1 workaround | HARD-EARNED |

### Axis 2: MLX drift minimization per primitive

Primitives planned for PHASE 3 L0 SCAFFOLD:
- **PQ codebook gather (MLX)**: float-deterministic; sister K=COIN++ + sister G=NIRVANA both use MLX `mx.take_along_axis` canonical; expected drift ≤ epsilon-machine
- **Bilinear residual upsample (MLX) if needed**: USE canonical `bilinear_resize2x_align_corners_false_nhwc` per sister A=DreamerV3 forensic (max_abs=24.34 caused by `align_corners=True` anti-pattern); expected drift ≤ Catalog #1265 threshold 0.001 with canonical helper
- **uint8 cast at output (MLX)**: USE canonical Catalog #205 sister rounding; expected drift = 0 (deterministic)
- **PQ encoding (numpy reference path)**: deterministic integer arithmetic; expected MLX↔numpy parity = byte-identical

### Axis 3: Portability via numpy per primitive

Every MLX primitive has sister numpy reference implementation:
- **PQ codebook gather**: numpy `np.take_along_axis` reference path
- **Bilinear upsample**: numpy reference via canonical helper (sister NIRVANA `numpy_reference.py` pattern)
- **uint8 cast**: numpy `np.clip(0, 255).astype(np.uint8)` reference path
- **PQ codebook training**: sklearn KMeans reference path (numpy-pure); Faiss-CPU is optional accelerator
- **PQ encoding**: numpy reference path; MLX-Faiss adapter is optional accelerator

Per directive #3 axis 3: substrate operable on CPU-only test rigs without MLX OR Faiss dependency.

## 10. Discipline compliance summary

- **Catalog #229** (premise verification before edit): COMPLETE — read V1+V4+V8 priors + sister scaffold pattern + canonical helpers BEFORE writing
- **Catalog #303** (cargo-cult audit per assumption): COMPLETE — 12 assumptions enumerated with classification + unwind path
- **Catalog #292** (per-deliberation assumption surfacing): Assumption-Adversary verdict in YAML frontmatter
- **Catalog #287** (placeholder-rationale rejection): all classifications carry non-placeholder rationale ≥4 chars
- **Catalog #290** (canonical-vs-unique decision per layer): PHASE 2 design memo will land canonical-vs-unique table; this PHASE 1 memo declares the META-ASSUMPTION misframing
- **Catalog #309** (horizon_class declaration): `horizon_class: frontier_pursuit` in YAML frontmatter
- **Catalog #294** (9-dim checklist evidence): PHASE 2 design memo will land 9-dim table; PHASE 1 declares METHOD
- **Catalog #296** (Dykstra-feasibility predicted-band check): PHASE 2 design memo will declare predicted band + Dykstra-feasibility intersection
- **Catalog #305** (observability surface): PHASE 2 design memo will declare 6-facet observability surface
- **Catalog #300** (council deliberation v2 frontmatter): COMPLETE — YAML frontmatter carries council_tier T2 + attendees + assumption_adversary_verdict + mission_alignment fields
- **Catalog #325** (per-substrate symposium discipline): PHASE 3 L0 SCAFFOLD landing memo will queue Catalog #325 symposium for dispatch eligibility
- **Catalog #117/#157/#174** (canonical commit serializer): committed via canonical serializer with POST-EDIT `--expected-content-sha256`
- **Catalog #119** (Co-Authored-By trailer): commit message carries canonical Claude trailer
- **Catalog #206** (subagent checkpoint discipline): 2 checkpoints emitted (session-start + post-PV)
- **Catalog #110/#113** (HISTORICAL_PROVENANCE APPEND-ONLY): NEW memo only; no mutation of V1+V4+V8 prior memos
- **Catalog #230** (sister-subagent ownership map): commit body cites E + G + A + F + K + H sister coordination
- **Catalog #287** (placeholder-rationale rejection): NO `<rationale>` / `<reason>` placeholders
- **Catalog #208** (docs/local-paths): NO `/Users/` absolute paths

## 11. Cross-references

- Parent V1 Faiss prior memos: `v1_faiss_v4_probe_plus_v8_design_landed_20260519.md` + `v1_faiss_v8_learned_compression_faiss_design_20260519.md` + `atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md`
- Sister Path 3 wave: A=`aaec7a0d220f31543` DreamerV3 RSSM landed + E=`a35f9f86781aaaa4f` BoostNeRV in-flight + G=`ae952528954e27bef` NIRVANA landed
- Path 3 inventory brief: `.omx/research/path_3_candidate_inventory_for_next_wave_spawning_20260526.md`
- Canonical helper to reuse (primitives): `src/tac/optimization/faiss_ivf_pq_atw_channel.py` (build_pq_codebook + serialize_codebook reusable; application layer FORKED for residual codec)
- Catalog #303 cargo-cult audit framework + addendum: `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`
- Sister NIRVANA scaffold pattern: `src/tac/substrates/nirvana_cascading_nerv/` (mlx_renderer + numpy_reference + archive + inflate + tests)

## 12. Next action

PHASE 2 substrate-design decision memo at `.omx/research/path_3_i_v1_faiss_ivf_pq_substrate_design_decision_20260526.md` declaring Path (b) SUBSTRATE-DESIGN REDIRECT with V1-V8 prior work as research INPUT only.
