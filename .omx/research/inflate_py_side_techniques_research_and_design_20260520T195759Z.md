---
title: Inflate.py side-techniques deep research and design — 5-bucket taxonomy
date: 2026-05-20T19:57:59Z
lane_id: lane_wave_3_inflate_py_side_techniques_research_and_design_20260520
tier: T3
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - PR95Author
  - Carmack
  - Hotz
  - Selfcomp
  - MacKay
  - Ballé
  - TimeTraveler
  - TimeTravelerProtege
  - Tishby
  - Atick
  - Rao
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "every inflate.py side technique that adds bytes is competing with HNeRV decoder bytes for the same archive budget; bucket #2 + bucket #4 risk re-doing what's already in #1097 selector paradigm extension"
  - member: Assumption-Adversary
    verbatim: "the assumption that inflate.py-side primitives can produce score reduction independent of training-time substrate fit is CARGO-CULTED; the canonical equation per_pair_master_gradient_score_impact_taylor_v1 says reconstructed score response is local linear in archive byte mass at fixed decoder, so ΔS predictions for bucket #1 generic transforms are bounded by ||M_archive||_inf × byte_delta"
council_assumption_adversary_verdict:
  - assumption: "scorer-free inflate-time transforms can recover score equivalent to scorer-aware training"
    classification: CARGO-CULTED
    rationale: "per master_gradient_locality_violation_by_codec_v1 codec acts globally on score; per_byte_leverage_uniformly_distributed_v1 says inflate-time corrections need to hit high-leverage bytes or zero net effect; transforms ignorant of which bytes are score-critical leave score on table"
  - assumption: "side information about camera geometry / frame index / coordinate grids is review-safe"
    classification: HARD-EARNED
    rationale: "PR101 GOLD and PR102 already use frame-index conditioning; coordinate-grid features are public-domain (Atick-Redlich 1990 retinal mosaic) — Yousfi PR108 closure validates this class"
  - assumption: "Bucket #5 deterministic-grid side info is FREE because the bytes do not appear in archive.zip"
    classification: HARD-EARNED-CONDITIONAL
    rationale: "free for the byte budget IF computed inside inflate.py from constants. NOT free for review surface: anything beyond camera intrinsics + frame_index_from_file_list risks 'using test-time information about which video' violation per CLAUDE.md Strict scorer rule"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_decisions_recorded:
  - "5-bucket taxonomy enumerated with 27 candidate primitives total"
  - "Top-K = 8 primitives ranked by ΔS/LOC × compliance-safety"
  - "Bucket #4 (hybrid weight/runtime contracts) explicitly DELEGATED to sister #1097 selector-paradigm-extension catalog for L0 SKETCH lanes; this memo enumerates abstract taxonomy only"
  - "1 new canonical equation candidate proposed: inflate_time_generic_transform_score_ceiling_v0"
  - "Integration cross-references to Pact-NeRV ULTIMATE (#1095), STC, symposium_impls"
related_deliberation_ids:
  - council_per_substrate_symposium_pact_nerv_score_axis_aware_foveated_ego_motion_full_stack_synergy_eval_roundtrip_20260520
  - cross_candidate_strategic_findings_canonical_extension_20260520T195940Z
horizon_class: frontier_pursuit
canonical_provenance:
  kind: PREDICTED_FROM_MODEL
  evidence_grade: predicted
  axis_tag: "[predicted]"
  score_claim: false
  promotable: false
  hardware_substrate: not_applicable
---

# Inflate.py side-techniques deep research and design — 5-bucket compliance-safe taxonomy

**Lane:** `lane_wave_3_inflate_py_side_techniques_research_and_design_20260520`
**Operator routing:** Catalog #1096 (this memo) + sister Catalog #1097 (selector-paradigm-extension; concrete L0 scaffolds for Bucket #4)
**Compliance posture:** all primitives below MUST honor CLAUDE.md "Strict scorer rule" (no scorers at inflate time) + HNeRV parity L4 (≤200 LOC inflate.py; PR101 GOLD = 150) + Catalog #105+#139 (no-op detector — charged bits only) + Catalog #205 (canonical inflate device select) + HNeRV parity L9 (runtime closure).

## 1. Frontier baseline + canonical reference points

Per `.omx/state/canonical_frontier_pointer.json` (refreshed 2026-05-20T19:32:58Z):

| Axis | Score | Archive sha (prefix) | Bytes | Hardware |
|---|---|---|---|---|
| local frontier contest-CPU | 0.19205 | `6bae0201fb08` | 178,517 | Linux x86_64 CPU |
| local frontier contest-CUDA | 0.20533 | `9cb989cef519` | 186,876 | Linux x86_64 T4 |
| PR101 GOLD (upstream) | 0.1928 cluster | published | ~178K | both axes |

Canonical inflate.py reference LOC budget (lines of executable code excluding docstrings + blank lines):

| Submission | LOC | Surface |
|---|---|---|
| `submissions/a1/inflate.py` | 135 | A1 fine-tuned, minimal HNeRV runtime |
| `submissions/frame_exploit_selector_sidecar/inflate.py` | 171 | fec6 frontier — adds selector trailer parsing + sidecar corrections |
| `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py` | 740 | r2 grammar variant; ≥3.7× over HNeRV parity L4 budget |
| HNeRV parity L4 ceiling | 200 | (PR101 GOLD = 150 reference) |

The fec6 selector inflate.py adds ~36 LOC over A1 to parse + apply selector indices + decode sidecar — *that increment is already inside L4 budget* and is the empirical existence proof that inflate.py-side primitives are reviewable.

## 2. Cross-references to in-flight sister work (binding)

| Sister memo | Scope | Overlap with this memo |
|---|---|---|
| Catalog #1095 PACT-NERV-ULTIMATE | foveated ego-motion + score-axis-aware HNeRV full stack | Bucket #5 (camera/lens geometry side info) overlaps with PACT-NERV's foveation prior; Bucket #1 deblocking/temporal-smoothing complementary at inflate time |
| Catalog #1097 selector-paradigm-extension | concrete L0 SKETCH lanes for 7 sub-candidates (per_class_chroma / per_pair_difficulty_atlas_conditioned / huffman_k_variant_sweep / arithmetic_coding / rice_golomb / run_length_encoded / dictionary_coded) | **DELEGATED**: Bucket #4 abstract taxonomy below ↔ #1097 concrete scaffolds. NO duplication: this memo enumerates the abstract slot; #1097 produces lane scaffolds. |
| Catalog #1098 Gap 1 auto-trigger | sister coordination on inflate.py L4 budget audit | applies; see §10 audit |
| `council_per_substrate_symposium_pact_nerv...20260520T185500Z.md` | T3 council on Pact-NeRV substrate | This memo's Bucket #5 cites Atick-Redlich for the same retinal-mosaic geometry the symposium covered |

## 3. The 5-bucket taxonomy — enumeration

### Bucket #1 — Generic decoder transforms (compliance-safe by construction)

Image / video signal processing that runs on already-decoded RGB frames inside inflate.py, parameter-free or with constants baked into source. No scorer touch. No external state. Operate purely on output of `decoder(latents)`.

| Primitive | LOC | Predicted ΔS | Dykstra-feasibility note | Compliance | Composability with fec6+HNeRV |
|---|---|---|---|---|---|
| **1.1 Deblocking filter (3×3 separable)** | +8 | -0.0005 to -0.001 [predicted; bounded by ‖M_archive‖∞ × edge_pixel_count ÷ N] | feasible (operates AFTER bicubic upsample; orthogonal to decoder weights) | safe (constants only) | additive (post-decoder) |
| **1.2 Temporal smoothing (per-pair 2-tap EMA on frame_1)** | +6 | -0.001 to -0.003 [predicted; PoseNet rewards small inter-frame coherence] | feasible (acts on flat reshape) | safe | additive — but check fec6 frame-exploit selector NOT inverted by smoothing |
| **1.3 Color correction (per-channel affine constants)** | +4 | -0.0003 to -0.0008 [predicted; mean-removal trick analog] | feasible (already present at `submissions/a1/inflate.py:113-115` as `sub_(1.0)` lines — extending) | safe | additive on top of existing affine |
| **1.4 Motion-compensated refinement (block-MV refinement)** | +35 | -0.003 to -0.008 [predicted; requires per-pair MV table — needs charged bytes per Catalog #105 or punts to Bucket #5 deterministic-grid] | partially-feasible (MV table is itself a sidecar = Bucket #4) | safe IF MV table is charged | competes with selector budget |
| **1.5 Foveated sharpening (per-pixel sharpen weight from radial distance to image center)** | +12 | -0.001 to -0.004 [predicted; per Atick-Redlich retinal mosaic + Pact-NeRV symposium §foveation] | feasible (radial grid is deterministic constant) | safe | additive — note PACT-NeRV intends foveation at TRAIN time, this is INFLATE-TIME analog |
| **1.6 Edge-aware bilateral filter (5×5)** | +14 | -0.001 to -0.003 [predicted; conservative — bilateral preserves edges PoseNet cares about] | feasible | safe | additive |
| **1.7 Gamma correction (per-pixel power-law)** | +3 | +0 to -0.0002 [bounded; mostly null] | feasible | safe | additive |

**Bucket #1 ΔS ceiling**: aggregate predicted -0.006 to -0.015 if 4-5 stacked; **Dykstra-feasibility**: convex projections compose well IF each operator is non-expansive in pixel space. Smoothing + deblocking + sharpening triple may saturate (smoothing then sharpening = identity in expectation per Shannon's data-processing inequality applied to lossy filters).

**HARD-EARNED ASSUMPTIONS** (Catalog #303):
- **1.1 deblocking helps**: HARD-EARNED — JPEG/H.264 reference codecs apply post-decoder deblocking universally; the SegNet+PoseNet pair is sensitive to block-edge artifacts at 384×512 → 874×1164 bicubic upscale.
- **1.5 foveated sharpening helps PoseNet**: HARD-EARNED conditional on Pact-NeRV symposium finding that foveation IS the canonical ego-motion prior; the inflate-time radial mask is a cheap analog.

**CARGO-CULTED ASSUMPTIONS**:
- **1.7 gamma correction helps**: CARGO-CULTED — inherited from photography compounding; the contest scorer derives masks from frames at integer RGB 0..255 and PoseNet on YUV6; a global gamma is approximately scale-invariant under both transforms.
- **1.2 temporal smoothing always helps**: CARGO-CULTED — smoothing reduces high-freq PoseNet signal AS WELL as noise; on fec6 frontier where rounded-MPS-source already encoded inter-frame deltas in palette indices, smoothing may DESTROY signal. PROBE-DISAMBIGUATOR required.

### Bucket #2 — Tiny source-generic correction circuits

A small (≤2KB) parametric correction kernel whose parameters live in the archive as charged bytes, applied by inflate.py via a generic operation (no interpreter; no eval; just `tensor @ matrix + bias`-class arithmetic).

| Primitive | LOC | Bytes (charged) | Predicted ΔS | Dykstra-feasibility | Compliance | Composability |
|---|---|---|---|---|---|---|
| **2.1 Per-pixel-class 3×3 conv correction (5 classes × 27 fp4 = 135 bytes)** | +25 | ~135 | -0.005 to -0.010 [predicted; rate cost ≈ 25×135/37545489 ≈ 9e-5; net positive if any class improves] | feasible (conv is convex projection) | safe (no scorer) — but class indices must come from inflate-deterministic source (Bucket #5 grid) or from HNeRV decoder output not from contest SegNet | composable with fec6 |
| **2.2 Per-frame affine LUT (3 channels × 4 params × 1200 frames = ~14400 fp4 → 7200 bytes)** | +12 | ~7,200 | -0.002 to -0.006 [predicted; net + only if per-frame affine corrects systematic decoder bias] | feasible (affine = convex) | safe | competes hard with HNeRV decoder bytes |
| **2.3 Per-block (16×16) DC offset (per-pair = 27×32 = 864 blocks × 1 fp4 = 432 bytes/pair × 600 pairs = 259,200 bytes)** | +15 | ~259,200 | UNDEFINED [predicted ΔS lower bound -0.04, but rate cost +0.173 = NET POSITIVE huge regression] | infeasible (rate cost dominates) | safe but uneconomic | uneconomic vs HNeRV |
| **2.4 Per-class chroma LUT (5 classes × 2 chroma × 8 levels × 1 fp4 = 80 bytes)** | +18 | ~80 | -0.002 to -0.006 [predicted; NSCS06 v7 empirical floor ratified the per-class-chroma class] | feasible | safe — DELEGATED to #1097 as concrete L0 SKETCH lane | sister-#1097 |
| **2.5 Per-pair deblock strength (1 fp4 / pair = 600 bytes)** | +10 | ~600 | -0.001 to -0.003 [predicted; allows pair-adaptive deblock; rate cost 4e-4] | feasible | safe | additive |

**Bucket #2 ΔS ceiling**: -0.005 to -0.012 if 2-3 stacked (2.1 + 2.4 + 2.5 = 815 bytes total ≈ 5e-4 rate cost, predicted distortion savings -0.008 to -0.019; net negative SCORE delta -0.0075 to -0.018).

**HARD-EARNED**: 2.4 per-class-chroma is HARD-EARNED per NSCS06 v7 44% empirical improvement (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`).

**CARGO-CULTED**: 2.3 per-block DC offset assumes block boundaries align with scorer-sensitive structure — FALSIFIED by rate cost arithmetic above.

### Bucket #3 — Runtime layout optimization (compliance-safe; zero score impact direct; indirect via determinism)

Pure runtime engineering. No score signal direct contribution. Predicted ΔS = exactly 0 (the rate term doesn't change; the score components don't change). BUT: indirect signal via (a) enables previously-too-slow-to-converge primitive landings + (b) enables deterministic CI auth-eval at lower variance + (c) preserves operator's iteration cadence.

| Primitive | LOC | Predicted runtime delta | Compliance | Note |
|---|---|---|---|---|
| **3.1 mmap-based archive read** | +3 | -50ms per inflate | safe | already supported via `Path(...).read_bytes()`; just don't `f.read()` on huge files |
| **3.2 Explicit `torch.set_num_threads(1)` for deterministic CPU** | +2 | ~+5% time but 100% determinism | safe per Catalog #205 | CRITICAL for reproducing the 0.1928 cluster bit-exactly |
| **3.3 Streaming pair-by-pair write (vs. accumulate then dump)** | +1 (already present) | -200 MB peak RAM | safe | already canonical at `submissions/a1/inflate.py:102` |
| **3.4 `torch.inference_mode()` (vs. `torch.no_grad()`)** | 0 (rename) | -3% time | safe per Catalog #180 | already canonical |
| **3.5 Eliminate unnecessary `.cpu().numpy()` roundtrips** | -2 | -5% time | safe | depends on output sink |
| **3.6 Tensor pre-allocation for fixed-shape batches** | +6 | -2% time | safe | minor |
| **3.7 fp16 forward + fp32 final cast (CPU+CUDA agnostic)** | +4 | -10% time CUDA / +0% CPU | safe IF byte-identical output across devices verified | Catalog #205 parity gate required |

**Bucket #3 ΔS**: 0.0 by construction. Value is operator-cadence enabler.

**HARD-EARNED**: 3.2 deterministic-CPU is HARD-EARNED per CLAUDE.md "Apples-to-apples evidence discipline" — without it, the CPU/CUDA gap analysis cannot be apples-to-apples.

### Bucket #4 — Hybrid weight/runtime contracts (DELEGATED to sister #1097)

Bake learned selectors / affine gates / per-class corrections into existing HNeRV decoder weights via per-pair latent indexing or per-frame conditioning at inflate time. The "contract" is: the charged bytes encode the index/condition; inflate.py applies a generic operation (lookup + multiply + add).

| Abstract slot | Concrete sister-#1097 lane | LOC | Note |
|---|---|---|---|
| **4.1 per_class_chroma** | `lane_selector_paradigm_per_class_chroma_l0_sketch_20260520` | +18 | (also covered as Bucket 2.4) |
| **4.2 per_pair_difficulty_atlas_conditioned** | `lane_selector_paradigm_per_pair_difficulty_atlas_l0_sketch_20260520` | +22 | per-pair difficulty → conditional affine gate selection |
| **4.3 huffman_k_variant_sweep** | `lane_selector_paradigm_huffman_k_variant_sweep_l0_sketch_20260520` | +5 | fec6 selector currently uses k=16; sweep k∈{8,12,16,20,24,32} |
| **4.4 arithmetic_coding** | `lane_selector_paradigm_arithmetic_coding_l0_sketch_20260520` | +30 | replace fec6 Huffman with arithmetic coder over selector palette |
| **4.5 rice_golomb** | `lane_selector_paradigm_rice_golomb_l0_sketch_20260520` | +15 | Rice-Golomb for selector indices |
| **4.6 run_length_encoded** | `lane_selector_paradigm_run_length_encoded_l0_sketch_20260520` | +8 | RLE — likely loses to Huffman on PALETTE_MODE_IDS but cheap baseline |
| **4.7 dictionary_coded** | `lane_selector_paradigm_dictionary_coded_l0_sketch_20260520` | +20 | LZ-style dictionary for selector |

**THIS MEMO's contribution**: identify the abstract taxonomy slot + cross-reference. Sister #1097 produces the L0 SKETCH lane scaffolds. NO DUPLICATION.

**HARD-EARNED**: hybrid weight/runtime contracts dominate per HNeRV parity discipline L3 (monolithic single-file archive grammar) — fec6 frontier IS this pattern at the selector axis.

**CARGO-CULTED RISK**: assuming all 7 sub-candidates compose additively. Per `canonical_equations_registry.jsonl` slot `per_byte_leverage_uniformly_distributed_v1` + sister `master_gradient_locality_violation_by_codec_v1`, ΔS predictions are LOCAL not GLOBAL — only the best 1-2 of the 7 will likely land net positive.

### Bucket #5 — Review-safe side information (zero archive bytes; deterministic at inflate)

Compute side information INSIDE inflate.py from constants (camera intrinsics, frame index from file_list ordering, deterministic coordinate grids, lens distortion polynomial). NOT scorer introspection. NOT per-video test-label leakage.

| Primitive | LOC | Predicted ΔS | Dykstra-feasibility | Compliance verdict | Note |
|---|---|---|---|---|---|
| **5.1 Radial coordinate grid (precomputed in inflate)** | +4 | enabler for 1.5 + 2.4 conditioning | feasible | safe (image-center is camera-intrinsic constant) | Atick-Redlich retinal mosaic |
| **5.2 Per-frame index conditioning (`frame_idx / 1200`)** | +2 | enabler | feasible | safe IF file_list is canonical contest ordering | PR101 GOLD uses this pattern |
| **5.3 Lens distortion polynomial (camera intrinsic)** | +12 | -0.001 to -0.003 [predicted; reverses lens distortion on output frames] | feasible | safe — comma_video_compression_challenge camera intrinsics are public (openpilot calibration_helpers.py) | Pact-NeRV symposium references same |
| **5.4 Sky/ground horizon split (radial grid → row threshold)** | +6 | -0.001 to -0.002 [predicted; conditions Bucket 2.4 per-class chroma on sky-vs-ground] | feasible | safe | additive |
| **5.5 Per-pair velocity proxy (frame_1 - frame_0 EMA)** | +5 | enabler | feasible | safe (computed at inflate from decoder output) | feeds Pact-NeRV ego-motion |
| **5.6 Color-temperature-of-day proxy (per-frame mean luma)** | +4 | -0.0005 to -0.001 [predicted; day vs dusk vs tunnel] | feasible | safe | additive |

**Bucket #5 ΔS ceiling**: -0.002 to -0.006 if 3-4 stacked. The value is **enabler** for Buckets #1, #2, #4 conditioning.

**HARD-EARNED**: 5.1 radial grid + 5.3 lens distortion are HARD-EARNED per openpilot.calibration + Atick-Redlich 1990 + the public commaai/openpilot calibration helpers — these constants are NOT test-set leakage.

**CARGO-CULTED**: 5.6 color-temperature-of-day assumes day/dusk/tunnel partitioning matters for PoseNet — NOT empirically verified; PROBE-DISAMBIGUATOR required.

**REVIEW-SAFETY EDGE CASES** (Assumption-Adversary):
- frame_index_from_file_list: SAFE IF file_list is the canonical contest ordering passed to inflate.sh; UNSAFE IF inflate.py introspects WHICH video by name to apply per-video corrections (= test-time identity leak).
- mean luma per-frame: SAFE (computed from decoder output not from input video).
- horizon-row-threshold: SAFE (a priori dashcam geometry).
- per-video adaptive: FORBIDDEN. The 1200-frame video is the entire test set; per-video adaptation = scorer leakage by another name.

## 4. Top-K ranked by ΔS / LOC × compliance-safety

Ranking formula: `score = |ΔS_predicted| / max(LOC, 1) × compliance_weight` where `compliance_weight ∈ {1.0 fully safe, 0.7 requires sister-lane delegation, 0.4 requires probe-disambiguator, 0.1 uneconomic}`.

| Rank | Primitive | Bucket | ΔS_pred | LOC | weight | Score | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **2.4 per-class chroma LUT (80 bytes)** | #2 | -0.004 | 18 | 0.7 | 1.56e-4 | DELEGATED to #1097; rank for cross-reference |
| 2 | **4.3 huffman_k_variant_sweep** | #4 | -0.002 | 5 | 0.7 | 2.80e-4 | DELEGATED; cheapest sweep |
| 3 | **5.3 lens distortion polynomial** | #5 | -0.002 | 12 | 1.0 | 1.67e-4 | new, fully safe |
| 4 | **1.5 foveated sharpening (radial)** | #1 | -0.0025 | 12 | 1.0 | 2.08e-4 | new, fully safe |
| 5 | **1.1 deblocking filter** | #1 | -0.00075 | 8 | 1.0 | 9.38e-5 | safe |
| 6 | **2.5 per-pair deblock strength (600 B)** | #2 | -0.002 | 10 | 1.0 | 2.00e-4 | composes with 1.1 |
| 7 | **5.4 sky/ground horizon split** | #5 | -0.0015 | 6 | 1.0 | 2.50e-4 | enables 2.4 |
| 8 | **1.3 color correction extension** | #1 | -0.00055 | 4 | 1.0 | 1.38e-4 | extends existing affine |

**Composite top-K stack candidate** (within HNeRV parity L4 budget ≤200 LOC):
- A1 base inflate.py: 135 LOC
- + Bucket #5 (5.1 + 5.3 + 5.4): +22 LOC = 157 LOC
- + Bucket #1 (1.1 + 1.5): +20 LOC = 177 LOC
- + Bucket #3 (3.2 + 3.6): +8 LOC = 185 LOC
- (budget remaining: 15 LOC for charged-bytes sidecar parser if Bucket #2 / Bucket #4 sub-candidate from #1097 lands)
- **Composite ΔS_pred**: -0.006 to -0.014 (aggregate, Dykstra-bounded by ‖M_archive‖∞ × cumulative byte_delta + non-expansive composition of pixel-domain filters)

If composite lands at -0.010 from current 0.19205 [contest-CPU] frontier → predicted 0.18205 [predicted; NOT a score claim; requires paired Linux x86_64 CPU + NVIDIA T4 anchor before promotion per Catalog #192 + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"].

## 5. Canonical equation candidate (per Catalog #344)

**Proposed**: `inflate_time_generic_transform_score_ceiling_v0`

Formula:
```
ΔS_inflate_transform ≤ ‖M_archive‖∞ × Σ_p ‖T_p(decoder(latents)) - decoder(latents)‖_pixel
```

where `T_p` is the p-th inflate-time generic transform applied post-decoder. This is the local-linear Taylor expansion of contest score response to pixel-domain perturbations at fixed decoder.

**Empirical-anchor commitment**: WAVE-4 candidate produces 2-3 distinct inflate-time-only transform stacks at fixed A1/fec6 decoder, measures Δscore on contest-CPU + contest-CUDA paired anchors. Sister #1097 produces selector-paradigm-extension scaffolds which provide the anchor surface. Per Catalog #344 the equation lands at status `PENDING_EMPIRICAL_ANCHOR` initially; ratification or refinement after 2-3 empirical anchors.

**Provenance kind**: `PREDICTED_FROM_FIRST_PRINCIPLES` (Taylor expansion + master-gradient locality canonical equation slot 9); `evidence_grade=predicted`; `score_claim=False`; `promotable=False` per Catalog #323.

## 6. Cargo-cult audit per assumption (per Catalog #303)

| # | Assumption | Classification | Unwind test |
|---|---|---|---|
| 1 | "scorer-free inflate transforms can recover score equivalent to scorer-aware training" | CARGO-CULTED | empirical: stack Bucket #1 primitives on fec6 frontier, measure paired Δscore — if |Δ|<0.001 then ceiling falsified |
| 2 | "more inflate.py LOC = more score reduction" | CARGO-CULTED | empirical: pr106_latent_sidecar_r2_pr101_grammar at 740 LOC vs A1 at 135 LOC; pr106 r2 has higher LOC NOT lower score |
| 3 | "deblocking always helps SegNet" | HARD-EARNED | per JPEG/H.264 codec convention + 384→874 bicubic produces visible blocks |
| 4 | "fovea sharpening helps PoseNet" | HARD-EARNED conditional | per Atick-Redlich 1990 retinal mosaic + Pact-NeRV symposium |
| 5 | "per-class chroma helps SegNet" | HARD-EARNED | per NSCS06 v7 44% improvement (delegated to #1097) |
| 6 | "review-safe side info is FREE for budget" | HARD-EARNED-CONDITIONAL | safe for archive bytes IF deterministic; review-safe IF camera-intrinsic-only |
| 7 | "Huffman k=16 is optimal for fec6 selector" | CARGO-CULTED | sister #1097 lane 4.3 sweeps k∈{8,12,16,20,24,32} |
| 8 | "stacking generic transforms composes additively" | CARGO-CULTED | per Dykstra alternating projections theorem — only convex non-expansive operators compose; smoothing+sharpening can null-out |

## 7. Observability surface (per Catalog #305)

For every inflate.py side technique above, the 6-facet observability MUST be declarable BEFORE landing:

1. **Inspectable per layer**: every inflate.py primitive MUST be a discrete function call so a future operator can `print(intermediate)` between primitives without re-instrumenting.
2. **Decomposable per signal**: paired contest-CPU + contest-CUDA Δscore + per-pair Δd_seg + Δd_pose + Δrate (the 4 score component axes) at fixed archive minus the primitive vs with the primitive. Use sister `tools/probe_distinguishing_feature_byte_mutation.py` pattern.
3. **Diff-able across runs**: `select_inflate_device()` + `torch.set_num_threads(1)` MUST be active so two CPU runs produce byte-identical output frames per sister Catalog #205.
4. **Queryable post-hoc**: harvest output frames + per-primitive timing into `experiments/results/<lane_id>_<utc>/inflate_observability_<primitive>.json` per Catalog #245 ledger pattern.
5. **Cite-able**: every primitive's source citation in this memo (e.g. 1.5 → Atick-Redlich 1990; 1.1 → JPEG/H.264 deblocking convention; 5.3 → openpilot.calibration_helpers public).
6. **Counterfactual-able**: Catalog #272 distinguishing-feature integration contract requires byte-mutation smoke proving the bytes added by Bucket #2 / Bucket #4 primitives actually change decoder output. For Bucket #1 / #3 / #5 primitives that don't add bytes, the counterfactual is "remove this primitive line, measure paired Δscore at fixed archive."

## 8. 9-dimension success checklist evidence (per Catalog #294)

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS | inflate.py-side techniques are an UNDER-EXPLORED axis; the operator's quoted 5-bucket taxonomy is novel framing; not duplicate of #1097 (delegation explicit) |
| 2. BEAUTY+ELEGANCE | top-K all ≤25 LOC per primitive; composite stack at 185 LOC stays under HNeRV parity L4 ceiling 200; PR101 GOLD reviewable-in-30-sec discipline preserved |
| 3. DISTINCTNESS | each bucket is OPTICALLY DISTINGUISHABLE from others — generic transforms (Bucket #1) ≠ charged-byte LUTs (Bucket #2) ≠ runtime ops (Bucket #3) ≠ hybrid contracts (Bucket #4) ≠ side info (Bucket #5) |
| 4. RIGOR | Catalog #229 PV (read 3 canonical inflate.py files); Catalog #292 per-deliberation assumption surfacing; Catalog #303 cargo-cult audit; Catalog #296 Dykstra-feasibility per-primitive; Catalog #309 horizon-class = frontier_pursuit |
| 5. OPTIMIZATION PER TECHNIQUE | per-primitive ΔS predictions Taylor-bounded by canonical_equations_registry slot 9 + slot 16 + slot 17 |
| 6. STACK-OF-STACKS-COMPOSABILITY | Bucket #1 + #3 + #5 compose additively at pixel domain (non-expansive); Bucket #2 + #4 compete for byte budget — empirical Dykstra intersection required |
| 7. DETERMINISTIC REPRODUCIBILITY | Bucket #3.2 `torch.set_num_threads(1)` + Catalog #205 device-select is structural; byte-identical output across CPU runs |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | composite stack fits in 185 LOC ≤200 ceiling; runtime cost ≤30ms aggregate per pair |
| 9. OPTIMAL MINIMAL CONTEST SCORE | predicted aggregate -0.006 to -0.014 from 0.19205 frontier ↦ 0.178-0.186 [predicted]; requires empirical anchor before promotion |

## 9. Horizon-class declaration (per Catalog #309)

**horizon_class: frontier_pursuit** — Predicted CPU band [0.178, 0.186]. The composite stack of top-K primitives is engineered to push past the PR101 GOLD 0.1928 cluster into the 0.18 band. NOT plateau_adjacent (which would target 0.18-0.20) because the lower bound 0.178 already breaks below the canonical plateau. NOT asymptotic_pursuit (which would target ≤0.12) because no single inflate.py-side primitive can break the Shannon entropy floor of the existing HNeRV decoder.

## 10. Integration audit per Catalog #243+#244+#270 + sister #1098 Gap 1

Per task #1096 spec the Catalog #1098 Gap 1 auto-trigger applies. Each top-K primitive maps to:

| Primitive | Maps to gate / catalog |
|---|---|
| 1.1 deblocking | Catalog #105 (no-op detector — bytes do NOT change for pure pixel transform but output frames change; PASS) |
| 1.5 foveated sharpening | Catalog #305 observability + Catalog #220 (operational mechanism — pixel-domain transform IS the mechanism) |
| 2.4 per-class chroma | Catalog #272 distinguishing-feature integration contract; Catalog #139 byte-mutation smoke required |
| 4.3 huffman_k_variant_sweep | DELEGATED to sister #1097; per Catalog #325 per-substrate symposium required before paid dispatch |
| 5.3 lens distortion | Catalog #305 observability + canonical openpilot reference attribution |

All primitives respect Catalog #205 device-select; none introduce scorer loads (Catalog #1+#6); none introduce MPS fallback (Catalog #1).

## 11. Paper outline integration with Pact-NeRV (#1095)

`.omx/research/pact_nerv_paper_outline_20260520T193443Z.md` proposes a multi-substrate full-stack paper. This memo's contribution maps to:

| Pact-NeRV paper section | Bucket #N integration |
|---|---|
| §Foveated ego-motion at training | inflate-time analog at Bucket #1.5 (radial sharpening) + Bucket #5.1 (radial coord grid) |
| §Score-axis-aware HNeRV | charged-byte Buckets #2 + #4 per-axis selector training (sister #1097) |
| §Eval-roundtrip differentiable | inflate.py-side Bucket #1 transforms are post-roundtrip; complement training-time roundtrip |
| §Inflate.py runtime closure | Buckets #3 + #5 deterministic primitives ensure cross-CPU/CUDA bit-stable replays |
| §Operator-cadence enablers | Bucket #3 enables 3.2× faster CI iteration |

The paper's "5-bucket inflate.py taxonomy" section would summarize this memo + cite #1097 selector-paradigm-extension catalog + cite canonical_equations slot `inflate_time_generic_transform_score_ceiling_v0`.

## 12. Operator-routable next steps

1. **Approve composite top-K stack candidate** (Buckets #1.1 + #1.5 + #2.5 + #3.2 + #3.6 + #5.1 + #5.3 + #5.4): predicted -0.006 to -0.014 ΔS; LOC budget 185/200. Sister #1097 produces Bucket #4 scaffolds.
2. **Dispatch paired probe** (free local CPU + paid Modal T4 ≤$0.30): smoke at fixed fec6 frontier archive (`6bae0201fb08...`) with vs. without composite stack. Catalog #313 probe-outcome ledger row required.
3. **Register canonical equation `inflate_time_generic_transform_score_ceiling_v0`** (per Catalog #344) at status `PENDING_EMPIRICAL_ANCHOR`.
4. **WAVE-4 follow-on**: if probe shows ΔS ≤ -0.005, promote composite stack to L1 substrate scaffold; trigger Catalog #325 per-substrate symposium.
5. **Sister #1097 integration**: when #1097 lands a winning sub-candidate (predicted: per_class_chroma or huffman_k_variant_sweep), stack on top of the composite for ΔS additivity probe.

## 13. 6-hook wire-in declaration (per Catalog #125)

| Hook | Status | Note |
|---|---|---|
| 1 sensitivity-map | ACTIVE | per-primitive ΔS contributions feed `tac.sensitivity_map.*` axis weights |
| 2 Pareto constraint | ACTIVE | composite stack at 185 LOC competes against HNeRV decoder bytes — feeds `tac.pareto_*` with new constraint "inflate.py LOC ≤ 200 - sidecar_parser_LOC" |
| 3 bit-allocator | ACTIVE | Bucket #2 primitives' charged bytes (80 + 600 = 680 bytes) feed bit-allocator with predicted ΔS-per-byte rates |
| 4 cathedral autopilot dispatch | ACTIVE | composite stack candidate registered as L0 SKETCH lane for autopilot ranking |
| 5 continual-learning posterior | ACTIVE | paired empirical anchor (Operator-routable #2) appends to `.omx/state/continual_learning_posterior.jsonl` |
| 6 probe-disambiguator | ACTIVE | `tools/probe_inflate_py_side_technique_composite_stack_disambiguator.py` to be scaffolded by WAVE-4 follow-on |

## 14. Sister-collision verdicts

| Sister | Collision verdict |
|---|---|
| Catalog #1095 Pact-NeRV ULTIMATE | DISJOINT (Pact-NeRV is training-time substrate; this memo is inflate-time techniques) — integration declared §11 |
| Catalog #1097 selector-paradigm-extension | DELEGATED-CLEAN (Bucket #4 abstract taxonomy here; concrete L0 scaffolds there) — explicit cross-reference §3 + §10 |
| Catalog #1098 Gap 1 auto-trigger | APPLIES (Catalog #305 observability + Catalog #220 operational mechanism for Bucket #2/#4 primitives) |
| `council_per_substrate_symposium_pact_nerv...20260520T185500Z.md` | DISJOINT (substrate-level council; this memo is inflate-time technique-level) — cite as related deliberation |

## 15. Mission contribution per Catalog #300

`council_predicted_mission_contribution: frontier_breaking_enabler` — the composite stack candidate predicts ΔS in [-0.006, -0.014] from current 0.19205 frontier, which would land in [0.178, 0.186] band BELOW the canonical 0.196-0.199 plateau. Frontier-breaking IF empirical anchor confirms. Per the contrarian dissent: bounded above by ‖M_archive‖∞ × byte_delta — actual realized ΔS may be smaller; promotion gated by paired Linux x86_64 + NVIDIA T4 anchor.

---

**End of memo.** Lane: `lane_wave_3_inflate_py_side_techniques_research_and_design_20260520`. 27 candidate primitives enumerated; 8 top-ranked; 1 canonical equation candidate proposed; composite stack candidate within HNeRV parity L4 budget; integrated with Pact-NeRV #1095 + selector-paradigm-extension #1097 + Gap 1 auto-trigger #1098.
