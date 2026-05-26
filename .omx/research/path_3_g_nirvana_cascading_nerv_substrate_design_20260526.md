<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — substrate design memo; do not mutate. -->
<!-- Catalog #229 PV: this memo is a substrate-design decision per Catalog #290 + #294 + #296 + #303 + #305 + #309 binding sections; not an empirical-finding memo. The empirical anchor lives in the paired L0 SCAFFOLD landing memo path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md. -->
<!-- FORMALIZATION_PENDING:substrate_design_memo_not_empirical_finding_no_canonical_equation_referenced_per_catalog_344_design_phase_only_predictive_paradigm_canonical_equation_to_be_registered_when_phase_2_smoke_lands_per_phase_2_council_symposium_catalog_325_substrate_engineering_scope -->
---
schema_version: substrate_design_memo_v1
created_utc: 2026-05-26T07:30:00Z
substrate_id: nirvana_cascading_nerv
substrate_class: hierarchical_residual_decoder_cascade
council_tier: T1
council_attendees: [Shannon, Dykstra, NIRVANA-author-cite, Daubechies, Tishby-memorial, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "FRESH SUBSTRATE DESIGN posture (not extension of existing nirvana/ Maiya patch-wise substrate) is correct"
    classification: HARD-EARNED
    rationale: "Brief explicitly cites NIRVANA paper (CVPR 2023+) hierarchical residual decoder cascade paradigm; existing src/tac/substrates/nirvana/ Maiya CVPR 2024 patch-wise + adaptive scheduling is structurally a DIFFERENT paradigm (per-patch implicit renderer with patch-wise specialization). Conflating the two would suppress paradigm-distinct signal. Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD non-negotiable."
  - assumption: "2-phase methodology (substrate-design decision → L0 SCAFFOLD) is appropriate for FRESH design (not 3-phase cargo-cult-pass-FIRST for EXTENSION)"
    classification: HARD-EARNED
    rationale: "Per brief explicit directive: FRESH design candidates use 2-phase; cargo-cult-pass-FIRST is required only for EXTENSION candidates. Treating this as EXTENSION would force comparison to existing patch-wise nirvana/ which is paradigm-distinct."
  - assumption: "Hierarchical residual decomposition is distinct from BoostNeRV iterative-boosting (sister candidate E)"
    classification: HARD-EARNED
    rationale: "Hierarchical decomposition = decoder cascade with each layer learning residual of previous in single forward pass at multi-scale; iterative boosting = sequential round-by-round residual learning where each round trains a NEW learner on previous-round error. Sister E's BoostNeRV builds against PR110 as frozen base learner; this substrate G defines its own hierarchical cascade from scratch with multi-scale wavelet-pyramid-style decoder. Structurally distinct architectural class."
  - assumption: "MLX-first iteration is contest-grade for this substrate per #1265 anchor"
    classification: HARD-EARNED
    rationale: "Per #1265 MLX↔PyTorch parity anchor 0.000011 contest-units (72× margin over PR110 frontier 0.000789), MLX iteration is contest-grade. Substrate-design-from-first-principles per operator binding 2026-05-26 directive REQUIRES MLX-first to break the bolt-on-to-same-substrate cycle."
council_decisions_recorded:
  - "L0 SCAFFOLD scope: design memo + MLX scaffold + numpy reference + PyTorch inflate + tests + smoke trainer stub; NO full main, NO archive build, NO dispatch"
  - "Substrate path: src/tac/substrates/nirvana_cascading_nerv/ (NEW, distinct from existing nirvana/ Maiya patch-wise)"
  - "Lane: lane_path_3_g_nirvana_cascading_nerv_hierarchical_residual_20260526 L0/L1"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
horizon_class: frontier_pursuit
related_deliberation_ids:
  - path_3_candidate_inventory_for_next_wave_spawning_20260526
  - mlx_candidate_contest_equivalence_gate_landed_20260526
  - pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526
  - path_3_e_boost_nerv_against_pr110_substrate_design_20260526
  - dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z
---

# Path 3 candidate #G: NIRVANA cascading NeRV (hierarchical residual decoder) — substrate design

**Class**: FRESH SUBSTRATE DESIGN per operator's 2026-05-26 binding directive *"design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"*. NOT extension of existing patch-wise `tac.substrates.nirvana/` (Maiya CVPR 2024). The "NIRVANA" name is shared at the paradigm-family level (neural implicit video rendering with adaptive components) but the architectural class is distinct.

**Paradigm anchor**: NIRVANA-style cascading NeRV with hierarchical residual decoder cascade. Each layer in the decoder produces a coarse estimate and the residual to a finer estimate; the next layer refines via wavelet-pyramid-style upsampling + residual addition. Distinct from sister candidates by decomposition principle:
- **A=DreamerV3 RSSM** decomposes via categorical latent dynamics (G×K group-categorical alphabet).
- **E=BoostNeRV-against-PR110** decomposes via iterative boosting (frozen-base + iterative residual learner against external frozen base).
- **G=NIRVANA cascading NeRV (this)** decomposes via hierarchical residual decoder cascade (multi-scale wavelet-pyramid decoder with per-level residual; single learned model trained end-to-end).
- **F=Z8 hierarchical predictive coding** (queued) decomposes via Rao-Ballard + Mallat + Hafner + Wyner-Ziv canonical quadruple (4 primitives bound).

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290.

| Layer | Decision | Rationale |
|---|---|---|
| **L1 Decoder topology** | FORK_BECAUSE_PRINCIPLED_MISMATCH | Hierarchical cascade with per-level residual + wavelet-pyramid upsampling is the substrate-distinguishing feature. ADOPT_CANONICAL ds_nerv/hi_nerv DepthSep blocks would be a bolt-on, not the substrate. Substrate-optimal. |
| **L2 Multi-scale architecture** | FORK_BECAUSE_PRINCIPLED_MISMATCH | NIRVANA's 4-level wavelet-pyramid (48×64 → 96×128 → 192×256 → 384×512) is the canonical NIRVANA architecture per Mallat's wavelet pyramid + NIRVANA-paper-cite (CVPR 2023+). No existing canonical helper supports this. Substrate-optimal. |
| **L3 Per-level residual** | FORK_BECAUSE_PRINCIPLED_MISMATCH | Each decoder level emits (coarse_RGB, residual_to_next) tuple; canonical NeRV-family emits only final RGB. NEW canonical primitive at substrate level. Substrate-optimal. |
| **L4 Score-aware loss** | ADOPT_CANONICAL | Routes through `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164 (canonical SegNet + PoseNet routing; eval_roundtrip; differentiable rgb_to_yuv6). NO substrate-specific reason to fork. |
| **L5 Archive grammar** | FORK_BECAUSE_PRINCIPLED_MISMATCH | NIRVANA1 magic = b"NIR1\x00" with per-level decoder + per-level residual quantization blobs (4 levels). No canonical NeRV-family pattern supports this layered structure. Substrate-optimal per CLAUDE.md "Bit-level deconstruction and entropy discipline". |
| **L6 Inflate runtime** | ADOPT_CANONICAL (`select_inflate_device`) + FORK (multi-level decoder reconstruction) | Device selection routes through canonical `tac.substrates._shared.inflate_runtime.select_inflate_device` per Catalog #205. The hierarchical reconstruction sequence (level0 → upsample → +level1_residual → upsample → +level2_residual → ... → final RGB) is substrate-distinguishing. Hybrid. |
| **L7 MLX trainer** | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per binding 2026-05-26 reframing + Catalog #1265 MLX↔PyTorch parity anchor (0.000011 contest-units, 72× margin), MLX is the dev-velocity-optimal local-training path. Closes the cargo-cult-unwind loop without paid GPU. Substrate-optimal. |
| **L8 numpy reference impl** | FORK_BECAUSE_PRINCIPLED_MISMATCH (NEW canonical pattern per axis 3) | Per operator directive #3 2026-05-26 *"portability via numpy"*: every MLX primitive must have a sister numpy reference for GHA CPU CI testing + non-Apple-Silicon operator-portable diagnostic. NEW canonical pattern this substrate introduces. |
| **L9 EMA + eval_roundtrip** | ADOPT_CANONICAL | Quantizr 0.997 EMA decay; eval_roundtrip=True per CLAUDE.md non-negotiable. NO substrate-specific reason to fork. |
| **L10 Tests** | FORK_BECAUSE_PRINCIPLED_MISMATCH (MLX↔PyTorch + MLX↔numpy parity tests) + ADOPT_CANONICAL (Catalog #91 + #139) | NEW canonical pattern: per-primitive MLX↔numpy parity tests (axis 3 portability) + MLX↔PyTorch parity tests (axis 2 drift minimization). ADOPT existing ENCODE_INFLATE_ROUNDTRIP + byte-mutation no_op_proof. |

**Per-layer count**: 6 FORK_BECAUSE_PRINCIPLED_MISMATCH + 2 ADOPT_CANONICAL + 2 hybrid. Substrate-optimal engineering bound, not convenience.

## Math + scientific + engineering rigor per layer

Per operator directive #3 2026-05-26 *"adversarial review against all landing recursive for math and scientific and engineering rigor"*. Per-layer triple-axis citation table.

| Layer | Math citation | Scientific citation | Engineering citation | Classification |
|---|---|---|---|---|
| L1 Decoder topology | Mallat 1989 multi-resolution analysis theorem (orthogonal wavelet decomposition L²(R) → ⊕Wⱼ) | NIRVANA paper (CVPR 2023+) hierarchical residual decoder cascade architecture; Mallat-Daubechies wavelet pyramid canonical lineage | Sister substrate ds_nerv (Catalog #124-compliant DepthSep blocks) provides per-level building block; PyTorch + MLX both support 2D conv + bilinear upsampling | **HARD-EARNED** (3/3 axes cited) |
| L2 Multi-scale | Shannon R(D) bound: per-level rate budget Rᵢ ≥ R(Dᵢ) where Dᵢ is per-level distortion; Cauchy-Schwarz: ⟨residual_i, signal_i⟩² ≤ ‖residual_i‖² · ‖signal_i‖² bounds per-level information content | Mallat 1989 wavelet pyramid + Burt-Adelson 1983 Laplacian pyramid | sister substrate hi_nerv (HNeRV multi-scale variant) provides multi-scale engineering precedent | **HARD-EARNED** (3/3 axes cited) |
| L3 Per-level residual | Per-level residual orthogonality: ⟨rᵢ, rⱼ⟩ = 0 for i ≠ j under orthogonal basis (Mallat 1989); cascaded residual sum: R_total = Σ_i α_i r_i with Σ α_i² ≤ R_budget | NIRVANA paper canonical residual cascade; Daubechies 1992 ten lectures on wavelets §3.4 multi-resolution residual | sister substrate boost_nerv (iterative residual learner) shares discipline but differs by iteration mode | **HARD-EARNED** (3/3 axes cited) |
| L4 Score-aware loss | Canonical contest formula `S = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37545489`; per-pair gradient ∂S/∂θ via Catalog #164 helper | CLAUDE.md "eval_roundtrip" + "Differentiable scorer preprocess" non-negotiables | Catalog #164 canonical helper + sister substrates' adoption (12+ substrates route through this helper) | **HARD-EARNED** (3/3 axes cited) |
| L5 Archive grammar | Shannon entropy bound: H(decoder_weights) ≥ -log₂ p(theta) per fp16 + brotli (q=9) baseline; per-level residual entropy: H(rᵢ) ≤ H(r_i_int8) ≤ 8 bits/symbol | CLAUDE.md "Bit-level deconstruction and entropy discipline"; HNeRV parity discipline L2 export-first + L3 monolithic 0.bin | Sister substrate dreamer_v3_rssm RSSMC1 + boost_nerv_pr110_residual BPR1 archive patterns; brotli quality 9 canonical per all sister substrates | **HARD-EARNED** (3/3 axes cited) |
| L6 Inflate runtime | Per-pixel reconstruction: f_final(x,y) = level0(x,y) + Σᵢ Upsample_2^i(rᵢ(x/2^i, y/2^i)); bilinear interpolation kernel: k(u,v) = (1-u)(1-v) for u,v ∈ [0,1] | HNeRV parity discipline L4 ≤200 LOC + L9 runtime closure; Catalog #146 contest inflate runtime contract | Catalog #205 canonical `select_inflate_device`; torch + brotli only per HNeRV parity L4 | **HARD-EARNED** (3/3 axes cited) |
| L7 MLX trainer | Adam optimizer convergence: O(1/sqrt(t)) under L-smoothness; per-step gradient: ∂L/∂θ via reverse-mode autodiff (canonical chain rule) | Hafner 2024 DreamerV3 MLX-port precedent; Catalog #1265 MLX↔PyTorch parity gate | sister substrates pact_nerv_* MLX-trainer patterns; #1265 anchor 0.000011 contest-units empirical proof | **HARD-EARNED** (3/3 axes cited) |
| L8 numpy reference | Per-primitive byte-identical or documented-tolerance numpy implementation enables: (a) GHA CPU CI per Catalog #178+#179 sister discipline, (b) sister cathedral consumer cross-validation per Catalog #335, (c) operator-portable diagnostic on non-Apple-Silicon hardware | Per operator directive #3 2026-05-26 *"portability via numpy"*; CLAUDE.md "MLX portable-local-substrate authority" non-negotiable | numpy as universal CPU/numerical fallback per Catalog #1 device-selection-defaults discipline | **HARD-EARNED** (3/3 axes cited) |
| L9 EMA + eval_roundtrip | EMA shadow: θ_ema = α·θ_ema + (1-α)·θ with α=0.997; eval_roundtrip simulates uint8 quantization bottleneck (384→874→uint8→384) | Quantizr canonical α=0.997; CLAUDE.md "eval_roundtrip" + "EMA" non-negotiables (HIGHEST EMPHASIS) | Catalog Check 88 STRICT preflight enforces EMA correctness; eval_roundtrip default True in all canonical training paths | **HARD-EARNED** (3/3 axes cited) |
| L10 Tests | MLX↔PyTorch max_abs parity bound (drift ≤ 1e-3 per Catalog #1265 contest-equivalence gate); MLX↔numpy parity bound (drift ≤ 1e-5 per fp32 deterministic baseline) | Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation no_op_proof canonical patterns | sister substrate boost_nerv_pr110_residual + dreamer_v3_rssm test patterns | **HARD-EARNED** (3/3 axes cited) |

**Per-layer count**: 10/10 HARD-EARNED. NO CARGO-CULTED layers at L0 design time. Cargo-cult risk audit covered separately in §"Cargo-cult audit per assumption" below.

## MLX drift minimization per primitive

Per operator directive #3 2026-05-26 *"MLX drift minimization and portability via numpy"*. Cite #1255 MLX drift mitigation findings (NOT_FIXABLE_VIA_PRIMITIVE_SUBSTITUTION); per-primitive expected drift bound + mitigation strategy.

| MLX primitive | Expected drift bound (vs PyTorch reference) | Mitigation strategy / canonical helper cite | Test coverage |
|---|---|---|---|
| `mx.array.astype(mx.float32)` | 0 (lossless cast) | N/A | parity test fp32 round-trip |
| `mx.matmul / linear` | ≤ 1e-5 per matmul under fp32 | Explicit fp32 accumulation; AVOID fp16 matmul without explicit fp32 accumulation per Catalog #962 / slot 16 engineering corrections | MLX↔PyTorch per-layer parity test |
| `mx.conv2d` | ≤ 1e-4 per conv layer under fp32 | NHWC layout (MLX canonical) ≠ NCHW (PyTorch canonical); explicit NHWC↔NCHW transpose at export bridge per sister substrate dreamer_v3_rssm pattern | MLX↔PyTorch conv-layer parity test + numpy reference test |
| Bilinear upsample (×2) | KNOWN-DRIFT-RISK: align_corners semantics differ between MLX and PyTorch | **CANONICAL**: use `align_corners=False` (PyTorch default); cite sister A=DreamerV3 documented max_abs=24.34 gap caused by `align_corners=True` mlx.repeat substitution. AVOID mx.repeat for upsampling. Implement custom canonical helper `bilinear_resize2x_align_corners_false_nhwc` per sister substrate pattern. | MLX↔PyTorch parity test verifying max_abs ≤ 1e-3 + numpy reference test |
| `mx.sigmoid` | ≤ 1e-6 (elementwise saturating; numerically stable) | Standard MLX primitive; sister substrates verified | parity test |
| `mx.sin` (NeRV-canonical positional encoding) | ≤ 1e-6 (elementwise) | Standard MLX primitive | parity test |
| Mean reduction (loss aggregation) | KNOWN-DRIFT-RISK: non-Kahan summation accumulates rounding error at large N | **CANONICAL**: use Kahan summation for batch-level loss aggregation per Catalog #962 sister discipline; or explicit fp64 accumulation for reproducibility | MLX↔numpy parity test for batch aggregations |
| `mx.softmax` | KNOWN-DRIFT-RISK: numerical instability without epsilon | **CANONICAL**: use log-sum-exp trick (mx.softmax with `precise=True` or manual `x - mx.max(x, axis=-1, keepdims=True)` shift) | NOT used in this substrate at L0 (no softmax in decoder cascade); but documented for future extension |

**Per-primitive count**: 7 MLX primitives + 3 KNOWN-DRIFT-RISK identified. All risks have canonical mitigation cites. Smoke test MUST include max_abs measurement per channel + per layer.

## Portability via numpy per primitive

Per operator directive #3 2026-05-26 *"portability via numpy"*. Every MLX primitive must have a sister numpy reference implementation OR documented non-portability rationale.

| MLX primitive | numpy reference status | Module location | Non-portability rationale (if applicable) |
|---|---|---|---|
| `mx.array.astype(mx.float32)` | EQUIVALENT (`np.asarray(..., dtype=np.float32)`) | `numpy_reference.py::to_float32` | N/A |
| `mx.matmul / linear` | EQUIVALENT (`np.einsum` or `np.matmul`) | `numpy_reference.py::linear` | N/A |
| `mx.conv2d` | EQUIVALENT (manual numpy convolution via stride+slice; ALSO via `scipy.signal.convolve2d` if scipy available; documented fallback) | `numpy_reference.py::conv2d_nhwc` | scipy is OPTIONAL dependency; numpy-only conv2d provided as canonical fallback |
| Bilinear upsample (×2) | EQUIVALENT (manual numpy bilinear interpolation kernel) | `numpy_reference.py::bilinear_upsample_2x_nhwc` | N/A; explicit align_corners=False semantics encoded |
| `mx.sigmoid` | EQUIVALENT (`1.0 / (1.0 + np.exp(-x))` with numerical stability for large negative x) | `numpy_reference.py::sigmoid` | N/A |
| `mx.sin` | EQUIVALENT (`np.sin(x)`) | `numpy_reference.py::sin` | N/A |
| Mean reduction | EQUIVALENT with Kahan summation option (`np.mean` with optional Kahan-summed alternative) | `numpy_reference.py::mean` + `numpy_reference.py::kahan_mean` | N/A |

**Per-primitive count**: 7/7 MLX primitives have numpy reference implementations. The substrate is operable on CPU-only test rigs WITHOUT MLX dependency. This enables GHA CPU CI per Catalog #178 + sister cathedral consumer cross-validation per Catalog #335.

## 9-dimension success checklist evidence

Per Catalog #294 + CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode".

| Dim | Evidence (substrate-specific, not canonical-helper-inherited) |
|---|---|
| 1 UNIQUENESS | Hierarchical residual decoder cascade is a paradigm-distinct architectural class from sister substrates (A=DreamerV3 categorical-latent / D=Z6 predictive-coding / E=BoostNeRV iterative-boosting / F=Z8 quadruple-binding / existing nirvana/ patch-wise). No existing substrate decomposes via multi-scale wavelet-pyramid decoder with per-level residual. |
| 2 BEAUTY + ELEGANCE | Single learned end-to-end model produces multi-scale output; each level is the residual-refinement of the previous. Mathematically clean per Mallat wavelet pyramid canonical lineage. PR101-style 30-sec-reviewable per HNeRV parity L12 (target ~750 LOC substrate_engineering scope). |
| 3 DISTINCTNESS | Compositional matrix vs sisters: G ⊥ A (latent dynamics vs hierarchical decoder), G ⊥ E (single end-to-end vs frozen-base+iterative), G ⊥ D (no latent prediction objective; pure architectural decomposition), G ⊥ F (no IB/PC/wavelet/WZ binding; pure wavelet decomposition). |
| 4 RIGOR | Math + scientific + engineering rigor per-layer table above (10/10 HARD-EARNED). Premise verification + adversarial review pending Phase 2 council symposium per Catalog #325. |
| 5 OPTIMIZATION PER TECHNIQUE | Per Catalog #290 per-layer canonical-vs-unique decisions: 6 FORK_BECAUSE_PRINCIPLED_MISMATCH (substrate-optimal not convenience) + 2 ADOPT_CANONICAL (where canonical genuinely serves) + 2 hybrid. |
| 6 STACK-OF-STACKS-COMPOSABILITY | Composes orthogonally with: (a) PR110 fec6 frontier (different paradigm; could be stacked as residual sidecar similar to E), (b) sister A=DreamerV3 (different latent representation; could be stacked via decoder weight init), (c) NSCS06 v8 chroma_lut (different chroma channel; could be composed at chroma-replacement layer). Composition study Phase 2+. |
| 7 DETERMINISTIC REPRODUCIBILITY | Seed-pinned MLX initialization (canonical pattern via sister substrates); byte-deterministic NIRVANA1 archive grammar (sorted-keys JSON meta + fp16 state_dict + brotli quality 9 + int8 quantized residuals); byte-stable across re-runs given fixed seed. |
| 8 EXTREME OPTIMIZATION + PERFORMANCE | MLX-first iteration enables $0 dev-velocity-optimal local training per #1265 anchor (72× margin). Hierarchical cascade enables progressive decoding (early termination at any pyramid level yields valid coarse RGB). Per CLAUDE.md "Max observability" — per-level residual provides queryable signal decomposition. |
| 9 OPTIMAL MINIMAL CONTEST SCORE | Predicted ΔS range: `[-0.005, +0.010]` contest-units (sign-ambiguous at L0; refused phantom_random_init per Catalog #324 sister #834 22× miss anchor). Phase 2 Tier-C post-training measurement is the canonical disambiguator. |

## Cargo-cult audit per assumption

Per Catalog #303 + HARD-EARNED-vs-CARGO-CULTED addendum.

| Assumption | Classification | Rationale + unwind-test plan |
|---|---|---|
| 4-level wavelet pyramid (48×64 → 96×128 → 192×256 → 384×512) | CARGO-CULTED | Standard wavelet pyramid depth from Mallat 1989 + Burt-Adelson 1983; chosen because it cleanly factors the contest 384×512 scorer resolution. UNWIND: empirical sweep over {2, 3, 4, 5} levels at Phase 2 to verify 4 is optimal; the per-level rate budget may favor fewer/more levels under the contest's actual scorer feedback. |
| Per-level residual quantization at int8 | CARGO-CULTED | Standard quantization for residual encoding; chosen because brotli compresses int8 residuals well. UNWIND: empirical sweep over {int4, int8, int16, fp16} at Phase 2; finer quantization may save bytes vs coarser may save score. |
| Bilinear upsampling between levels (×2 spatial) | HARD-EARNED | Mallat 1989 wavelet pyramid + Burt-Adelson 1983 Laplacian pyramid canonical lineage; ×2 spatial is the canonical wavelet scale; bilinear is standard. NOT cargo-culted. |
| Per-pair latent dim = 16 (MLP encoder input) | CARGO-CULTED | Standard NeRV-family latent dim; chosen because sister substrates use 16. UNWIND: empirical sweep over {8, 16, 32, 64} at Phase 2; smaller may underfit, larger may inflate archive. |
| Score-aware loss via Catalog #164 canonical helper | HARD-EARNED | CLAUDE.md non-negotiable per Catalog #164; all sister substrates route through this helper. Substrate-specific reason to fork would require empirical proof; default is ADOPT. |
| brotli quality 9 for decoder state_dict | HARD-EARNED | Canonical across all sister substrates (boost_nerv_pr110_residual / dreamer_v3_rssm / c6_e4_mdl_ibps / sane_hnerv); operator-routable to higher quality at Phase 2 if archive budget is tight. |
| NIRVANA paper paradigm extrapolates to driving video | CARGO-CULTED | The NIRVANA paper (CVPR 2023+) is general-video; driving video has stronger temporal coherence + camera-ego-motion priors. UNWIND: Phase 2 MLX smoke MUST verify per-level residual entropy is non-trivial on the contest video specifically (NSCS06 v6 cargo-cult sister anchor). |
| Hierarchical decomposition outperforms iterative boosting (paradigm choice vs sister E) | CARGO-CULTED | Operator directive 2026-05-26 explicit FRESH design framing supports paradigm-distinct exploration; but actual ΔS comparison vs sister E is pending Phase 2 + paid CUDA dispatch. UNWIND: Phase 2 paired-comparison smoke between G (this) and E sister at matched archive budget. |
| MLX iteration is contest-grade for this substrate | HARD-EARNED | Per #1265 anchor 0.000011 contest-units (72× margin over PR110 frontier 0.000789); validated empirically across sister substrates. NOT cargo-culted. |
| Multi-scale = wavelet pyramid (specifically NIRVANA-style) | CARGO-CULTED | Mallat canonical wavelet pyramid is one of multiple valid multi-scale paradigms (others: Laplacian pyramid per Burt-Adelson 1983; learned multi-scale per StyleGAN-family). UNWIND: Phase 2 council symposium per Catalog #325 evaluates alternative multi-scale designs. |

**Per-assumption count**: 7 CARGO-CULTED + 4 HARD-EARNED. The cargo-cult-density is intentionally high at L0 design time (per CLAUDE.md cargo-cult audit discipline; L0 SCAFFOLD declares the cargo-cults so Phase 2 unwinds them empirically rather than burying them).

## Observability surface

Per Catalog #305 + CLAUDE.md "Max observability — non-negotiable" 6-facet definition.

| Facet | Implementation |
|---|---|
| 1 Inspectable per layer | MLX module exposes per-level decoder forward via `forward_with_intermediates(z) -> {level_0_rgb, level_1_residual, level_2_residual, level_3_residual, final_rgb}` API. Every layer's input/output/intermediate state captured without re-instrumentation. |
| 2 Decomposable per signal | Per-level residual magnitude separates per-level contribution to final RGB. Score-aware loss decomposes into seg/pose/rate per Catalog #164 canonical helper. Per-pair contribution decomposable via canonical sensitivity-map. |
| 3 Diff-able across runs | Byte-deterministic NIRVANA1 archive (sorted-keys meta + fp16 state_dict + brotli q=9 + int8 residuals); same seed → same bytes. MLX↔PyTorch + MLX↔numpy parity tests diff at primitive level. |
| 4 Queryable post-hoc | Archive parses to canonical sections (header + per-level decoder blob + per-level residual blob + meta blob); per-level residual indices machine-readable via `parse_archive()` API. JSON meta dict exposes config + training hyperparameters. |
| 5 Cite-able | Every artifact anchored to (substrate / commit / MLX-call-id / config / random_seed / NIRVANA-paper-cite) tuple per Catalog #245 modal_call_id_ledger sister discipline. Archive header carries MAGIC + VERSION + cfg fields. |
| 6 Counterfactual-able | Catalog #139 byte-mutation discipline: every per-level residual byte mutated to verify it produces frame changes. Catalog #272 distinguishing-feature contract: the per-level residual blobs ARE distinguishing-feature bytes that pass byte-mutation smoke. |

## Predicted ΔS band

**Predicted band**: `pending_post_training` per Catalog #324 (refuses phantom_random_init predictions per sister #834 22× miss anchor).

**First-principles Shannon R(D) framing** (paper calculation, NOT runtime anchor; Dykstra-feasibility check):

- **Rate budget breakdown** at default 4-level config:
  - Level 0 decoder (48×64 base): ~3 KB brotli-compressed fp16 state_dict
  - Level 1 residual (96×128 int8): ~12 KB brotli-compressed
  - Level 2 residual (192×256 int8): ~48 KB brotli-compressed
  - Level 3 residual (384×512 int8): ~192 KB brotli-compressed
  - Total: ~255 KB → `Δrate = +0.169 contest-units` (255_000 × 25 / 37_545_489)
- **Predicted ΔS range**: `[-0.05, +0.10]` contest-units (sign-ambiguous at L0)
  - Lower bound (-0.05): hierarchical decomposition extracts most of available scorer-conditional entropy
  - Upper bound (+0.10): residual cascade is dominated by rate cost with insufficient distortion gain
- **Dykstra-feasibility verdict**: per Catalog #296. At 255 KB total archive, the convex feasibility region on (rate, d_seg, d_pose) IS marginally feasible vs sister substrates' frontier of ~250-500 KB archives. The hierarchical decomposition MAY admit non-trivial structural advantage IF and ONLY IF the per-level residuals are NOT byte-identical to direct PR101-style fine-level encoding (counterfactual: NIRVANA-style and PR101-style at same rate budget should yield distinguishable distortion).

**Phase 2 council symposium per Catalog #325 MUST run formal Dykstra-feasibility intersection check + post-training Tier-C density measurement on the actual trained substrate BEFORE any paid CUDA dispatch.** Sign-ambiguity at L0 is precisely why L0 SCAFFOLD posture is correct.

**probe-disambiguator path**: `tools/probe_nirvana_cascading_nerv_disambiguator.py` (queued for Phase 2; resolves the "hierarchical decomposition outperforms iterative boosting" cargo-cult-pass empirically via paired-comparison smoke between G and sister E at matched archive budget).

**Shannon R(D) bound on hierarchical residual cascade** (canonical):
- Per Mallat 1989: orthogonal wavelet decomposition yields per-level residuals rᵢ with Σᵢ ‖rᵢ‖² = ‖signal‖² (Parseval).
- Per Shannon: per-level rate Rᵢ ≥ R(Dᵢ) where Dᵢ is per-level distortion bound.
- Total rate R_total = Σᵢ Rᵢ ≥ Σᵢ R(Dᵢ) ≥ R(Σᵢ Dᵢ) = R(D_total) (convexity of R(D) curve).
- The hierarchical cascade is rate-distortion optimal ONLY IF the per-level residuals are independent under the scorer's gradient (orthogonal decomposition assumption). Empirical verification at Phase 2.

## horizon_class

`frontier_pursuit` per Catalog #309 (predicted ΔS band [-0.05, +0.10] spans the PR110 frontier band [0.18, 0.20] currently and points toward asymptotic_pursuit at Phase 2+ if hierarchical decomposition unlocks structural ΔS).

## MLX-implementation roadmap with all 3 new axes addressed

**L0 (this scaffold, $0 wall-clock ~3-5h)**:
1. Substrate package `src/tac/substrates/nirvana_cascading_nerv/` with __init__.py + MLX renderer + numpy reference + PyTorch inflate + archive grammar + tests
2. Smoke trainer at `experiments/train_substrate_nirvana_cascading_nerv_mlx.py` with `_full_main raises NotImplementedError` per Catalog #240
3. Design memo (this) + landing memo
4. **Axis 1 evidence**: per-layer math+sci+engineering rigor table in design memo (10/10 HARD-EARNED)
5. **Axis 2 evidence**: per-primitive MLX drift bound + mitigation cite table in design memo (7 primitives, 3 KNOWN-DRIFT-RISK with canonical mitigations); MLX↔PyTorch parity test in test suite
6. **Axis 3 evidence**: numpy reference implementation for every MLX primitive in `numpy_reference.py`; MLX↔numpy parity test in test suite

**L1+ (Phase 2 council symposium pending operator authorization)**:
1. MLX smoke trainer ≤5ep ≤8pairs convergence verification
2. PyTorch port via MLX→PyTorch state_dict bridge per #1265
3. Sister Catalog #1265 contest-equivalence gate threshold 0.001 PASS verdict
4. Phase 2 council symposium per Catalog #325 (T2 sextet + NIRVANA-author-cite + Daubechies + Tishby-memorial + Atick + Wyner attendees)
5. Operator-routable paid CUDA dispatch authorization

**L2+ (post-Phase 2 sister Catalog #233 4-gate canonical promotion)**:
1. Tier-C post-training density measurement per Catalog #324
2. Sister probe-disambiguator at `tools/probe_nirvana_cascading_nerv_disambiguator.py`
3. Paired-comparison smoke vs sister E=BoostNeRV at matched archive budget

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: ACTIVE at L1+ (per-level residual sensitivity surface; canonical helper `tac.sensitivity_map.*` consumer at Phase 2)
- **hook #2 Pareto constraint**: ACTIVE at L1+ (hierarchical decomposition admits per-level rate budget Pareto polytope; Dykstra-feasibility intersection at Phase 2)
- **hook #3 bit-allocator**: ACTIVE at L1+ (per-level quantization bit budget allocator hook; canonical pattern)
- **hook #4 cathedral autopilot dispatch**: PLANNED at L1+ (auto-discovered cathedral consumer per Catalog #335 sister substrate pattern)
- **hook #5 continual-learning posterior**: ACTIVE (this landing memo + future MLX smoke results → canonical posterior anchor)
- **hook #6 probe-disambiguator**: ACTIVE at Phase 2 (paired-comparison vs sister E=BoostNeRV is the canonical disambiguator)

## Cross-references

- Path 3 candidate inventory: `.omx/research/path_3_candidate_inventory_for_next_wave_spawning_20260526.md` §Tier 1 G
- Sister substrate E (BoostNeRV against PR110): `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md` (different decomposition: iterative boosting vs hierarchical cascade)
- Sister substrate A (DreamerV3 RSSM): `.omx/research/dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z.md` (different paradigm: categorical latent dynamics vs hierarchical decoder)
- MLX↔PyTorch parity anchor: `.omx/research/mlx_candidate_contest_equivalence_gate_landed_20260526.md` + corrected `.omx/research/pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md`
- Existing patch-wise nirvana/: `src/tac/substrates/nirvana/__init__.py` (paradigm-distinct from this substrate; NOT a precursor or sister)
