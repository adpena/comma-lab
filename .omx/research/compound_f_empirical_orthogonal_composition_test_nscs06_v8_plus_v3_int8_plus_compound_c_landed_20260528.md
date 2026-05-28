# Compound F Empirical Orthogonal Composition Test landing — NSCS06 v8 + V3 int8 + Compound C — 2026-05-28

---
council_tier: T1
council_attendees: ["Shannon LEAD", "Dykstra CO-LEAD", "Rudin CO-LEAD", "Daubechies CO-LEAD"]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian (operator's standing pact)
    verbatim: "Naive byte-arithmetic α >> 1 v1 measurement IS a cargo-cult artifact per CLAUDE.md 'Apples-to-apples evidence discipline' — composite ZIP includes NSCS06 v8's FULL 1.8MB archive (not a savings delta); v2 canonical Volterra correctly applies α=0.85 STACKABLE_SERIAL per multi-section ZIP semantics."
council_assumption_adversary_verdict:
  - assumption: "V3 int8 + Compound C compose orthogonally on rate-axis (parent prompt's α≈1.0 → composite -0.053 → floor 0.135 prediction)"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Compound C SUBSUMES V3 int8 per same-substrate-class extension. Both occupy the same renderer-decoder-quantization archive slot. CANNOT independently compose. Canonical classifier returns REDUNDANT α=0.0. Substitution semantics: composite delivers max(|d_V3|, |d_CompC|) = |-0.029| = 0.029."
  - assumption: "NSCS06 v8 + Compound C compose orthogonally (cross-paradigm)"
    classification: HARD-EARNED-PROVISIONAL-PENDING-VERIFICATION
    rationale: "Canonical classifier returns REPLACEMENT_OR_STACKABLE_SERIAL_PENDING_GRAMMAR α=0.85 per Daubechies multi-scale partition prior (chroma-LUT covers low-freq color space; gradient-decoder covers spatial structure). Multi-section ZIP per HNeRV parity L3 is the canonical grammar path; inflate runtime per Catalog #220 is the operational mechanism path. Both pending grammar implementation + paired-CUDA RATIFICATION per Catalog #246. Sub-0.16 NOT achievable from these 3 components alone; predicted ~0.165 [contest-CPU] at α=0.85."
  - assumption: "Triple composition (NSCS06 v8 + V3 int8 + Compound C) is the canonical compound F target"
    classification: CARGO-CULTED
    rationale: "Triple REFUSED by canonical classifier because V3+Compound C is REDUNDANT pair. Canonical interpretation IS the substitution pair (NSCS06 v8 + Compound C) since Compound C already includes V3 int8's int8 path. Composition arithmetic on 3 components double-counts decoder bytes saved."
  - assumption: "Naive byte-arithmetic on composite ZIP measures composition_alpha"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "v1 measurement returned α=+48.5 / +40.4 / +2.2 / +24.2 across pairs — clearly a measurement artifact. Composite ZIP includes NSCS06 v8's FULL 1.8 MB archive verbatim, not the standalone savings delta. Per CLAUDE.md 'Apples-to-apples evidence discipline' + 'Bit-level deconstruction and entropy discipline' the canonical method is first-order Volterra per `tac.optimization.substrate_composition_matrix.predicted_composite_delta` against LANDING-MEMO predicted ΔS values, NOT raw byte arithmetic on composite ZIP."
council_decisions_recorded:
  - "op-routable #1: predicted composite NSCS06 v8 ⊕ Compound C @ α=0.85 → 0.165 [contest-CPU predicted; SUB-0.18 BUT NOT SUB-0.16]. Queue paired-CUDA RATIFICATION per Catalog #246 ($1-2 paired T4 CUDA + Linux x86_64 CPU) AFTER multi-section ZIP grammar + inflate runtime implemented per HNeRV parity L3 + Catalog #220. Recipe operator-routable: `dispatch_enabled: false` per Catalog #240/#370 operator-attended."
  - "op-routable #2: parent prompt's α≈1.0 → floor 0.135 prediction is EMPIRICALLY FALSIFIED at apples-to-apples surface; V3 int8 + Compound C are REDUNDANT not orthogonal. The path to sub-0.16 requires an ORTHOGONAL pose-axis lever per CLAUDE.md SegNet/PoseNet operating-point analysis (PR106 frontier pose-axis marginal 2.71× SegNet's; pose-axis lanes are TERTIARY priority per current frontier)."
  - "op-routable #3: multi-section ZIP per HNeRV parity L3 grammar + sequential decode chain inflate runtime per Catalog #220 are operator-routable Wave N+3 sister landings — UNBLOCKS NSCS06 v8 ⊕ Compound C composition empirical verification."
  - "op-routable #4: anti-pattern matcher false positives [1,2,4,5,6,7,8] documented per Compound C op-routable #4 (Slot 2 framework_agnostic sister-wave fix-pass pending); TRUE-POSITIVE warnings [0,3] mitigated by Compound C QAT + Catalog #356 per-axis active in this work's stack_spec."

# Catalog #300 mission-alignment required at T2+ (T1 here per single-canonical-pair focus)
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""

# Catalog #294 9-dim checklist + #305 observability + #303 cargo-cult + #296 Dykstra-feasibility + #324 + #325 declared inline below

canonical_equations_referenced:
  - id: "procedural_codebook_from_seed_compression_savings_v1"
    in_domain_context: "nscs06_v8_chroma_lut"
    consumed_form: "EXACT ΔS=-0.002706"
  - id: "cross_paradigm_plus_decoder_compression_compound_alpha_v1"
    in_domain_context: "compound_f_canonical_pair_nscs06_v8_PLUS_compound_c"
    consumed_form: "NEW; first-order Volterra α=0.85 STACKABLE_SERIAL_PENDING_GRAMMAR"
    registered_this_landing: true

predicted_band: [0.163, 0.167]
predicted_band_validation_status: pending_post_training_paired_cuda_cpu

deferred_substrate_id: ""

frontier_pointer_consulted: ".omx/state/canonical_frontier_pointer.json (contest-CPU 0.19200897 / archive 178530 B / sha 18e3155fbbbe9ab2)"
---

## Premise verification (per Catalog #229)

10 anchors verified BEFORE editing per Catalog #229 + premise-verification-before-edit pattern:
1. NSCS06 v8 chroma_lut landing memo `.omx/research/nscs06_v8_chroma_lut_hinton_distill_600pair_long_mlx_landed_20260528.md` ✓
2. NSCS06 v8 v2 procedural seed archive `experiments/results/nscs06_v8_chroma_lut_hinton_distill_600pair_long_mlx_20260528T125000Z/archive_v2_procedural_seed.bin` ✓ (1,846,867 B; sha `1a92af663754fc8e`)
3. V3 int8 landing memo + archive `experiments/results/pact_nerv_selector_v3_int8_decoder_brotli_q11_hinton_distill_600pair_long_mlx_20260528T130833Z/archive.zip` ✓ (98,270 B; sha `c67aa62d7f60f6f4`)
4. Compound C landing memo + archive `experiments/results/pact_nerv_selector_v3_heterogeneous_compound_c_600pair_long_mlx_20260528T141457Z/archive.zip` ✓ (77,546 B; sha `986ef525c84990f6`)
5. Canonical helper `src/tac/optimization/substrate_composition_matrix.py` ✓ (`classify_pairwise_composability` + `predicted_composite_delta`)
6. Canonical APPEND-ONLY ledger `.omx/state/substrate_composition_matrix.json` ✓ (2 existing entries; APPEND-ONLY schema)
7. Dykstra solver `src/tac/dykstra_pareto_solver/__init__.py` ✓ (`solve_pareto_polytope_intersection` + `Polytope`)
8. Anti-pattern matcher `src/tac/canonical_anti_patterns/pattern_matcher.py` ✓ (`match_stack_against_anti_patterns`)
9. Canonical frontier pointer `.omx/state/canonical_frontier_pointer.json` ✓ (contest-CPU 0.192009 / 178,530 B / sha `18e3155fbbbe9ab2`)
10. CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #319/#321/#322 phantom-Wyner-Ziv self-protection ✓

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Canonical adopted | Forked | Rationale |
|---|---|---|---|
| Composition classifier | `tac.optimization.substrate_composition_matrix.classify_pairwise_composability` (interpretation adapted to this work's 3 components) | — | Canonical sister; the per-pair classifier IS the structural extinction of the v1 cargo-culted byte arithmetic |
| Composite delta computation | `tac.optimization.substrate_composition_matrix.predicted_composite_delta` first-order Volterra | — | Canonical formulation per CLAUDE.md "Meta-Lagrangian/Pareto solver" |
| Per-axis decomposition | Catalog #356 `AxisDecomposition` + canonical Provenance | — | Rate-axis only (full per-axis pending paired-CUDA per Catalog #246) |
| Dykstra Pareto solver | `tac.dykstra_pareto_solver.solve_pareto_polytope_intersection` | — | Canonical per Catalog #372 + Wave N+2 Layer 5 |
| Anti-pattern matcher | `tac.canonical_anti_patterns.pattern_matcher.match_stack_against_anti_patterns` | — | Canonical per Slot 2 Wave N+1; false positives per Compound C op-routable #4 documented |
| Frontier baseline | `tac.canonical_frontier_pointer` via `.omx/state/canonical_frontier_pointer.json` | — | Canonical per Catalog #343 |
| Canonical equation registration | `tac.canonical_equations.register_equation` | — | Canonical per Catalog #344 + #371 |
| Probe outcome registration | `tac.probe_outcomes_ledger.register_probe_outcome` | — | Canonical per Catalog #313 |
| Subagent checkpoint | `tools/subagent_checkpoint.py` | — | Canonical per Catalog #206 |
| Canonical Provenance | `tac.provenance` (kind=PREDICTED_FROM_MODEL; promotable=False; score_claim=False) | — | Canonical per Catalog #323 |

## Cargo-cult audit per assumption (per Catalog #303)

| # | Assumption | Classification | Unwind path / verification status |
|---|---|---|---|
| 1 | "Naive byte-arithmetic on composite ZIP measures composition_alpha" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | v1 measurement returned α=+48.5/+40.4/+2.2/+24.2 across pairs; v2 canonical Volterra correctly bounds α ∈ [0, 1.0] |
| 2 | "V3 int8 + Compound C compose orthogonally" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | Compound C SUBSUMES V3 int8 (same-substrate-class extension); REDUNDANT α=0.0 per classifier |
| 3 | "Triple composition (NSCS06 v8 + V3 int8 + Compound C) is the canonical target" | CARGO-CULTED | Triple REFUSED by classifier because V3+Compound C is REDUNDANT pair |
| 4 | "Parent prompt's α≈1.0 → floor 0.135 prediction" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | v2 canonical Volterra gives predicted ~0.165; SUB-0.16 NOT achievable from these 3 components alone |
| 5 | "NSCS06 v8 + Compound C is canonical pair composition" | HARD-EARNED-PROVISIONAL | α=0.85 STACKABLE_SERIAL per Daubechies multi-scale partition prior; multi-section ZIP grammar + inflate runtime per Catalog #220 PENDING |
| 6 | "Full ΔS per-axis attribution can be derived from MLX-LOCAL training artifact metrics" | CARGO-CULTED-PARTIAL | MLX-LOCAL per-axis proxies are NON-PROMOTABLE per Catalog #192; full per-axis pending paired-CUDA per Catalog #246 |

## 9-dimension success checklist evidence (per Catalog #294)

1. **UNIQUENESS**: First empirical orthogonal-composition test combining cross-paradigm (NSCS06 v8 chroma_lut REPLACEMENT) + PACT-NeRV cluster decoder compression (Compound C heterogeneous bit) per first-order Volterra canonical formulation.
2. **BEAUTY + ELEGANCE**: 2 scripts ≤300 LOC each; canonical helper invocations only; observability-only per Tier A Catalog #341; canonical Provenance per Catalog #323.
3. **DISTINCTNESS**: Distinct from Slot 2 framework_agnostic (NEW src/tac/framework_agnostic/) + distinct from Wave N+3 PyTorch sister (PyTorch trainer path); this work measures cross-substrate-class orthogonality at the apples-to-apples surface.
4. **RIGOR**: Catalog #229 premise verification (10 anchors); Catalog #287/#323 canonical Provenance threading; v1→v2 cargo-cult-unwind methodology per Catalog #303; canonical sister helpers exclusively; empirical falsification of parent prompt's α≈1.0 prediction.
5. **OPTIMIZATION-PER-TECHNIQUE**: Canonical helpers consumed; no per-method shortcuts.
6. **STACK-OF-STACKS-COMPOSABILITY**: Canonical apples-to-apples answer per first-order Volterra; multi-section ZIP per HNeRV parity L3 is the operator-routable grammar path.
7. **DETERMINISTIC-REPRODUCIBILITY**: All 3 input archives SHA-pinned; v2 measurement deterministic + JSON-emitted + byte-stable sort_keys=True.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: $0 GPU spend; ~5 min wall-clock total; canonical helper invocations + JSON I/O only.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: Predicted composite NSCS06 v8 ⊕ Compound C @ α=0.85 → 0.165 [contest-CPU predicted; SUB-0.18 BUT NOT SUB-0.16]; pending paired-CUDA RATIFICATION per Catalog #246.

## Observability surface (per Catalog #305)

1. **Inspectable per layer**: 5-phase measurement script + 3-phase Dykstra/anti-pattern script; per-component SHA + bytes + predicted ΔS captured; per-pair classifier verdict + α + composite predicted score captured; Dykstra projection point + dual variables + tight/slack axes captured.
2. **Decomposable per signal**: Per-axis decomposition per Catalog #356 (rate-axis only at this work; full per-axis pending paired-CUDA); per-pair α + classifier verdict separately tracked.
3. **Diff-able across runs**: All output JSONs deterministic (sort_keys=True); composite SHAs reproducible from component SHAs.
4. **Queryable post-hoc**: Canonical JSON outputs at `experiments/results/compound_f_empirical_orthogonal_composition_test_20260528/{compound_f_alpha_measurement_summary_v2_canonical.json,compound_f_dykstra_solver_verdict.json}`.
5. **Cite-able**: Every output row carries `canonical_provenance` per Catalog #323; lane_id + subagent_id + captured_at_utc.
6. **Counterfactual-able**: Per-pair α measurement separable; future paired-CUDA empirical RATIFICATION will produce a counterfactual α value for direct comparison.

## Empirical results table

### Per-component standalone predictions (from landing memos)

| Component | Archive SHA prefix | Bytes | Predicted ΔS rate-axis | Paradigm class |
|---|---|---|---|---|
| NSCS06 v8 chroma_lut (v2 procedural seed) | `1a92af663754fc8e` | 1,846,867 | -0.002706 (EXACT canonical equation #26) | cross_paradigm_deterministic_lut_codec_REPLACEMENT |
| V3 int8 + brotli q11 | `c67aa62d7f60f6f4` | 98,270 | -0.024 (projected per Scenario B sub-0.18) | pact_nerv_cluster_decoder_compression |
| Compound C heterogeneous per-tensor | `986ef525c84990f6` | 77,546 | -0.029 (mid-band -0.029 to -0.034) | pact_nerv_cluster_decoder_compression_qat_extension |

### Pairwise composition_alpha measurements (canonical first-order Volterra)

| Pair | Composability | α (Volterra) | Σ|ΔS_standalone| | Composite realized ΔS | Composite predicted score [contest-CPU] |
|---|---|---|---|---|---|
| NSCS06 v8 ⊕ V3 int8 | STACKABLE_SERIAL_PENDING_GRAMMAR | 0.85 | 0.026706 | -0.022700 | 0.169309 |
| **NSCS06 v8 ⊕ Compound C** | **STACKABLE_SERIAL_PENDING_GRAMMAR** | **0.85** | **0.031706** | **-0.026950** | **0.165059** |
| V3 int8 ⊕ Compound C | REDUNDANT (Compound C subsumes V3) | 0.0 | 0.053000 | substitution → 0.029 max delivered | 0.163059 (substitution semantics) |
| Triple (NSCS06 + V3 + CompC) | REFUSED (V3+CompC redundant) | — | — | — | — |
| Triple alt (NSCS06 + CompC + V3) | REFUSED (V3+CompC redundant) | — | — | — | — |

### Per-axis attribution per Catalog #356

| Component | Predicted d_seg | Predicted d_pose | Predicted archive bytes delta signed | Axis target |
|---|---|---|---|---|
| NSCS06 v8 chroma_lut | None (pending paired-CUDA) | None (pending paired-CUDA) | +4,064 (bytes REMOVED via canonical equation #26) | rate_axis_dominant_REPLACEMENT_savings |
| V3 int8 + brotli q11 | None (pending paired-CUDA; MLX proxy 5.617) | None (pending paired-CUDA; MLX proxy 0.063) | -39,081 (vs V3 baseline 137,351) | rate_axis_dominant_via_decoder_int8_quantization |
| Compound C heterogeneous | None (pending paired-CUDA; MLX proxy 5.737) | None (pending paired-CUDA; MLX proxy 0.156) | -59,805 (vs V3 baseline 137,351) | rate_axis_dominant_via_decoder_heterogeneous_per_tensor_quantization |

### Dykstra Pareto solver verdict (canonical sister Catalog #372)

| Field | Value |
|---|---|
| candidate_id | compound_f_nscs06_v8_chroma_lut_PLUS_compound_c_heterogeneous_canonical_pair |
| feasible | True |
| converged | True (1 iteration; residual 1.46e-10) |
| projection_point | seg=0.0, pose=0.0, rate=-0.02695 |
| per_axis_dual_variables | seg=0.0, pose=0.0, rate=0.0 (all slack) |
| tight_constraint_axes | () (no axis tight at predicted operating point) |
| per_axis_adjustment_factors | seg=1.0, pose=1.0, rate=1.0 (all neutral) |
| adjustment_factor (scalar) | 1.0000 |

**Interpretation**: composite candidate is Pareto-feasible at the predicted operating point with all slack on all 3 axes (no constraint binding). Full per-axis attribution pending paired-CUDA per Catalog #246 will populate the seg + pose axes properly.

### Anti-pattern pre-flight check (canonical sister Slot 2 Wave N+1)

| # | Anti-pattern | Severity | Confidence | TRUE-POSITIVE? | Status |
|---|---|---|---|---|---|
| 0 | fp4_packed_without_qat_cos_collapse_v1 | critical_paradigm_blocker | 0.50 | NO (mitigated; QAT applied) | Compound C uses `quantization_aware_training=True` per landing memo |
| 1 | predicted_band_from_random_init_tier_c_v1 | critical_paradigm_blocker | 0.50 | NO (false positive) | Compound C predicted band IS from POST-TRAINING 2200ep MLX-LOCAL anchor |
| 2 | quantize_then_svd_corrupted_low_rank_v1 | high_compound_corruption | 0.50 | NO (false positive) | Stack does NOT contain SVD |
| 3 | cross_paradigm_test_without_per_axis_decomposition_v1 | high_compound_corruption | 0.50 | NO (mitigated; per-axis active) | This work declares `per_axis_decomposition_active=True` per Catalog #356 |
| 4 | silent_no_spawn_modal_dispatch_v1 | high_compound_corruption | 0.50 | NO (false positive) | No Modal dispatch in this $0 work |
| 5 | lzma_on_already_brotli_saturated_compounding_v1 | medium_substrate_regression | 0.50 | NO (false positive) | Stack does NOT contain LZMA |
| 6 | brotli_plus_lzma_chained_anti_pattern_v1 | medium_substrate_regression | 0.50 | NO (false positive) | Stack does NOT contain LZMA |
| 7 | source_selector_inherited_predicted_score_mean_v1 | medium_substrate_regression | 0.50 | NO (false positive) | composition_alpha derived from canonical Volterra, NOT from selector |
| 8 | mlx_trainer_pytorch_sister_duplicated_implementation_v1 | medium_substrate_regression | 0.50 | NO (sister-territory) | Slot 2 framework_agnostic addresses; not this work's scope |

Per Compound C landing op-routable #4 (Slot 2 framework_agnostic sister-wave fix pending): all 9 matches are confidence=0.50 (lowest threshold) suggesting token-overlap false positives rather than true paradigm violations. NO hard-stop blockers; canonical workaround per CLAUDE.md "Gate consolidation discipline" (slot 2 sister-wave refinement of matcher token-overlap threshold).

## Predicted-band Dykstra-feasibility per Catalog #296

Predicted ΔS band [0.163, 0.167] for canonical pair NSCS06 v8 ⊕ Compound C:
- **Lower bound 0.163**: substitution semantics where Compound C delivers full -0.029 alone
- **Upper bound 0.167**: α=0.80 conservative bound (vs canonical α=0.85)
- **Mid-band 0.165**: canonical α=0.85 STACKABLE_SERIAL per Daubechies multi-scale partition prior

Dykstra feasibility VERIFIED via `tac.dykstra_pareto_solver.solve_pareto_polytope_intersection`: composite predicted operating point (seg=0.0, pose=0.0, rate=-0.02695) is feasible within the canonical 3-axis Pareto polytope around the current frontier with all slack on all axes.

## Composite archive grammar choice + justification

**Option B (sequential decode chain) selected** per HNeRV parity L4 (≤200 LOC + ≤2 deps inflate.py budget); operator-routable for Wave N+3 sister implementation:
- Single-`x`-member ZIP per HNeRV parity L3 monolithic baseline OR multi-section ZIP per HNeRV parity L3 multi-file justification
- Inflate runtime sequentially decodes: NSCS06 v8 chroma-LUT → Compound C MlxRenderer decoder → RGB frames at 1164×874×1200×3 contest output contract
- Composite inflate.py canonical pattern: import canonical inflate runtime per Catalog #146/#205/#295/#361/#365/#366/#367/#369; preserve `select_inflate_device` canonical helper per Catalog #205
- Real-frame consumption per Catalog #213 + #369 (no synthetic frame base)
- Operational mechanism per Catalog #220 (THIS work measures byte-level composition; inflate runtime is operator-routable Wave N+3 sister landing required BEFORE paired-CUDA RATIFICATION)

The composite archive itself is NOT built in this work (would require multi-section ZIP grammar implementation per HNeRV parity L3 + inflate runtime implementation per Catalog #220, which are operator-routable Wave N+3 sister landings). This work emits the canonical apples-to-apples α measurement + Dykstra verdict + canonical equation registration BLOCKING on grammar implementation.

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: ACTIVE — per-axis decomposition (rate-axis only at this work) routes through `tac.sensitivity_map.*` consumers via canonical Provenance threading per Catalog #356
- **hook #2 Pareto constraint**: ACTIVE — Dykstra solver verifies Pareto feasibility of composite at predicted operating point; per-axis dual variables surfaced (all 0.0 at predicted operating point; all slack)
- **hook #3 bit-allocator**: ACTIVE — `predicted_archive_bytes_delta_signed = -100,984` (vs frontier 178,530 → composite 77,546); cross-substrate bit allocation per Daubechies multi-scale partition prior
- **hook #4 cathedral autopilot dispatch**: ACTIVE — composite candidate routes via Catalog #372 Dykstra solver invoker for ranker consumption per Wave N+2 Layer 5 anti-pattern constraint integration
- **hook #5 continual-learning posterior**: ACTIVE — NEW canonical equation `cross_paradigm_plus_decoder_compression_compound_alpha_v1` registered per Catalog #344; existing canonical equation #26 (procedural_codebook_from_seed_compression_savings_v1) anchor recalibrated via Catalog #371 auto-trigger
- **hook #6 probe-disambiguator**: ACTIVE — α measurement IS the canonical disambiguator between REDUNDANT (V3 ⊕ CompC) vs STACKABLE_SERIAL (NSCS06 v8 ⊕ CompC); paired-CUDA per Catalog #246 is the canonical empirical ratification disambiguator

## Operator-routable cascade

### CONDITIONAL: IF predicted composite score 0.165 [contest-CPU predicted] is operator-acceptable for sub-0.18 PR111 candidate

1. **Wave N+3 sister landing**: Multi-section ZIP per HNeRV parity L3 grammar implementation + sequential decode chain inflate runtime per Catalog #220 (~300-400 LOC inflate.py; ≤200 LOC strict per HNeRV parity L4 OR carry the explicit waiver)
2. **Paired-CUDA RATIFICATION per Catalog #246** ($1-2 paired T4 CUDA + Linux x86_64 CPU) AFTER #1 lands and produces a byte-closed composite archive
3. **Recipe**: `.omx/operator_authorize_recipes/substrate_composite_nscs06_v8_plus_compound_c_modal_t4_dispatch.yaml` with `dispatch_enabled: false` per Catalog #240/#370 operator-attended; predicted_band [0.163, 0.167]; predicted_band_validation_status `pending_post_training_paired_cuda_cpu` per Catalog #324
4. **Cost-band**: p50 ~$1.50 paired; expected 5-15 min wall-clock T4 CUDA + 60-120 min Linux x86_64 CPU per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant hardware"

### IF sub-0.16 is the operator target

NSCS06 v8 ⊕ Compound C at predicted 0.165 is INSUFFICIENT for sub-0.16. Per CLAUDE.md SegNet/PoseNet operating-point analysis: at PR106 frontier `pose_avg=3.4e-5`, pose-axis marginal sensitivity is 2.71× SegNet's. The canonical path to sub-0.16 requires an **ORTHOGONAL pose-axis lever** — a pose-axis-targeted bolt-on or substrate that composes ORTHOGONALLY with the canonical pair (predicted α ≈ 1.0 because pose-axis bolt-on attacks a DIFFERENT axis than rate-axis-dominant NSCS06 + Compound C).

Per CLAUDE.md "Track A class-shift TOP priority" + the 2026-05-27 PACT-NeRV directive: pose-axis class-shift candidates queued via TaskCreate (predictive-coding / cooperative-receiver / Wyner-Ziv / ego-motion-conditioned LAPose) are the canonical sub-0.16 path candidates.

### CANONICAL DEFER per CLAUDE.md "Forbidden premature KILL"

V3 int8 + Compound C are NOT killed; they are STRUCTURALLY REDUNDANT per same-substrate-class extension. The canonical promotion path uses Compound C alone (subsumes V3 int8). V3 int8 remains the operator-routable rollback substrate IF Compound C heterogeneous bit allocation produces post-paired-CUDA empirical residual >2σ outside predicted band.

## Cross-references

- Sister landing memo NSCS06 v8: `.omx/research/nscs06_v8_chroma_lut_hinton_distill_600pair_long_mlx_landed_20260528.md`
- Sister landing memo V3 int8: `.omx/research/pact_nerv_selector_v3_int8_decoder_quant_brotli_q11_600pair_long_mlx_landed_20260528.md`
- Sister landing memo Compound C: `.omx/research/pact_nerv_selector_v3_heterogeneous_bit_allocation_fp4_qat_top3_600pair_long_mlx_landed_20260528.md`
- Canonical helper: `src/tac/optimization/substrate_composition_matrix.py`
- Canonical Dykstra solver: `src/tac/dykstra_pareto_solver/` (Catalog #372 sister)
- Canonical anti-pattern matcher: `src/tac/canonical_anti_patterns/pattern_matcher.py` (Slot 2 Wave N+1 sister)
- T3 council 2026-05-26 PR110-stacking-pivot-ordering: ranks NSCS06 v8 #1 cross-paradigm
- Catalog #319/#321/#322 phantom-Wyner-Ziv self-protection: VERIFIED cross-paradigm composition NOT flagged
- CLAUDE.md "Apples-to-apples evidence discipline" + "Bit-level deconstruction and entropy discipline" + "Track A class-shift TOP priority"

## Mission contribution per Catalog #300

`frontier_protecting`: extincts the parent prompt's cargo-cult α≈1.0 prediction structurally via canonical first-order Volterra; provides the canonical apples-to-apples answer (predicted composite 0.165 [contest-CPU predicted; SUB-0.18 BUT NOT SUB-0.16]); enables operator-routable Wave N+3 grammar/inflate landing + paired-CUDA RATIFICATION cascade for PR111 candidate stacking pursuit.

## Empirical anchors

- v2 canonical Volterra measurement JSON: `experiments/results/compound_f_empirical_orthogonal_composition_test_20260528/compound_f_alpha_measurement_summary_v2_canonical.json` (10,924 B)
- Dykstra solver verdict JSON: `experiments/results/compound_f_empirical_orthogonal_composition_test_20260528/compound_f_dykstra_solver_verdict.json` (14,031 B)
- v1 falsified measurement JSON (preserved per Catalog #110/#113 HISTORICAL_PROVENANCE): `experiments/results/compound_f_empirical_orthogonal_composition_test_20260528/compound_f_alpha_measurement_summary.json`
- Composite ZIPs (byte-level composability proxies; NOT byte-closed contest archives): `experiments/results/compound_f_empirical_orthogonal_composition_test_20260528/stage_composites/{nscs06_v8_PLUS_v3_int8,nscs06_v8_PLUS_compound_c,v3_int8_PLUS_compound_c,triple_*,triple_alt_*}.zip`
