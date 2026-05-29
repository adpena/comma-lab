# PR110-OPT-4 Grouped Color/Geometry Calibration with Cross-Pair Perturbation Reuse — Design Memo (L0 SCAFFOLD)

**Date:** 2026-05-29T07:00:00Z (Slot X)
**Lane:** `lane_slot_x_pr110_opt_4_grouped_color_geometry_calibration_l0_scaffold_cross_pair_perturbation_reuse_mlx_local_20260529`
**TaskCreate:** #1316 (PR110-OPT-4)
**Operator:** Slot X cap≥4 maintenance per directive 2026-05-29 "continue MLX first and focused on score lowering and why we haven't produced on original frontier score yet"
**Sister Non-Interference (Catalog #302/#314/#340):** DISJOINT with Slot R (PR110 × SYNTHESIZE_FRAME at `src/tac/substrates/_shared/synthesize_frame_emission_atick_redlich`), Slot U (`src/tac/composition/super_additive_alpha_4_74_*`), Slot V (`.omx/research/why_have_we_not_produced_original_frontier_score_meta_diagnostic_*`), Slot W (`src/tac/substrates/z6_v2/identity_predictor_disambiguator`). THIS slot OWNS: `src/tac/composition/pr110_opt_4_grouped_color_geometry_calibration/` + `src/tac/tests/test_pr110_opt_4_*` + this design memo.

## horizon_class

`horizon_class: plateau_adjacent`

Per CLAUDE.md "HORIZON-CLASS evaluation axis" standing directive 2026-05-16 + canonical frontier pointer at `.omx/state/canonical_frontier_pointer.json`: live PR110 frontier ≈ 0.1920 [contest-CPU] sits in the canonical PLATEAU-ADJACENT band [0.180, 0.200]. PR110-OPT-4 grouped color/geometry calibration cross-pair perturbation reuse targets within-class refinement of the PR110 archive grammar (selector stream + 16-symbol palette + 600-pair format) — within-class by construction. Per Catalog #309 + CATHEDRAL-SMARTER-DESIGN-MEMO Dim 6: this is NOT a class-shift candidate; the design honors plateau-adjacent semantics and does NOT make asymptotic-pursuit claims.

## TL;DR Bayesian honesty

**Wave N+34 analytical investigation (commit `wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json`) ALREADY verdict'd OPT-4 grouped color/geometry calibration as `IMPLEMENTATION_FALSIFIED`**: Shannon-coded grouped wire estimate = 258 bytes vs FEC6 baseline = 249 bytes = **+9 bytes WORSE**. Fixed-width grouped wire estimate = 383 bytes = **+134 bytes WORSE**. Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307: this is **IMPLEMENTATION-LEVEL FALSIFICATION** not paradigm-level. The grouping paradigm (cross-pair perturbation reuse) remains DEFERRED-PENDING-RESEARCH per Catalog #308 alternative-reducer enumeration.

This L0 SCAFFOLD landing therefore serves the canonical dual role:
1. **Preserve the canonical analytical primitive** as a queryable system surface so future widened-catalog probes can compare against the baked-in 17-group / 21-mode counts without re-deriving them.
2. **Enumerate alternative reducer methodologies per Catalog #308** so the operator can route the next iteration through one of N≥3 candidates (NOT just the falsified Shannon-coded grouping).

The L0 SCAFFOLD does NOT claim score improvement; the `verdict` field is `DEFERRED_PENDING_ALTERNATIVE_REDUCER` per Catalog #308. Wave N+34 IMPLEMENTATION_FALSIFIED verdict is preserved verbatim as the canonical historical anchor.

## Predicted ΔS band

`predicted_band: [+0.0, +0.0001]` (rate-axis only; component-axis = 0 because L0 SCAFFOLD does not perturb frames)

**Dykstra-feasibility citation per Catalog #296:** The grouped-encoder approach intersects 3 polytope constraints — (a) wire-bytes ≤ 249 (FEC6 baseline; binding); (b) decoded-symbol-fidelity = lossless (binding); (c) header + per-group menu overhead ≤ 6 bytes (FEC6 binding). Wave N+34's `258B Shannon-coded` upper bound + `383B fixed-width` upper bound prove constraint (a) is INFEASIBLE for the 21-mode source distribution; Dykstra alternating-projections per Catalog #372 would similarly report no feasible intersection. Per the canonical Pareto-polytope-intersection equation `dykstra_pareto_polytope_intersection_compounding_v1` per Catalog #344: predicted ΔS upper-bound from this approach is `+0.0 to +0.0001` (worse-or-equal to FEC6 baseline rate-axis contribution) when the 17-group / 21-mode distribution is the source.

**Probe-disambiguator per Catalog #296 (alternative-reducer enumeration):**
The empirical L0 SCAFFOLD smoke (Phase C) confirms or refutes the analytical upper bound via byte-closed `compute_grouped_wire_bytes(source_modes_per_pair, group_size_strategy)` invocation. Sister probe paths per Catalog #308 alternative-reducer methodologies:
- `tools/probe_pr110_opt_4_widened_catalog_disambiguator.py` (DEFERRED — requires widened mode-catalog beyond the 21 modes Wave N+34 sourced)
- `tools/probe_pr110_opt_4_per_region_grouping_disambiguator.py` (DEFERRED — per-region grouping instead of per-pair-class grouping)
- `tools/probe_pr110_opt_4_temporal_window_grouping_disambiguator.py` (DEFERRED — temporal-window grouping vs per-frame grouping)
- `tools/probe_pr110_opt_4_per_segnet_class_chroma_disambiguator.py` (DEFERRED — sister of master-gradient exploit #5 per Catalog #354)

## Cargo-cult audit per assumption

Per Catalog #303 sister discipline + hard-earned-vs-cargo-culted addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`):

| # | Assumption | Classification | Rationale | Unwind Path |
|---|---|---|---|---|
| 1 | Grouped color/geometry calibration applicable to PR110 archive grammar | **HARD-EARNED** | Wave N+34 sourced 21 modes from `experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl`; PR110 wire format already uses 16-symbol K=16 palette per FEC6 design | N/A — empirical |
| 2 | Cross-pair perturbation reuse compatible with PR110 16-symbol palette + 600-pair format | **CARGO-CULTED** | The 16-symbol palette was empirically derived for FEC6 0-order Huffman coding; cross-pair reuse implies HIGHER-ORDER context (sister of Variant B Markov which IS the canonical higher-order context coder for this stream). PR110-OPT-4 grouping at the per-symbol level does NOT compose with FEC6 0-order Huffman without RE-DESIGNING the wire format | Test alternative wire formats (FEC8 Markov per Variant B; sister with grouping) OR DEFER to widened-catalog probe |
| 3 | Per-region vs per-pair vs per-frame perturbation budget grouping | **CARGO-CULTED** | Wave N+34 used per-pair-class grouping (21 → 17 groups via "best mode per pair" reducer); other reducers per Catalog #308 (per-region / per-frame / per-temporal-window) UNPROBED | Enumerate alternative reducers per Catalog #308; each reducer is its own L0 SCAFFOLD sister candidate |
| 4 | The 17-group fan-out is the dominant grouping cardinality | **CARGO-CULTED** | "21 modes → 17 unique best-modes over 600 pairs" is the Wave N+34 empirical reducer output; widening the catalog may produce different cardinality | Widened-catalog probe DEFERRED per Wave N+34 reactivation criteria |
| 5 | Shannon-coded entropy is the relevant lower bound | **HARD-EARNED** | Per Shannon 1948 source coding theorem; the canonical CACM-87 32-bit arithmetic coder (sister Variant B implementation) achieves within 0.01 bits/symbol of Shannon | N/A — first-principles |
| 6 | Header + per-group menu overhead is < 6 bytes | **CARGO-CULTED** | FEC6 used 6-byte header; PR110-OPT-4 grouping needs MORE header bytes for per-group menu lookup table (G × 1 byte = 17 bytes minimum for 17-group case) | Empirical measurement; the L0 SCAFFOLD's `compute_grouped_wire_bytes` helper exposes this directly |
| 7 | 21-mode source distribution from `experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl` is representative | **HARD-EARNED** | Direct measurement from live PR101 paired-component sweep | N/A — empirical |

## 9-dimension success checklist evidence

Per Catalog #294 sister discipline:

1. **UNIQUENESS**: Grouped color/geometry calibration with cross-pair perturbation reuse is structurally distinct from Variant A (0-order adaptive arith) + Variant B (Markov context) + Variant C (variable-K escape) per Wave N+34's `composition_analysis.orthogonality_matrix` — `opt_4_opt_11: MUTUALLY_EXCLUSIVE_OPT_4_REDUCES_ALPHABET_OPT_11_EXPANDS_TO_K_SQUARED`.

2. **BEAUTY + ELEGANCE**: L0 SCAFFOLD ≤250 LOC per HNeRV parity L4 bolt-on budget; canonical 30-sec-reviewable shape: input `source_modes_per_pair: list[int]` → `compute_grouped_wire_bytes(...) -> int` + `apply_grouped_color_geometry_calibration_to_pr110_archive(...) -> tuple[bytes, AxisDecomposition]`. NO training scaffold + NO subprocess invocation + NO mutable state.

3. **DISTINCTNESS**: Cross-pair perturbation reuse is DISTINCT from PR110 × SYNTHESIZE_FRAME (Slot R; Atick-Redlich cooperative-receiver) + super_additive alpha=4.74 (Slot U; lane_g_v3 × siren cross-substrate composition) + meta-diagnostic (Slot V) + identity-predictor-disambiguator (Slot W) per ownership map.

4. **RIGOR**: Premise verification per Catalog #229 — Wave N+34 analytical JSON read + confirmed `verdict: IMPLEMENTATION_FALSIFIED`. Adversarial review per inner-council sextet pact (Shannon LEAD: rate-axis upper bound proven; Dykstra CO-LEAD: 3-constraint polytope infeasible; Contrarian: "+9 bytes worse already empirically established"; Assumption-Adversary: cargo-cult audit above; Rudin CO-LEAD: per-cell rule-list reducer enumeration per Catalog #308; Daubechies CO-LEAD: multi-scale partition prior per Catalog #277 wavelet hierarchy applies to per-region grouping sister probe). Empirical anchor per Wave N+34 + sister L0 SCAFFOLD smoke (Phase C; macOS-CPU advisory; non-promotable per Catalog #192).

5. **OPTIMIZATION PER TECHNIQUE** (covered by sister Catalog #290 canonical-vs-unique decision per layer below).

6. **STACK-OF-STACKS-COMPOSABILITY**: Per Wave N+34 `composition_analysis.orthogonality_matrix`: `opt_4_opt_7: PARTIALLY_COMPATIBLE_DIFFERENT_SURFACES_OPT_4_REDUCES_ALPHABET_OPT_7_SPARSIFIES_SELECTOR`. PR110-OPT-4 is composable with OPT-7 (UNIWARD inverse-scorer) under sub-additive composition (the empirical compounding factor α must be empirically anchored per Catalog #322 NOT assumed additive).

7. **DETERMINISTIC REPRODUCIBILITY**: Byte-stable (no random sampling in encoder/decoder); seed-pinned via canonical `_compute_grouping_signature(source_modes_per_pair) -> bytes` hash function.

8. **EXTREME OPTIMIZATION + PERFORMANCE**: L0 SCAFFOLD is encoder-side only ($0 inflate-time cost); helper `compute_grouped_wire_bytes(source_modes_per_pair, group_size_strategy)` is O(N_pairs × log_2(G_groups)) for the grouping decision + O(N_pairs) for the encoding pass. Decoder LOC for inflate path: ≤80 LOC (group-menu lookup table + per-group palette decode); within HNeRV parity L4 ≤200 LOC inflate budget.

9. **OPTIMAL MINIMAL CONTEST SCORE**: Per Wave N+34 verdict the analytical upper bound is `+0.0 to +0.0001` rate-axis WORSE than FEC6 baseline. THIS IS THE EXPLICIT PARADIGM-VS-IMPLEMENTATION ACKNOWLEDGMENT per Catalog #307: the L0 SCAFFOLD does NOT claim score improvement; reactivation criteria are pinned per Wave N+34 reactivation anchor `If OPT-4 verdict=PROCEED_CANDIDATE: spawn full widened-catalog 600-pair sweep`.

## Canonical-vs-unique decision per layer

Per Catalog #290 sister discipline + CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

| Layer | Decision | Rationale |
|-------|----------|-----------|
| Archive grammar adapter | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | The PR110 archive grammar (16-symbol K=16 palette + 600-pair FEC6 0-order Huffman) does not have a canonical grouping primitive; the per-archive adapter is substrate-specific. The L0 SCAFFOLD adapter is ≤80 LOC and isolated from sister archive grammars. |
| Canonical Provenance | **ADOPT_CANONICAL_BECAUSE_SERVES** | Use `tac.provenance.builders.build_provenance_for_predicted` per Catalog #323 sister discipline; canonical helper IS the appropriate primitive for L0 SCAFFOLD predicted-score-claim emission. |
| AxisDecomposition emission | **ADOPT_CANONICAL_BECAUSE_SERVES** | Use `tac.cathedral.consumer_contract.AxisDecomposition` per Catalog #356 sister discipline; canonical per-axis surface contract IS the appropriate primitive for downstream Pareto polytope solver consumption per Catalog #372. |
| Tier A canonical-routing markers | **ADOPT_CANONICAL_BECAUSE_SERVES** | Per Catalog #341 + #357: L0 SCAFFOLD predicted-score returns MUST carry `predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"`. |
| MLX-LOCAL smoke runner | **ADOPT_CANONICAL_BECAUSE_SERVES** | Use `tac.optimization.macos_cpu_advisory_signal.append_manifest_row_to_jsonl` per Catalog #192 sister discipline; canonical advisory-grade smoke runner IS the appropriate primitive for $0 macOS-CPU exploration. |
| Framework-agnostic core | **ADOPT_CANONICAL_BECAUSE_SERVES** | Core math operates on `list[int]` / `bytes` / `numpy.ndarray` only (no torch/mlx dependency for the encoder/decoder primitive). Framework-agnostic by construction; consumable by both MLX-LOCAL smoke + future Vast.ai/Modal CUDA dispatch via the same `compute_grouped_wire_bytes` entry point. |
| Test fixtures | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Wave N+34 source dataset `pair_component_rows.jsonl` carries 21 modes × 600 pairs; the L0 SCAFFOLD tests bake the 17-group fan-out as a regression guard (sister of Wave N+34 anchor). Test fixtures are substrate-specific. |

## Observability surface

Per Catalog #305 + CLAUDE.md "Max observability — non-negotiable" 6-facet definition:

1. **Inspectable per layer**: `compute_grouped_wire_bytes(...)` returns dict with per-group fanout + per-symbol counts + Shannon entropy estimate + header overhead breakdown.
2. **Decomposable per signal**: `AxisDecomposition` emission separates `predicted_d_seg_delta=0.0` + `predicted_d_pose_delta=0.0` + `predicted_archive_bytes_delta=+9 to +134` (signed; positive = WORSE per Wave N+34 verdict).
3. **Diff-able across runs**: Byte-stable encoder; `_compute_grouping_signature(source_modes_per_pair) -> sha256` enables diffing two run outputs.
4. **Queryable post-hoc**: All outputs JSON-serializable via `as_dict()` and `AxisDecomposition.as_dict()` (Catalog #356 sister discipline).
5. **Cite-able**: Canonical Provenance per Catalog #323 + `source_path = "<predictor:pr110_opt_4_grouped_color_geometry_calibration_l0_scaffold>"` + `source_sha256` over input modes + `canonical_helper_invocation` field.
6. **Counterfactual-able**: `compute_grouped_wire_bytes_for_alternative_grouping(...)` helper accepts arbitrary `group_size_strategy ∈ {shannon_coded, fixed_width, per_region, per_temporal_window}` so the operator can ask "what if we grouped by region instead?" without re-running the encoder.

## Sister cross-references

- Wave N+34 canonical artifact: `.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json` (analytical investigation; IMPLEMENTATION_FALSIFIED verdict).
- PR110-OPT-3 canonical pattern: Variant A `.omx/research/pr110_opt3_adaptive_arith_landed_20260526.md` + Variant B `.omx/research/pr110_opt3_variant_b_markov_landed_20260526.md` + Variant C `.omx/research/pr110_opt3_variant_c_variable_k_escape_mechanism_landed_20260526.md`.
- Slot L Symposium 1 (PR110 × SYNTHESIZE_FRAME): `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_l_slot_h_top_3_super_additive_per_substrate_symposium_prep_landed_20260529.md`.
- Canonical equation candidate (DEFERRED until paired-CUDA empirical anchor; per Catalog #344): `pr110_opt_4_grouped_color_geometry_calibration_cross_pair_perturbation_reuse_savings_v1`.
- Canonical anti-pattern (DEFERRED until 3+ empirical falsification anchors land; per Catalog #344 sister): `pr110_opt_4_grouped_color_geometry_calibration_implementation_falsified_at_wave_n_34_v1` (already anchored at Wave N+34 IMPLEMENTATION_FALSIFIED verdict).

## Acceptance criteria for promotion (DEFERRED-PENDING-RESEARCH per Catalog #308)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" — reactivation criteria pinned:

1. Widened mode catalog (≥40 modes vs Wave N+34's 21 modes) shows G_groups ≤ 8 (sufficient for fixed-width 3-bit menu to fit in 6 bytes header).
2. OR: per-region grouping with G_groups ≤ 4 (sufficient for fixed-width 2-bit menu to fit in 4 bytes header).
3. OR: per-temporal-window grouping with G_groups ≤ 6 (sufficient for fixed-width 3-bit menu to fit in 5 bytes header).
4. OR: composition with OPT-7 UNIWARD sparse selector per Wave N+34 `opt_4_opt_7: PARTIALLY_COMPATIBLE` matrix entry; empirical paired-smoke required per Catalog #373 Dykstra Pareto polytope feasibility.

Per Catalog #313 probe outcomes ledger: this gate emits `DEFER` verdict with 30-day reactivation window per Catalog #298 staleness discipline.

## Mission contribution per Catalog #300

`council_predicted_mission_contribution: apparatus_maintenance`

THIS L0 SCAFFOLD is `apparatus_maintenance` (Catalog #300 enum); it preserves the Wave N+34 analytical primitive as a queryable system surface + enumerates alternative reducer methodologies per Catalog #308 for future iterations. It does NOT make a frontier-breaking claim (verdict = IMPLEMENTATION_FALSIFIED per Wave N+34); it does NOT consume rigor budget beyond the L0 SCAFFOLD landing wall-clock (~60-90 min). Score-lowering work routes through alternative reducer probes per the canonical reactivation criteria above (not through this L0 SCAFFOLD directly).

Per the 4-theme operator directive 2026-05-29:
- (a) AUDIT NEGATIVE RESULTS: ✅ Wave N+34 IMPLEMENTATION_FALSIFIED verdict preserved + alternative reducer enumeration per Catalog #308.
- (b) MLX-FIRST: ✅ $0 macOS-CPU advisory smoke per Catalog #192 (non-promotable).
- (c) FOCUS SCORE LOWERING: ✅ Alternative reducer enumeration provides 4 op-routable paths; PR110 attack vector diversification (cross-pair perturbation reuse DISTINCT from Slot R SYNTHESIZE_FRAME + Slot Q macOS-CPU-advisory).
- (d) WHY HAVEN'T WE PRODUCED ORIGINAL FRONTIER YET: Sister Slot V meta-diagnostic; THIS L0 SCAFFOLD's contribution to (d) is the explicit IMPLEMENTATION_FALSIFIED anchoring + reactivation criteria so future agents do NOT re-discover Wave N+34's verdict via another paid GPU dispatch.
