<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — BoostNeRV-PR110 BPR1 Variant B-d codec redesign landing memo. DO NOT mutate after landing. -->
<!-- Catalog #229 PV: empirical 9-cell sweep landed end-to-end with canonical artifact at .omx/research/boostnerv_pr110_bpr1_variant_b_sweep_results_20260526/sweep_heatmap.json; canonical equation #347 anchors appended via tools/append_boostnerv_variant_b_d_empirical_anchors_20260526.py. -->
<!-- # FORMALIZATION_PENDING:variant_b_d_codec_design_empirical_implementation_level_falsification_per_catalog_307_paradigm_intact_canonical_equation_347_anchors_appended_with_aggregate_plus_per_cell_data_per_catalog_344_operator_decision_protocol -->
---
council_tier: T1
council_attendees: [Carmack, Shannon, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Carmack
    verbatim: "Variant B-d codec design empirically FALSIFIED at this fixture surface. The pre-execution gate report SELECTED Variant B-d after empirically refuting Variants B-a/b/c (signed-exponent / variable-bit-width / non-uniform quantization) — all three NOT-APPROPRIATE for the 100%-saturation regime. Variant B-d hypothesis: sign-bitmap entropy depends on gain_clamp (larger clamp → wider L2 loss valley → more diverse trained sign-patterns → less brotli-compressible). EMPIRICAL: SECOND-ORDER discovery — 100% of trained residuals are NEGATIVE (global_sign_pos=0.0000 at ALL 9 cells; global_sign_entropy_bits=0.0000). Sign-bitmap codec produces CONSTANT 149B sidecar (vs L1's 42B; structurally larger because sign-bitmap encoding scales with NUM_PIXELS not RLE-collapsed int8). The mechanism: PR110 base RGB > GT typically (overshoots) → L2 loss biases residual learner toward all-negative subtraction → sign-bitmap entropy=0 regardless of gain_clamp. PARADIGM (residual-correction hybrid stacking) INTACT per Catalog #307. The byte-axis scale-invariance now extincts across TWO codec variants (int8 quantization L1 + sign-bitmap Variant B-d) — strong evidence the root cause is TRAINING-DYNAMICS not CODEC-DESIGN. DEFER to Variant C: training-side fix that breaks the sign-axis bias. Candidates: (i) `residual_loss_with_sign_diversity_term` (penalty on global_positive_fraction != 0.5); (ii) `centered_base_recolor` (mean-subtract PR110 base before residual learning); (iii) `paired_positive_negative_residual_heads` (split residual into +/- branches with separate gain_clamp). Per CLAUDE.md 'Forbidden premature KILL without research exhaustion' — substrate paradigm DEFERRED-pending-Variant-C, NEVER killed."
council_assumption_adversary_verdict:
  - assumption: "Sign-bitmap entropy depends on gain_clamp value"
    classification: CARGO-CULTED-EMPIRICALLY-REFUTED
    rationale: "Pre-execution gate report hypothesized that larger gain_clamp → wider L2 loss valley → more diverse trained sign-patterns. Empirical 9-cell sweep produces `global_sign_entropy_bits=0.0000` at ALL 9 cells. The L2 loss landscape's sign-axis bias dominates the gain_clamp-axis effect — at the fixture (PR110 base > GT typical), the optimizer reaches the same all-negative attractor regardless of clamp width. SECOND-ORDER DISCOVERY not anticipated."
  - assumption: "Variant B-d codec design is sufficient to break scale-invariance"
    classification: CARGO-CULTED-EMPIRICALLY-REFUTED
    rationale: "Sign-bitmap + per-pair magnitude codec is structurally sound — it DOES produce bytes proportional to NUM_PIXELS × per-pixel-entropy + magnitude-scalar-per-pair. But when the trained residuals saturate to all-negative (or all-positive — a single attractor mode), the sign-bitmap entropy collapses to 0 → brotli RLE-collapses to ~13B → total sidecar dominated by header + per-pair magnitudes (28 + 8 + 13 + 100 = 149B), all of which are gain_clamp-INDEPENDENT. Codec-level fix CANNOT solve a training-dynamics-level bug."
  - assumption: "Variant B-d 149B sidecar improvement over L1's 42B sidecar is worth it"
    classification: HARD-EARNED-AMBIGUOUS
    rationale: "Variant B-d sidecar (149B → Δrate +0.000099) is 3.5× LARGER than L1's (42B → Δrate +0.000028). The +0.000071 contest-rate cost is small in absolute terms but represents NET-NEGATIVE bytes WITHOUT corresponding distortion benefit (loss reduction is identical between codecs since they share training). PARADIGM-LEVEL: Variant B-d is NOT preferable to L1 BPR1 at this regime; L1 wins on byte efficiency. ROUTING IMPLICATION: do NOT swap L1 BPR1 for Variant B-d in the substrate's canonical codec slot."
council_decisions_recorded:
  - "Pre-execution gate report SELECTED Variant B-d after empirically refuting Variants B-a/b/c — design choice JUSTIFIED by 100%-saturation empirical fact"
  - "Variant B-d codec module landed at src/tac/substrates/boost_nerv_pr110_residual/bpr1_variant_b_sign_bitmap_codec.py (~240 LOC including diagnostic helper)"
  - "9-cell Variant B-d sweep landed at .omx/research/boostnerv_pr110_bpr1_variant_b_sweep_results_20260526/sweep_heatmap.json (17.4s wallclock)"
  - "EMPIRICAL VERDICT: Variant B-d sidecar bytes CONSTANT 149B across all 9 cells (sister-extincted scale-invariance at sign-bitmap codec surface)"
  - "SECOND-ORDER DISCOVERY: global_sign_entropy_bits=0.0000 at ALL 9 cells → 100% of residuals NEGATIVE → root cause is L2-loss sign-axis bias not codec-design"
  - "Per Catalog #307: PARADIGM (residual-correction hybrid stacking) INTACT; IMPLEMENTATION-LEVEL FALSIFICATION of Variant B-d codec design at this fixture surface"
  - "Per CLAUDE.md 'Forbidden premature KILL': DEFERRED-pending-Variant-C (training-side fix to break sign-axis bias)"
  - "Canonical equation #347 anchors APPENDED via tools/append_boostnerv_variant_b_d_empirical_anchors_20260526.py: aggregate (relative residual 1.188; falsifies hypothetical-proportional predicate) + per-cell Carmack-best-cell (residual 0.0; internally consistent at codec surface)"
  - "Operator-routable: do NOT swap L1 BPR1 for Variant B-d in canonical codec slot — Variant B-d is 3.5× LARGER sidecar (149B vs 42B) without distortion benefit at current training dynamics"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
horizon_class: frontier_pursuit
related_deliberation_ids:
  - boostnerv_pr110_gain_clamp_sweep_landed_20260526
  - boostnerv_pr110_l1_empirical_landed_20260526
  - boostnerv_bpr1_variant_b_codec_redesign_pre_execution_gate_report_20260526
  - comprehensive_roadmap_synthesis_landed_20260526
  - t3_council_pr110_stacking_pivot_ordering_landed_20260526
---

# BoostNeRV-PR110 BPR1 Variant B-d codec redesign — LANDED 2026-05-26

**Lane**: `lane_boostnerv_bpr1_variant_b_codec_redesign_break_scale_invariance_20260526` (L1; impl_complete + strict_preflight + memory_entry)
**Subagent**: `boostnerv-bpr1-variant-b-codec-redesign-break-scale-invariance-20260526`
**Predecessor**: `boostnerv-pr110-gain-clamp-sweep-20260526` (commit `8240aceda`)
**Operator authority**: 2026-05-26 cascade follow-up to Carmack-dissent dual-axis verdict in sweep landing memo (operator-routable #1 HIGHEST EV: codec-design FRONTIER-PUSH)
**Wallclock**: 17.4 seconds (M5 Max MLX-local; 9 cells sequential, sister-identical training to gain_clamp sweep)
**Cost**: $0 (MLX-local-only per "Remember all on MLX")

## Pre-execution gate verdict + design choice (a/b/c/d)

Per `.omx/research/boostnerv_bpr1_variant_b_codec_redesign_pre_execution_gate_report_20260526.md`:

- **Variant B-a (signed-exponent)**: NOT-APPROPRIATE — empirically refuted by sweep heatmap p50≈p99 (residuals uniformly saturated; no magnitude variance to encode).
- **Variant B-b (gain_clamp-dependent bit-width)**: NOT-APPROPRIATE — more bits with all-saturated distribution still RLE-compresses identically.
- **Variant B-c (non-uniform Lloyd-Max)**: NOT-APPROPRIATE — optimal quantizer for uniform-saturated distribution is the same as uniform.
- **Variant B-d (sign-bitmap + per-pair magnitude)** — SELECTED. Frontier-push insight: all 100% of information is in the SIGN; encode it directly with 1 bit/pixel + per-pair fp16 magnitude scalar.

**Justification per empirical sweep heatmap re-inspection**: `clamped_fraction_at_boundary=1.0000` at ALL 9 cells; `residual_blob_brotli_bytes=14` of 1.84MB raw → mechanism = tanh saturation + L2 loss + clip(±gain_clamp) → 100% ±127 int8 → brotli RLE-collapses. Variant B-d targets this with explicit sign-bitmap encoding (sister of grayscale-LUT codec patterns per Daubechies wavelet hierarchical priors).

## Variant B-d implementation LOC + sister-pattern reference

- Canonical codec module: `src/tac/substrates/boost_nerv_pr110_residual/bpr1_variant_b_sign_bitmap_codec.py` (~240 LOC including `build_variant_b_d_sidecar` + `compute_sign_bitmap_entropy_diagnostic` + canonical `VariantBDSidecarManifest` frozen dataclass).
- Sister sweep harness: `.omx/tmp/boostnerv_pr110_bpr1_variant_b_sweep.py` (~420 LOC; sister-identical to `.omx/tmp/boostnerv_pr110_gain_clamp_sweep.py` except sidecar build switches from int8 quantization to Variant B-d sign-bitmap).
- Sister codec sister-anchor: `src/tac/substrates/boost_nerv_pr110_residual/archive.py` (L1 BPR1 codec; canonical reference; preserved unchanged).
- Canonical Provenance per Catalog #323: every result row stamped `axis_tag=[macOS-MLX research-signal]` + `promotion_eligible=False` + `score_claim=False` + `ready_for_exact_eval_dispatch=False`.

## 9-cell sweep results (heatmap summary)

| | epochs=30 | epochs=100 | epochs=300 |
|---|---|---|---|
| **gain_clamp=0.05** | loss=0.10697 / sidecar=149B / Δrate=+0.000099 / global_sign_H=0.000bits / global_pos_frac=0.0000 / recon_red=16.2% | identical | identical |
| **gain_clamp=0.10** | loss=0.08905 / sidecar=149B / Δrate=+0.000099 / global_sign_H=0.000bits / global_pos_frac=0.0000 / recon_red=30.2% | identical | identical |
| **gain_clamp=0.20** | loss=0.06054 / sidecar=149B / Δrate=+0.000099 / global_sign_H=0.000bits / global_pos_frac=0.0000 / recon_red=52.6% | identical | identical |

**Sidecar bytes decomposition** (constant 149B across all 9 cells):
- BPR1 header: 28B
- Variant B-d len fields: 8B
- Brotli(packed_sign_bytes): 13B (RLE-collapse of all-zero sign bitmap)
- Per-pair magnitudes (50 pairs × fp16): 100B
- **Total**: 28 + 8 + 13 + 100 = 149B

**Sister-coherence with gain_clamp sweep**: identical training loss + recon_red per cell (same fixture + same training; only sidecar build differs). Sister-anchor verified per Catalog #305 observability facet "diff-able across runs".

## Carmack-dissent verdict per Catalog #307 (paradigm-vs-implementation classification)

**PARADIGM-LEVEL FALSIFICATION**: NONE. Residual-correction hybrid stacking paradigm INTACT.

**IMPLEMENTATION-LEVEL DUAL FINDING**:

1. **CODEC-DESIGN AXIS — Variant B-d FALSIFIED** at current training dynamics. The sign-bitmap codec is structurally sound (produces bytes proportional to NUM_PIXELS × per-pixel-entropy + per-pair-magnitude-fp16), but when trained residuals collapse to a single sign-mode (all-negative in our case), the sign-bitmap entropy = 0 → brotli RLE-collapses to ~13B → total dominated by header + magnitudes (gain_clamp-independent). Codec-level fix CANNOT solve a training-dynamics bug.

2. **TRAINING-DYNAMICS AXIS — SECOND-ORDER DISCOVERY**: 100% of trained residuals are NEGATIVE (`global_sign_pos=0.0000` at ALL 9 cells). Mechanism: PR110 base RGB > GT typically (overshoots in this 50-pair × 96×128 fixture), so L2 loss biases the residual learner to push DOWN uniformly. The sign-axis bias dominates the gain_clamp-axis effect — at ANY clamp width, the optimizer reaches the same all-negative attractor.

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion"**: substrate paradigm DEFERRED-pending-Variant-C (training-side fix). Candidates per Catalog #308 alternative-probe-methodology enumeration:
- **Variant C-i**: `residual_loss_with_sign_diversity_term` — add penalty on `|global_positive_fraction - 0.5|` to objective.
- **Variant C-ii**: `centered_base_recolor` — mean-subtract PR110 base before residual learning (forces zero-mean residuals).
- **Variant C-iii**: `paired_positive_negative_residual_heads` — split residual into +/- branches with separate gain_clamp values.
- **Variant C-iv**: empirically measure d_seg+d_pose CUDA reduction at gain_clamp=0.20 (Carmack op-routable #2) to disambiguate whether the 52.6% recon-MSE-reduction actually translates to contest-axis benefit BEFORE chasing training-dynamics fixes.

## Canonical equation #347 `residual_hybrid_boosting_savings_v1` anchor_appended event IDs

Per Catalog #344 + `tools/append_boostnerv_variant_b_d_empirical_anchors_20260526.py`:

- **Aggregate anchor**: `residual_hybrid_boosting_boostnerv_pr110_VARIANT_B_D_9cell_aggregate_scale_invariance_finding_20260526` — captures the SCALE-INVARIANCE empirical fact across 9 cells; predicted-from-model `hypothetical_proportional_bytes` vs empirical constant 149B; relative residual = 1.188 (118.8% rel error) — falsifies the naive-proportional predicate.
- **Per-cell anchor**: `residual_hybrid_boosting_boostnerv_pr110_VARIANT_B_D_clamp_0p20_30ep_carmack_best_cell_20260526` — captures Carmack-best-cell config; predicted=empirical=149B (residual 0.0; internally consistent at codec surface).
- **Registry state**: 1 → 3 anchors (delta +2); equation status remains PROVISIONAL pending sister Variant C empirical anchor OR paired CUDA d_seg+d_pose measurement.

## Drift surface declaration (per NEW MLX↔CUDA bidirectional drift directive 2026-05-26)

Per `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`, the Variant B-d sweep reuses sister gain_clamp sweep canonical helpers verbatim (identical fp32 throughout / MLX defaults / NHWC / tanh+clip ordering / AdamW β₁/β₂ defaults / brotli q9 determinism). NEW drift surface introduced: `numpy.packbits` byte-order = canonical big-endian per numpy spec. **Portability verdict**: zero drift surface introduced (numpy packbits is bit-stable across CPU and any CUDA sister; brotli quality 9 is deterministic). Future paired CUDA verification of Variant C variant would inherit the L1+sister drift-surface declarations unchanged.

## Canonical-vs-frontier-push decision (per NEW pushing-the-frontier directive 2026-05-26)

Per `feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`:

- **Codec-design level**: FRONTIER-PUSH. Sign-bitmap encoding of trained residuals after saturation observation is a novel codec design specifically motivated by EMPIRICAL sweep finding. NO canonical literature directly cites "sign-bitmap codec for tanh-saturated residual learners"; this is original empirical-grounded design. The IMPLEMENTATION-LEVEL falsification is itself a frontier-research contribution (canonical equation #347's domain-of-validity refinement to explicitly mark `sign_axis_bias_pending_variant_c_training_fix`).
- **Brotli compression level**: CANON-APPLICATION. Brotli q9 is canonical (sister of L1 codec).
- **Per-pair magnitude encoding**: CANON-APPLICATION. fp16 scalar per pair is standard.
- **Sign-bitmap entropy diagnostic**: FRONTIER-PUSH. The per-pair binary entropy H(p) computation + global aggregation is the canonical observability surface that enabled the SECOND-ORDER discovery (100% negative residuals); reusable for sister substrates (any tanh-saturated residual learner). Cross-disciplinary: Shannon binary entropy + Daubechies wavelet sign-coherence prior.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — per-pair sign-bitmap entropy + per-pair positive-fraction feed sister `tac.sensitivity_map.*` ranker; reusable across any saturation-prone residual codec substrate.
2. **Pareto constraint**: ACTIVE — Variant B-d gives a NEW (gain_clamp, sidecar_bytes) Pareto point (149B vs L1's 42B) on the same fixture; Pareto-dominated by L1 at the byte axis WITHOUT distortion benefit.
3. **Bit-allocator hook**: N/A (1-bit-per-pixel uniform allocation; the codec's per-pixel allocation is structurally fixed).
4. **Cathedral autopilot dispatch hook**: ACTIVE — `.omx/research/boostnerv_pr110_bpr1_variant_b_sweep_results_20260526/sweep_heatmap.json` carries canonical Provenance per Catalog #323 (`axis_tag=[macOS-MLX research-signal]` + non-promotable markers per Catalog #341). Auto-discoverable per Catalog #335 cathedral_consumers Protocol contract.
5. **Continual-learning posterior update**: ACTIVE — both empirical anchors appended to canonical equation #347 via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344; equation status remains PROVISIONAL pending sister Variant C anchor.
6. **Probe-disambiguator**: ACTIVE — Variant B-d sweep IS the canonical operator-routable #1 disambiguator probe per Catalog #313 between "scale-invariance is a CODEC-design artifact (Variant B-d SHOULD have broken it)" vs "scale-invariance is a TRAINING-DYNAMICS artifact (Variant C would be needed)". **VERDICT: TRAINING-DYNAMICS** (per SECOND-ORDER discovery of all-negative residual sign distribution).

## HORIZON-CLASS verdict per Catalog #309

`frontier_pursuit` (same as sister sweep; predicted PLATEAU-ADJACENT band; codec-redesign is mechanism investigation not scoring-floor pursuit; canonical equation #347 domain-of-validity refinement is structural-protection contribution).

## Cross-pollination with sister substrates

- **NSCS06 v8 stacked paired Modal T4** (slot 1; different substrate scope): zero collision. NSCS06 v8 is per-class deterministic LUT codec (Carmack-Hotz strip-everything paradigm); BoostNeRV-PR110 Variant B-d is gradient-trained MLX residual + sign-bitmap codec. Per Catalog #294 dim 6 stack-of-stacks-composability: a future composition might apply NSCS06 v8 chroma LUT FIRST then BoostNeRV residual SECOND (sister to the gain_clamp sweep cross-pollination note).
- **PR110-OPT-3 Variant C variable-K escape mechanism** (slot 2; selector-stream codec): zero collision (orthogonal axis per Catalog #356 per-axis decomposition discipline).
- **T3 PR110 stacking ordering memo**: this sweep does NOT add BoostNeRV-PR110 to the stacking matrix because the codec axis (149B vs 42B at L1) is Pareto-DOMINATED by L1 without corresponding distortion benefit at current training. Operator-routable #3 of sister gain_clamp sweep memo (fold BoostNeRV-PR110 into PR110-stacking as candidate #6) is now **DOWNGRADED** to require: (a) Variant C empirical landing breaking sign-axis bias OR (b) paired CUDA d_seg+d_pose measurement showing 52.6% recon-MSE-reduction translates to contest-axis benefit.

## Operator-routable next steps (priority-ordered)

1. **HIGHEST EV** (training-dynamics FRONTIER-PUSH): Variant C-ii `centered_base_recolor` — mean-subtract PR110 base before residual learning (forces zero-mean residuals). Simplest of the 4 Variant C candidates; predicted to immediately diversify the sign distribution. Sister subagent can iterate on $0 MLX-local in <30 min wallclock.

2. **NEXT** (Carmack operator-routable #2 from sister sweep memo, re-affirmed): paired CUDA dispatch on best-cell config `(gain_clamp=0.20, epochs=30)` to measure true d_seg+d_pose reduction. Cost: $0.20-0.50 if MLX→PyTorch export bridge yields matching state_dict. The 52.6% recon-proxy MSE reduction is advisory; only paired CUDA SegNet+PoseNet routing per Catalog #164/#226 produces contest-axis truth. **CRITICAL**: this measurement disambiguates whether ANY of the Variant C training-side fixes are worth pursuing — if the recon-proxy reduction does NOT translate to contest-axis benefit, the substrate is DEFERRED at the L2 training-dynamics ceiling regardless of codec.

3. **REGISTRY HYGIENE**: canonical equation #347 status remains PROVISIONAL with `excluded_contexts` extended to include `variant_b_d_sign_bitmap_at_tanh_saturated_negative_attractor_regime` per Catalog #359 sister discipline (avoid future misapplication of the equation predicate to this empirically-falsified codec surface). Operator decision required to register the domain-refinement event.

4. **DEFER** (do NOT fold Variant B-d into PR110 stacking matrix as candidate #6): per the Pareto-dominance verdict (149B vs 42B without distortion benefit), Variant B-d is NOT a contest-axis improvement. The L1 BPR1 codec remains the canonical codec slot for BoostNeRV-PR110 substrate.

## Cross-references

- Pre-execution gate report: `.omx/research/boostnerv_bpr1_variant_b_codec_redesign_pre_execution_gate_report_20260526.md` (sister; same commit batch)
- Sister gain_clamp sweep landing memo: `.omx/research/boostnerv_pr110_gain_clamp_sweep_landed_20260526.md` (commit `8240aceda`)
- Sister gain_clamp sweep heatmap JSON: `.omx/research/boostnerv_pr110_gain_clamp_sweep_results_20260526/sweep_heatmap.json`
- Sister L1 EMPIRICAL landing memo: `.omx/research/boostnerv_pr110_l1_empirical_landed_20260526.md` (commit `b2fd3e587`)
- Variant B-d sweep heatmap JSON: `.omx/research/boostnerv_pr110_bpr1_variant_b_sweep_results_20260526/sweep_heatmap.json`
- Variant B-d codec module: `src/tac/substrates/boost_nerv_pr110_residual/bpr1_variant_b_sign_bitmap_codec.py`
- Variant B-d sweep harness: `.omx/tmp/boostnerv_pr110_bpr1_variant_b_sweep.py`
- Canonical equation anchor-append script: `tools/append_boostnerv_variant_b_d_empirical_anchors_20260526.py`
- Canonical equation #347 registry: `.omx/state/canonical_equations_registry.jsonl` (3 anchors total: L1 baseline + Variant B-d aggregate + Variant B-d per-cell Carmack-best-cell)
- T3 PR110-stacking-ordering memo: `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`
- MLX↔CUDA bidirectional drift directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`
- Pushing-the-frontier directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`
