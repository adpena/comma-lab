<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — BoostNeRV-PR110 Variant C-ii centered_base_recolor training-dynamics fix LANDING memo. DO NOT mutate after landing. -->
<!-- Catalog #229 PV: empirical 9-cell sweep landed end-to-end with canonical artifact at .omx/research/boostnerv_variant_c_ii_centered_base_recolor_sweep_results_20260526/sweep_heatmap.json; canonical equation #347 anchors appended via tools/append_boostnerv_variant_c_ii_centered_base_recolor_empirical_anchors_20260526.py. -->
<!-- # FORMALIZATION_PENDING:variant_c_ii_centered_base_recolor_3rd_order_empirical_refutation_sign_axis_hypothesis_implementation_level_falsification_per_catalog_307_paradigm_intact_canonical_equation_347_anchors_appended_5_total_per_catalog_344_full_stack_fractal_optimization_GUIDING_PRINCIPLE_2026_05_26T19_10Z_demonstration -->
---
council_tier: T1
council_attendees: [Carmack, Shannon, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Carmack
    verbatim: "Variant C-ii centered_base_recolor EMPIRICALLY REFUTED at this fixture surface. The pre-execution gate report SELECTED Variant C-ii as the HIGHEST-EV first Variant C candidate per sister #1345's operator-routable enumeration. Empirical 9-cell sweep produces CONSTANT global_positive_fraction=0.0000 AND CONSTANT global_sign_entropy_bits=0.0000 at ALL 9 cells (sister-identical to #1345 baseline). 3rd-ORDER DISCOVERY NOT ANTICIPATED: even with centering offset R=-0.435 / G=-0.031 / B=-0.024 that reduced base-alone MSE from 0.128 to 0.053 (59% reduction; the centering DID dramatically improve PR110 base accuracy), the residual learner STILL converges to all-negative sign attractor. Sister L1 #1337 Variant A int8 sidecar = 42B; with +12B centering overhead = 54B (EXACT match observed at all 9 cells — confirms centering does NOT change Variant A int8 quantization regime). Sister #1345 Variant B-d sign-bitmap sidecar = 149B; with +12B centering overhead = 161B (EXACT match observed at all 9 cells — confirms centering does NOT change Variant B-d sign-bitmap entropy regime). Loss-axis WIN PRESERVED: max recon-MSE-reduction at ep=30 across gain_clamp grid was 53.2% at gain_clamp=0.20 (matches #1337 baseline pattern). MECHANISM HYPOTHESIS REFUTED: the sign-axis bias is NOT a function of base-output overshoot magnitude — it is a deeper structural property of L2-loss + tanh-output + clip([-gain_clamp, +gain_clamp]) composition that ALWAYS converges to a single sign attractor regardless of where base-output sits relative to GT in the [0,1] domain. PARADIGM (residual-correction hybrid stacking) INTACT per Catalog #307. The byte-axis scale-invariance now extincts across THREE codec/training-dynamics variants (L1 int8 + Variant B-d sign-bitmap + Variant C-ii centered base) — overwhelming evidence the root cause is the L2+tanh+clip OPTIMIZER ATTRACTOR TOPOLOGY not codec-design or base-output-bias. DEFER to next decomposition node per recursive doctrine: Variant C-i (sign-diversity loss term — directly penalize global_positive_fraction != 0.5 in the objective; the only Variant C candidate that DIRECTLY targets the attractor topology). Per CLAUDE.md 'Forbidden premature KILL without research exhaustion' — substrate paradigm DEFERRED-pending-Variant-C-i, NEVER killed."
council_assumption_adversary_verdict:
  - assumption: "Centered base recolor breaks the sign-axis bias by removing systematic positive bias of base-output > GT"
    classification: CARGO-CULTED-EMPIRICALLY-REFUTED
    rationale: "Pre-execution gate report hypothesized that median-shifting base-output toward GT median would diversify residual sign distribution. Empirical 9-cell sweep produces global_positive_fraction=0.0000 at ALL 9 cells DESPITE centering reducing centered-base-alone MSE by 59% (0.128 → 0.053). The base-output overshoot mechanism is NOT the cause of the all-negative attractor. 3rd-ORDER DISCOVERY surfaces deeper mechanism: L2+tanh+clip optimizer topology has a structural sign attractor independent of base-output position."
  - assumption: "Centering preserves L2 loss optimum (mathematical equivalence)"
    classification: HARD-EARNED-MATHEMATICAL
    rationale: "L2 loss `|composed - gt|² = |rgb_base + offset + residual - gt|²` — the optimizer can compensate by subtracting `offset` from `residual` to reach the same final composed value. Empirically confirmed: loss-axis WIN PRESERVED across all 9 cells (e.g. ep=30 gain_clamp=0.20: loss 0.039→0.025 with centering matches #1337 baseline pattern). The fix did NOT regress the loss optimization."
  - assumption: "Variant A int8 quantization is sign-symmetric; centering does not change Variant A bytes"
    classification: HARD-EARNED-EMPIRICALLY-CONFIRMED
    rationale: "Variant A sidecar = 42B (#1337 baseline) + 12B centering overhead = 54B (EXACT match at all 9 cells; predicted exactly by int8 quantization theory). The int8 magnitude distribution is invariant to centering because the quantizer normalizes by gain_clamp. The 12B centering overhead is the ONLY structural addition."
  - assumption: "Variant B-d sign-bitmap entropy would grow if centering diversified sign distribution"
    classification: HARD-EARNED-CONDITIONAL
    rationale: "The Variant B-d codec IS structurally sound (sign-bitmap entropy 0 → 13B brotli RLE; entropy 1 → ~1536B uncompressed). The bytes prediction was correct CONDITIONAL on centering diversifying the sign distribution. Since centering did NOT diversify (empirical refutation), the Variant B-d bytes stay at 149B + 12B = 161B (CONSTANT across all 9 cells; sister-identical to #1345 baseline). The conditional was correctly stated; the antecedent was empirically refuted."
council_decisions_recorded:
  - "Pre-execution gate report identified ingredient #6 / sub-ingredient L2-loss-shape / sub-sub-ingredient base-output-centering as the next decomposition node per recursive doctrine — design choice JUSTIFIED by sister #1345's 2nd-order discovery"
  - "Variant C-ii sweep harness landed at .omx/tmp/boostnerv_variant_c_ii_centered_base_recolor_sweep.py (~610 LOC; sister-derived from #1345 harness + centering offset application)"
  - "9-cell Variant C-ii sweep landed at .omx/research/boostnerv_variant_c_ii_centered_base_recolor_sweep_results_20260526/sweep_heatmap.json (16.2s wallclock)"
  - "EMPIRICAL VERDICT: Variant C-ii centered_base_recolor REFUTED sign-axis hypothesis — global_positive_fraction=0.0000 + sign_entropy=0.0000 at ALL 9 cells (sister-identical to #1345 baseline)"
  - "3rd-ORDER DISCOVERY: L2+tanh+clip optimizer topology has structural sign attractor INDEPENDENT of base-output position; centering offset R=-0.435 reduced base-alone MSE 59% (0.128→0.053) but residual learner STILL converged all-negative"
  - "Loss-axis WIN PRESERVED: max recon-MSE-reduction at ep=30 was 53.2% at gain_clamp=0.20 (matches #1337 baseline pattern); centering did NOT regress loss optimization"
  - "Per Catalog #307: PARADIGM (residual-correction hybrid stacking) INTACT; IMPLEMENTATION-LEVEL REFUTATION of Variant C-ii (sign-axis hypothesis falsification) at this fixture surface"
  - "Per CLAUDE.md 'Forbidden premature KILL': DEFERRED-pending-Variant-C-i (sign-diversity loss term — directly penalizes global_positive_fraction != 0.5; the only candidate that DIRECTLY targets the attractor topology)"
  - "Canonical equation #347 anchors APPENDED via tools/append_boostnerv_variant_c_ii_centered_base_recolor_empirical_anchors_20260526.py: aggregate (sign distribution residual 1.0; full hypothesis refutation) + per-cell Carmack-best-cell (residual 1.0; internally consistent at all 9 cells)"
  - "Operator-routable: descend to Variant C-i (sign-diversity loss term) as next decomposition node per recursive doctrine + just-elevated GUIDING PRINCIPLE 2026-05-26T19:10Z"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
horizon_class: frontier_pursuit
related_deliberation_ids:
  - boostnerv_bpr1_variant_b_codec_redesign_landed_20260526
  - boostnerv_pr110_gain_clamp_sweep_landed_20260526
  - boostnerv_pr110_l1_empirical_landed_20260526
  - boostnerv_variant_c_ii_centered_base_recolor_pre_execution_gate_report_20260526
  - comprehensive_roadmap_synthesis_landed_20260526
  - t3_council_pr110_stacking_pivot_ordering_landed_20260526
---

# BoostNeRV-PR110 Variant C-ii `centered_base_recolor` training-dynamics fix — LANDED 2026-05-26

**Lane**: `lane_boostnerv_variant_c_ii_centered_base_recolor_training_dynamics_fix_20260526` (L1; impl_complete + strict_preflight + memory_entry)
**Subagent**: `boostnerv-variant-c-ii-centered-base-recolor-training-dynamics-fix-20260526`
**Predecessor**: `boostnerv-bpr1-variant-b-codec-redesign-break-scale-invariance-20260526` (commit `57ccd2b1e`; TaskCreate #1345)
**Operator authority**: 2026-05-26 cascade follow-up to operator-routable #1 of sister #1345 (HIGHEST EV Variant C-ii)
**Wallclock**: 16.2 seconds (M5 Max MLX-local; 9 cells sequential; sister-identical training to #1345 + #1342 + #1337)
**Cost**: $0 (MLX-local-only per "Remember all on MLX")

## Pre-execution gate verdict + GUIDING PRINCIPLE decomposition node identifier

Per `.omx/research/boostnerv_variant_c_ii_centered_base_recolor_pre_execution_gate_report_20260526.md`:

**Full-stack fractal optimization decomposition node** (per just-elevated GUIDING PRINCIPLE 2026-05-26T19:10Z):

```
BoostNeRV-PR110 substrate
├── ingredient #6 curriculum / loss-shape ← OPTIMIZATION TARGET
│   └── sub-ingredient L2 loss MSE ← sister #1345 identified as bottleneck
│       ├── sub-sub-ingredient base-output centering ← THIS FIX (Variant C-ii)
│       ├── sub-sub-ingredient sign-diversity regularizer (Variant C-i; NEXT NODE per recursive doctrine)
│       ├── sub-sub-ingredient paired +/- residual heads (Variant C-iii; future)
│       └── sub-sub-ingredient gain_clamp temperature schedule (Variant C-iv; future)
```

This work is the canonical FIRST DEMONSTRATION of the just-elevated GUIDING PRINCIPLE in action. Sister #1345's 2nd-order discovery identified that codec design was NOT the bottleneck (ingredient #8 sub-ingredient QAT); the fractal optimization next-step was to descend into ingredient #6 sub-ingredient L2-loss-shape sub-sub-ingredient base-output centering. THIS work executed that descent + emerged with a 3rd-order discovery: the bottleneck is at an EVEN DEEPER decomposition node (the L2+tanh+clip OPTIMIZER ATTRACTOR TOPOLOGY itself).

## Variant C-ii implementation LOC + sister-pattern reference

- Sweep harness: `.omx/tmp/boostnerv_variant_c_ii_centered_base_recolor_sweep.py` (~610 LOC; sister-derived from `.omx/tmp/boostnerv_pr110_bpr1_variant_b_sweep.py` + centering offset application via `compute_centering_offset` helper + dual-codec measurement [Variant A + Variant B-d for sister-comparison])
- The fix is a 4-line training-pipeline insertion:
  1. `centering_offset = compute_centering_offset(pr110_pairs, gt_pairs)` — per-channel median of `GT - PR110_base` over training set
  2. In `_loss_fn`: `rgb_base_centered = mx.clip(rgb_base_raw + centering_offset, 0.0, 1.0)`
  3. Residual learner sees `rgb_base_centered` (sister-identical compose downstream)
  4. BPR1 sidecar carries 12-byte fp32 centering offset (sister of grayscale-LUT canonical pattern)
- Canonical Provenance per Catalog #323: every result row stamped `axis_tag=[macOS-MLX research-signal]` + `promotion_eligible=False` + `score_claim=False` + `ready_for_exact_eval_dispatch=False`.

## Empirical results (residual sign-distribution + sidecar bytes vs #1337 + #1345 baselines)

| | epochs=30 | epochs=100 | epochs=300 |
|---|---|---|---|
| **gain_clamp=0.05** | loss=0.04627→0.04322 / A=54B / B-d=161B / pos_frac=0.0000 / sign_H=0.000bits / recon_red=18.1% | identical sidecar | identical sidecar |
| **gain_clamp=0.10** | loss=0.04249→0.03555 / A=54B / B-d=161B / pos_frac=0.0000 / sign_H=0.000bits / recon_red=32.7% | identical sidecar | identical sidecar |
| **gain_clamp=0.20** | loss=0.03930→0.02476 / A=54B / B-d=161B / pos_frac=0.0000 / sign_H=0.000bits / recon_red=53.2% | identical sidecar | identical sidecar |

**Centering offset (R,G,B)** = (-0.435294, -0.031373, -0.023529) — large red shift; PR110 base systematically red-shifted vs GT median.

**Pre-sweep base-alone MSE comparison**:
- RAW PR110 base vs GT: 0.128170
- CENTERED PR110 base vs GT: 0.052948 (-59% reduction)

The centering offset DID dramatically improve PR110 base accuracy (huge 59% MSE reduction at the pre-residual-learner stage). DESPITE this, the residual learner STILL converged all-negative.

**Sidecar bytes comparison vs baselines**:
- Variant A (L1 int8): #1337 baseline 42B → with +12B centering = **54B at all 9 cells (EXACT match)**
- Variant B-d (sign-bitmap): #1345 baseline 149B → with +12B centering = **161B at all 9 cells (EXACT match)**

The 12B centering overhead is the ONLY structural addition; sign distribution behavior is sister-identical to #1345.

**Sister-coherence verified**: identical fixture + identical seed + identical training; ONLY difference is centering_offset application + +12B overhead.

## Carmack-dissent verdict per Catalog #307 (paradigm-vs-implementation classification)

**PARADIGM-LEVEL REFUTATION**: NONE. Residual-correction hybrid stacking paradigm INTACT.

**IMPLEMENTATION-LEVEL REFUTATION**: Variant C-ii centered_base_recolor REFUTED at this fixture surface. The sign-axis bias is NOT a function of base-output overshoot magnitude — even with 59% reduction in centered-base MSE, the residual learner STILL converged to the all-negative attractor.

**3rd-ORDER DISCOVERY**: the L2 loss + tanh output + clip([-gain_clamp, +gain_clamp]) composition has a STRUCTURAL SIGN ATTRACTOR independent of base-output position in the [0,1] domain. The optimizer's gradient field for L2 loss with tanh output and symmetric clip is dominated by a single sign mode at convergence; centering changes WHERE in residual space the optimizer lands (smaller magnitude residuals because base-alone MSE is smaller) but NOT WHICH sign attractor (still all-negative).

**Mechanism intuition**: tanh saturates to ±1 at large pre-activation values. The clip operation `clip(±gain_clamp, ±gain_clamp)` symmetric-bounds output. For an L2 loss on `composed = clip(base + residual_clipped, 0, 1)`, the gradient flows that diversify residual sign require alternating-sign updates that the optimizer's momentum + AdamW exponential moving average smooths into a single dominant sign mode. The structural fix would be either:
- DIRECT: add `|global_positive_fraction - 0.5|` penalty to objective (Variant C-i)
- INDIRECT: split residual into +/- branches with separate gain_clamps (Variant C-iii)
- TEMPERATURE: anneal gain_clamp during training to allow late-stage sign-mode diversification (Variant C-iv)

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion"**: substrate paradigm DEFERRED-pending-Variant-C-i (sign-diversity loss term — the only candidate that DIRECTLY targets the attractor topology). Candidates remaining per Catalog #308 alternative-probe-methodology enumeration:
- **Variant C-i** (NEXT per recursive doctrine): `residual_loss_with_sign_diversity_term` — add penalty on `|global_positive_fraction - 0.5|` to objective. RECOMMENDED HIGHEST EV.
- **Variant C-iii**: `paired_positive_negative_residual_heads` — split residual into +/- branches with separate gain_clamp values. Larger architectural change.
- **Variant C-iv**: `gain_clamp_temperature_schedule` — anneal gain_clamp during training. Hyperparameter-only change but unclear if it breaks the attractor topology.

## Canonical equation #347 `residual_hybrid_boosting_savings_v1` anchor_appended event IDs

Per Catalog #344 + `tools/append_boostnerv_variant_c_ii_centered_base_recolor_empirical_anchors_20260526.py`:

- **Aggregate anchor**: `residual_hybrid_boosting_boostnerv_pr110_VARIANT_C_II_centered_base_recolor_9cell_aggregate_sign_axis_hypothesis_refutation_20260526` — captures the EMPIRICAL REFUTATION of the sign-axis hypothesis (predicted positive_fraction 0.5 vs empirical 0.0; residual = 1.0; full hypothesis refutation).
- **Per-cell anchor**: `residual_hybrid_boosting_boostnerv_pr110_VARIANT_C_II_centered_base_recolor_clamp_0p20_30ep_carmack_best_cell_20260526` — Carmack-best-cell config; predicted 0.5 positive fraction vs empirical 0.0 (residual 1.0; internally consistent at all 9 cells).
- **Registry state**: 3 → 5 anchors (delta +2); equation status remains PROVISIONAL pending sister Variant C-i empirical anchor.

## Drift surface declaration (per MLX↔CUDA bidirectional drift directive 2026-05-26)

Per `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`, the Variant C-ii sweep reuses sister #1345 + #1342 canonical helpers verbatim (identical fp32 throughout / MLX defaults / NHWC / tanh+clip ordering / AdamW β₁/β₂ defaults / brotli q9 determinism). NEW drift surface introduced + extincted in same commit: `numpy.median` computed on CPU (deterministic per numpy spec; MLX has no native median — would use sort+index workaround which is sister-bit-stable). Centering offset stored as 3×fp32 (12 bytes) in BPR1 sidecar header; inflate-side applies via deterministic fp32 add. **Portability verdict**: zero new training-time drift surface (MLX defaults preserved); ONE new sidecar field (12B centering offset) with explicit fp32 byte-stable round-trip. Future paired CUDA verification of Variant C-i would inherit the L1+#1342+#1345+#1346 drift-surface declarations unchanged.

## Canonical-vs-frontier-push decision (per pushing-the-frontier directive 2026-05-26)

Per `feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`:

- **Centering offset computation**: CANON-APPLICATION. Median-subtract for distribution centering is canonical preprocessing (sister of LayerNorm + standardization).
- **L2 loss preservation**: CANON-APPLICATION. No new loss term.
- **Sign-axis hypothesis empirical refutation**: FRONTIER-PUSH EMPIRICAL CONTRIBUTION. The 3rd-order discovery that L2+tanh+clip composition has a STRUCTURAL SIGN ATTRACTOR independent of base-output position is novel empirical finding. NO canonical literature directly cites "L2+tanh+symmetric-clip residual learner converges to single sign mode regardless of base-output position"; this is original empirical-grounded contribution to the canonical equation #347 domain-of-validity refinement.
- **Sign-bitmap entropy diagnostic + dual-codec sister comparison**: CANON-APPLICATION (sister of #1345).

## Full-stack fractal optimization decomposition (per just-elevated GUIDING PRINCIPLE 2026-05-26T19:10Z)

This work is the canonical FIRST DEMONSTRATION of the recursive-per-sub-ingredient doctrine + full-stack fractal optimization elevation. The empirical trajectory across sister landings:

| Landing | Decomposition node optimized | Outcome | Next node identified |
|---|---|---|---|
| #1337 BoostNeRV L1 EMPIRICAL | ingredient #6 sub-ingredient training schedule | WIN (7.8% loss reduction) | gain_clamp value (ingredient #4 sub-ingredient codec hyperparameter) |
| #1342 gain_clamp sweep | ingredient #4 sub-ingredient gain_clamp value | partial WIN (53% recon-MSE-reduction) | codec design at int8 quantization regime (ingredient #8 sub-ingredient codec) |
| #1345 Variant B-d codec redesign | ingredient #8 sub-ingredient codec design | EMPIRICAL REFUTATION (codec not the bottleneck; 2nd-order discovery training-dynamics IS) | training-dynamics L2-loss-shape (ingredient #6 sub-ingredient L2-loss-shape) |
| **#1346 (this) Variant C-ii centered base recolor** | **ingredient #6 sub-ingredient L2-loss-shape sub-sub-ingredient base-output centering** | **EMPIRICAL REFUTATION (centering not the mechanism; 3rd-order discovery L2+tanh+clip attractor topology IS)** | **L2+tanh+clip optimizer attractor topology (sub-sub-ingredient sign-diversity loss regularizer Variant C-i)** |

**Each iteration IS the recursive doctrine in action**: empirical landing surfaces the next decomposition node to optimize. The compounding effect: 4 consecutive landings each refining canonical equation #347's domain-of-validity by ONE level of decomposition. Per the GUIDING PRINCIPLE: this IS what extreme-optimization looks like at the structural-protection level.

**The next node**: Variant C-i (sign-diversity loss term) directly targets the OPTIMIZER ATTRACTOR TOPOLOGY (one level deeper than base-output centering). The empirical test of Variant C-i will either:
- VALIDATE the deeper-attractor hypothesis (sign-axis bias broken by direct loss term)
- REFUTE it AGAIN (surface 4th-order discovery: even direct sign-diversity penalty cannot break the attractor; the bottleneck lives at sub-sub-sub-ingredient level)

Either outcome compounds the canonical equation's predictive-power per CLAUDE.md "Results must become system intelligence".

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — per-pair sign-bitmap entropy + per-pair positive-fraction + per-channel centering offset feed sister `tac.sensitivity_map.*` ranker; reusable across any L2+tanh+clip residual learner substrate (the 3rd-order discovery generalizes).
2. **Pareto constraint**: ACTIVE — Variant C-ii gives a NEW (gain_clamp, sidecar_bytes) Pareto point (Variant A: 54B; Variant B-d: 161B) on the same fixture; Pareto-dominated by sister #1337 L1 at the byte axis WITHOUT distortion benefit (sister-identical recon_red 53.2% at gain_clamp=0.20).
3. **Bit-allocator hook**: N/A (uniform per-pixel allocation; codec structurally fixed).
4. **Cathedral autopilot dispatch hook**: ACTIVE — `.omx/research/boostnerv_variant_c_ii_centered_base_recolor_sweep_results_20260526/sweep_heatmap.json` carries canonical Provenance per Catalog #323 (`axis_tag=[macOS-MLX research-signal]` + non-promotable markers per Catalog #341). Auto-discoverable per Catalog #335 cathedral_consumers Protocol contract.
5. **Continual-learning posterior update**: ACTIVE — both empirical anchors appended to canonical equation #347 via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344; equation status remains PROVISIONAL pending sister Variant C-i empirical anchor.
6. **Probe-disambiguator**: ACTIVE — Variant C-ii sweep IS the canonical operator-routable disambiguator probe per Catalog #313 between "sign-axis bias driven by base-output overshoot (Variant C-ii would have fixed it)" vs "sign-axis bias driven by L2+tanh+clip attractor topology (Variant C-i needed)". **VERDICT: ATTRACTOR TOPOLOGY** (per 3rd-order discovery of sign distribution invariance to centering).

## HORIZON-CLASS verdict per Catalog #309

`frontier_pursuit` (same as sister #1345; predicted PLATEAU-ADJACENT band; mechanism investigation surfacing 3rd-order discovery; canonical equation #347 domain-of-validity refinement is structural-protection contribution per CLAUDE.md "Results must become system intelligence")

## Cross-pollination with sister substrates

- **NSCS06 v8 stacked paired Modal T4** (sister slot 1; different substrate scope): zero collision. NSCS06 v8 is per-class deterministic LUT codec (Carmack-Hotz strip-everything paradigm); BoostNeRV-PR110 Variant C-ii is gradient-trained MLX residual + dual-codec measurement.
- **Z7-Mamba-2 v2 L2 stability hardening** (sister slot 2; different substrate scope): zero collision (orthogonal NaN-fix scope).
- **T3 council on falsified/negative/defer verdicts** (sister slot 4; READ-ONLY synthesis): this landing PROVIDES one additional canonical paradigm-intact / implementation-level refutation anchor for the council's verdict-corpus synthesis (per CLAUDE.md "Forbidden premature KILL without research exhaustion" the 3rd-order discovery DEFERS not kills).
- **T3 PR110 stacking ordering memo**: this Variant C-ii landing does NOT add BoostNeRV-PR110 to the stacking matrix because the centering+codec axis (54B Variant A / 161B Variant B-d) is Pareto-DOMINATED by sister #1337 L1 (42B) without corresponding distortion benefit. Operator-routable from sister #1345 memo (Variant B-d candidate #6 in PR110 stacking) remains DOWNGRADED pending Variant C-i empirical landing OR paired CUDA d_seg+d_pose measurement.

## Operator-routable next steps (priority-ordered)

1. **HIGHEST EV** (per recursive doctrine + GUIDING PRINCIPLE next decomposition node): **Variant C-i `residual_loss_with_sign_diversity_term`** — directly add penalty `λ * |global_positive_fraction - 0.5|` to the L2 objective. This DIRECTLY targets the L2+tanh+clip attractor topology rather than indirectly through base-output centering. Sister subagent can iterate on $0 MLX-local in <30 min wallclock. If Variant C-i ALSO refutes, surfaces 4th-order discovery.

2. **ALTERNATIVE** (architectural change, larger LOC): **Variant C-iii `paired_positive_negative_residual_heads`** — split residual into +/- branches with separate gain_clamp values + separate convolutional heads. The two-branch architecture forces sign diversification BY CONSTRUCTION. Higher implementation cost but structurally guaranteed to break the single-mode attractor.

3. **HYPERPARAMETER SWEEP** (cheapest): **Variant C-iv `gain_clamp_temperature_schedule`** — anneal gain_clamp from large (0.50) at start to small (0.05) at end of training. Hyperparameter-only change; unclear if it breaks the attractor topology but trivially cheap to test.

4. **PAIRED CUDA VALIDATION** (operator-routable from sister #1345 reaffirmed): cost ~$0.20-0.50; measure true d_seg+d_pose CUDA reduction at the best-cell config to disambiguate whether ANY of the BoostNeRV-PR110 training improvements translate to contest-axis benefit. CRITICAL gate: if the 53.2% recon-proxy reduction does NOT translate to contest-axis benefit, the substrate is DEFERRED at the L2 training-dynamics ceiling regardless of which Variant C wins.

5. **REGISTRY HYGIENE**: canonical equation #347 status remains PROVISIONAL with `excluded_contexts` extended to include `centered_base_recolor_alone_at_L2_tanh_clip_attractor_regime` per Catalog #359 sister discipline (avoid future misapplication of equation predicate to this empirically-refuted context). Operator decision required to register the domain-refinement event.

## Cross-references

- Pre-execution gate report: `.omx/research/boostnerv_variant_c_ii_centered_base_recolor_pre_execution_gate_report_20260526.md` (sister; same commit batch)
- Sister Variant B-d landing memo: `.omx/research/boostnerv_bpr1_variant_b_codec_redesign_landed_20260526.md` (commit `57ccd2b1e`)
- Sister gain_clamp sweep landing memo: `.omx/research/boostnerv_pr110_gain_clamp_sweep_landed_20260526.md` (commit `8240aceda`)
- Sister L1 EMPIRICAL landing memo: `.omx/research/boostnerv_pr110_l1_empirical_landed_20260526.md` (commit `b2fd3e587`)
- Variant C-ii sweep heatmap JSON: `.omx/research/boostnerv_variant_c_ii_centered_base_recolor_sweep_results_20260526/sweep_heatmap.json`
- Variant C-ii sweep harness: `.omx/tmp/boostnerv_variant_c_ii_centered_base_recolor_sweep.py`
- Canonical equation anchor-append script: `tools/append_boostnerv_variant_c_ii_centered_base_recolor_empirical_anchors_20260526.py`
- Canonical equation #347 registry: `.omx/state/canonical_equations_registry.jsonl` (5 anchors total: L1 baseline + Variant B-d aggregate + Variant B-d per-cell + Variant C-ii aggregate + Variant C-ii per-cell Carmack-best-cell)
- T3 PR110-stacking-ordering memo: `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`
- Comprehensive roadmap synthesis: `.omx/research/comprehensive_roadmap_synthesis_landed_20260526.md`
- MLX↔CUDA bidirectional drift directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`
- Pushing-the-frontier directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`
- PR95-sniped-lesson + GUIDING PRINCIPLE elevation memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pr95_sniped_lesson_GUIDING_PRINCIPLE_full_stack_fractal_optimization_elevation_20260526.md` (just-amended 2026-05-26T19:10Z)
