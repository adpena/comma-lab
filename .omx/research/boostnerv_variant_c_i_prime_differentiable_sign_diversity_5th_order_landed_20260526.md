<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — BoostNeRV Variant C-i' 5TH-ORDER landing memo. DO NOT mutate after landing. -->
<!-- # FORMALIZATION_PENDING:variant_c_i_prime_5th_order_differentiable_tanh_proxy_SOFT_VALIDATED_HARD_PARTIAL_byte_axis_failed_4TH_order_mechanism_diagnosis_retroactively_validated_at_gradient_flow_surface_but_NEW_5TH_ORDER_DISCOVERY_residual_magnitude_explosion_canonical_equation_347_9_anchors_per_catalog_344_GUIDING_PRINCIPLE_5TH_ORDER_DEMONSTRATION_2026_05_26 -->
---
council_tier: T1
council_attendees: [Carmack, Shannon, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Carmack
    verbatim: "Variant C-i' DIFFERENTIABLE tanh-proxy SOFT-VALIDATED at the gradient-flow + sign-distribution surfaces (positive_fraction_soft -> 0.5000 at ALL 6/6 cells; hard-indicator positive_fraction moved from C-i baseline 0.0000 -> range [0.33, 0.66] across cells; sign-bitmap entropy grew from 0.0000 -> 0.92-1.00 bits) BUT BYTE-AXIS FAILED (Variant B-d sidecar exploded 149B -> [166B, 70784B]; Variant A 42B -> ~1.5MB). The 4TH-order mechanism diagnosis (non-differentiable indicator IS the gradient-flow bottleneck) is RETROACTIVELY VALIDATED. NEW 5TH-ORDER DISCOVERY: differentiable penalty BYPASSED the gradient bottleneck BUT triggered residual MAGNITUDE EXPLOSION because soft sign-balance is achievable at ANY magnitude — the L2 term then drove residuals to clip saturation. The clip operation normalizes sign distribution at the boundary but FAILS TO BOUND magnitude distribution in a way brotli can compress. This is a COUPLED-OBJECTIVE problem the L2+sign-penalty+clip topology cannot solve alone. Per Catalog #307: PARADIGM intact (residual-correction hybrid stacking); IMPLEMENTATION-LEVEL 5TH-ORDER refinement needed: add magnitude-regularizer (Variant C-i'') OR move to per-pixel sign-diversity (Variant C-vii) OR architectural paired heads (Variant C-iii). Per CLAUDE.md 'Forbidden premature KILL': substrate paradigm DEFERRED-pending-6TH-ORDER decomposition."
council_assumption_adversary_verdict:
  - assumption: "Differentiable tanh-proxy will provide chain-rule gradient back to residual parameters and break the L2+tanh+clip attractor"
    classification: HARD-EARNED-EMPIRICALLY-CONFIRMED-AT-GRADIENT-FLOW-AND-SIGN-DISTRIBUTION-SURFACES
    rationale: "Empirical 6-cell sweep produces positive_fraction_soft within +/-0.10 of 0.5 at ALL 6/6 cells AND sign-bitmap entropy > 0.2 bits at ALL 6/6 cells. The gradient k*(1 - tanh^2(k*r))/2 flowed back to residual parameters AS PREDICTED. The 4TH-order mechanism diagnosis (non-differentiable indicator was the bottleneck) is RETROACTIVELY VALIDATED at the gradient-flow surface."
  - assumption: "Variant B-d sidecar bytes will shrink below 157B baseline at >=1 cell once sign distribution diversifies"
    classification: CARGO-CULTED-EMPIRICALLY-REFUTED-NEW-5TH-ORDER-DISCOVERY
    rationale: "Pre-execution gate report assumed sign-diversification would propagate to byte-axis savings. Empirical 6-cell sweep produces Variant B-d sidecar bytes [166B, 70784B] (range 1.06x - 451x baseline) and Variant A 126066B - 1517079B (3000x - 36000x baseline). NEW 5TH-ORDER DISCOVERY: the differentiable penalty bypassed the gradient bottleneck (success) BUT triggered residual MAGNITUDE EXPLOSION (new failure mode). Mechanism: soft sign-balance via (tanh(k*r)+1)/2 -> 0.5 is achievable at ANY MAGNITUDE because tanh saturates symmetrically; the L2-MSE term then drove residuals to clip saturation to reduce composed-frame error, producing high-magnitude residuals that brotli cannot compress. This is a COUPLED-OBJECTIVE problem (sign-balance AND magnitude-bounding) the L2+single-penalty+clip composition cannot solve alone."
  - assumption: "Loss-axis WIN is preserved across all 6 cells"
    classification: HARD-EARNED-EMPIRICALLY-CONFIRMED
    rationale: "Max recon-MSE-reduction = 38.9% at gain_clamp=0.20 k=1.0; range across cells [7.7%, 38.9%]. L2 MSE component continues to reduce in all cells (initial->final). The L2 fit was preserved across all 6 cells. The trade-off is L2-fit-vs-magnitude-bounding the optimizer makes when given freedom by the soft proxy."
council_decisions_recorded:
  - "Pre-execution gate report identified 5TH-ORDER sub-sub-sub-ingredient differentiable-sign-proxy-via-tanh as the canonical fix"
  - "Variant C-i' sweep harness landed at .omx/tmp/boostnerv_variant_c_i_prime_differentiable_sign_diversity_sweep.py (~715 LOC; sister-derived from #1351 harness + tanh-proxy replacement)"
  - "6-cell Variant C-i' sweep landed at .omx/research/boostnerv_variant_c_i_prime_differentiable_sign_diversity_sweep_results_20260526/sweep_heatmap.json (4.3s wallclock)"
  - "EMPIRICAL VERDICT: Variant C-i' SOFT-VALIDATED at gradient-flow + sign-distribution surfaces (positive_fraction_soft -> 0.5 at 6/6 cells; sign entropy 0.92-1.00 bits at 6/6 cells) BUT BYTE-AXIS FAILED (V3 criterion 0/6 cells below 157B baseline)"
  - "4TH-order mechanism diagnosis (non-differentiable indicator IS the gradient-flow bottleneck) RETROACTIVELY VALIDATED at gradient-flow surface"
  - "NEW 5TH-ORDER DISCOVERY: differentiable penalty bypassed gradient bottleneck BUT triggered residual magnitude EXPLOSION because soft sign-balance achievable at any magnitude + L2 drives residuals to clip saturation"
  - "Loss-axis WIN PRESERVED: max recon-MSE-reduction = 38.9% at gain_clamp=0.20 k=1.0; range [7.7%, 38.9%] across cells"
  - "Per Catalog #307: PARADIGM (residual-correction hybrid stacking) INTACT; IMPLEMENTATION-LEVEL PARTIAL-VALIDATION with NEW PHENOMENON revealed at this fixture surface"
  - "Per CLAUDE.md 'Forbidden premature KILL': DEFERRED-pending-6TH-ORDER decomposition (Variant C-i'' magnitude-regularizer / C-vii per-pixel sign-diversity / C-iii paired heads / C-v asymmetric activation / C-vi Huber loss)"
  - "Canonical equation #347 anchors APPENDED via tools/append_boostnerv_variant_c_i_prime_differentiable_sign_diversity_empirical_anchors_20260526.py: aggregate + per-cell (sweet-spot k=5.0 gain_clamp=0.20); 7 -> 9 anchors (delta +2)"
  - "Operator-routable: descend to 6TH-ORDER decomposition with Variant C-i'' (add lambda_mag * mean(r^2) magnitude-regularizer) — highest-EV next step because it directly addresses the discovered 5TH-order coupled-objective bug class"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
horizon_class: frontier_pursuit
related_deliberation_ids:
  - boostnerv_variant_c_i_residual_loss_sign_diversity_term_landed_20260526
  - boostnerv_variant_c_ii_centered_base_recolor_landed_20260526
  - boostnerv_bpr1_variant_b_codec_redesign_landed_20260526
  - boostnerv_pr110_gain_clamp_sweep_landed_20260526
  - boostnerv_pr110_l1_empirical_landed_20260526
  - boostnerv_variant_c_i_prime_differentiable_sign_diversity_5th_order_pre_execution_gate_report_20260526
  - comprehensive_roadmap_synthesis_landed_20260526
  - t3_council_pr110_stacking_pivot_ordering_landed_20260526
---

# BoostNeRV-PR110 Variant C-i' `differentiable_sign_diversity_via_tanh_proxy` — LANDED 2026-05-26 (5TH-ORDER)

**Lane**: `lane_boostnerv_variant_c_i_prime_differentiable_sign_diversity_5th_order_20260526` (L1; impl_complete + memory_entry)
**Subagent**: `boostnerv-variant-c-i-prime-differentiable-sign-diversity-via-tanh-proxy-5th-order-recursive-doctrine-20260526`
**Predecessor**: Variant C-i 4TH-ORDER REFUTATION (commit `1075a2f30`; subagent `a760c7d0720813162`)
**Operator authority**: 2026-05-26 verbatim "all are approved"
**Wallclock**: 4.3 seconds (M5 Max MLX-local; 6 cells sequential)
**Cost**: $0 (MLX-local-only per "Remember all on MLX")

## Pre-execution gate verdict

Per `.omx/research/boostnerv_variant_c_i_prime_differentiable_sign_diversity_5th_order_pre_execution_gate_report_20260526.md` (PRE-DECLARED before empirical run):
- **V1** (positive_fraction_soft -> 0.5 at >=4 of 6 cells): **MET** (6/6 cells within +/-0.10 of 0.5)
- **V2** (sign-bitmap entropy > 0.2 bits at >=4 of 6 cells): **MET** (6/6 cells with entropy 0.92-1.00 bits)
- **V3** (Variant B-d sidecar shrinks below 157B baseline at >=1 cell): **FAILED** (0/6 cells below 157B; range [166B, 70784B])

Overall verdict: **SPLIT/PARTIAL** per pre-declared criteria. 4th-order mechanism diagnosis RETROACTIVELY VALIDATED at gradient-flow + sign-distribution surfaces; NEW 5TH-ORDER DISCOVERY at byte-axis surface.

## Entropy-position declaration

**Entropy position**: P2 = loss-shape (TRAIN phase). Same as C-i: intervention modifies UPSTREAM residual sign distribution. **Difference from C-i**: Variant C-i' provides nonzero chain-rule gradient back to residual parameters via `(tanh(k*r) + 1) / 2` (gradient `k*(1 - tanh^2(k*r))/2` nonzero almost everywhere). Empirical result: gradient flow WORKED at intended sign-distribution layer but triggered downstream magnitude-distribution failure.

## Full-stack fractal optimization decomposition (5TH-ORDER + next-level 6TH-ORDER nodes)

```
BoostNeRV-PR110 substrate
├── ingredient #6 curriculum / loss-shape
│   └── sub-ingredient L2 loss MSE
│       └── sub-sub-ingredient sign-diversity-regularizer (C-i) ← 4TH-ORDER REFUTED
│           └── sub-sub-sub-ingredient differentiable sign-proxy (C-i' tanh) ← 5TH-ORDER SOFT-VALIDATED + NEW DISCOVERY
│               ├── sub-sub-sub-sub-ingredient magnitude-regularizer (C-i'' lambda_mag*mean(r^2)) ← 6TH-ORDER NODE
│               ├── sub-sub-sub-sub-ingredient per-pixel sign-diversity (C-vii) ← 6TH-ORDER NODE
│               ├── sub-sub-sub-sub-ingredient paired +/- residual heads (C-iii) ← 6TH-ORDER NODE
│               ├── sub-sub-sub-sub-ingredient asymmetric activation (C-v) ← 6TH-ORDER NODE
│               └── sub-sub-sub-sub-ingredient Huber loss (C-vi) ← 6TH-ORDER NODE
```

**Recursive doctrine cumulative trajectory** (THIS = 6th landing):

| # | Landing | Decomposition node | Outcome | Next |
|---|---|---|---|---|
| 1 | #1337 L1 EMPIRICAL | training schedule | WIN | gain_clamp |
| 2 | #1342 gain_clamp sweep | codec hyperparameter | WIN | codec design |
| 3 | #1345 Variant B-d codec | codec design | REFUTED -> 2nd-order | training-dynamics |
| 4 | #1349 Variant C-ii centering | base-output centering | REFUTED -> 3rd-order | L2-loss-shape |
| 5 | #1351 Variant C-i sign-diversity | sign-diversity-regularizer | REFUTED -> 4th-order | differentiable proxy |
| 6 | **THIS Variant C-i' tanh-proxy** | **5TH-ORDER differentiable tanh-proxy** | **SOFT-VALIDATED gradient-flow + sign-distribution surfaces; NEW DISCOVERY magnitude explosion** | **6TH-ORDER C-i'' magnitude-regularizer** |

## Variant C-i' implementation LOC

- Sweep harness: `.omx/tmp/boostnerv_variant_c_i_prime_differentiable_sign_diversity_sweep.py` (~715 LOC; sister-derived from #1351 harness + tanh-proxy replacement)
- THE 5TH-ORDER FIX is a 1-line change in `_loss_fn`:
  - OLD: `positive_fraction = mx.mean((residual_pred > 0.0).astype(mx.float32))`
  - NEW: `positive_fraction_soft = mx.mean((mx.tanh(sharpness_k * residual_pred) + 1.0) / 2.0)`
- BPR1 sidecar carries 8-byte header overhead (4 bytes fp32 lambda + 4 bytes fp32 k; train-time-only; inflate doesn't need it; forensic reproducibility)
- Canonical Provenance per Catalog #323: every result row stamped `axis_tag=[macOS-MLX research-signal]` + `promotion_eligible=False` + `score_claim=False` + `ready_for_exact_eval_dispatch=False`.

## Empirical results (6-cell heatmap)

| | k=1.0 | k=5.0 | k=20.0 |
|---|---|---|---|
| **gc=0.05** | pos_soft=0.5000 / hard=0.333 / H=0.918 / A=126066B / B-d=166B / recon=11.2% | pos_soft=0.5008 / hard=0.333 / H=0.918 / A=496783B / B-d=166B / recon=11.1% | pos_soft=0.5011 / hard=0.504 / H=0.9999 / A=366240B / B-d=42050B / recon=7.7% |
| **gc=0.20** | pos_soft=0.4972 / hard=0.491 / H=0.9998 / A=1517079B / B-d=70784B / recon=38.9% | pos_soft=0.5002 / hard=0.663 / H=0.922 / A=1511406B / B-d=5009B / recon=25.7% | pos_soft=0.4976 / hard=0.630 / H=0.951 / A=974376B / B-d=34265B / recon=31.8% |

**KEY OBSERVATIONS**:
1. **positive_fraction_soft -> 0.500 at ALL 6/6 cells** (range [0.4972, 0.5011]). The differentiable tanh-proxy WORKED — chain-rule gradient flowed back to residual parameters.
2. **Hard positive_fraction moved from C-i baseline 0.0000 -> range [0.333, 0.663]** across cells. The soft proxy's gradient pressure propagated to the hard distribution at most cells.
3. **Sign-bitmap entropy grew from C-i baseline 0.0000 -> 0.918-1.000 bits** at all 6 cells (max approaches Shannon limit of 1.0 bit for binary).
4. **Variant A sidecar exploded** from 42B baseline -> 126066B - 1517079B (3000x - 36000x). The penalty bypassed gradient bottleneck but L2-MSE drove residuals to clip saturation.
5. **Variant B-d sidecar grew** from 149B baseline -> 166B - 70784B (1.06x - 451x). The sign-bitmap RLE compression fights with the magnitude explosion.
6. **Sweet-spot cell**: gc=0.20 k=5.0 has the SMALLEST Variant B-d at 5009B (still 33x baseline, but minimum among diversified cells).
7. **L2 reduction preserved** at all cells (max 38.9% at gc=0.20 k=1.0; range [7.7%, 38.9%]).

## Sign-distribution outcome

**SOFT-VALIDATED** at gradient-flow surface AND sign-distribution surface (entropy + hard indicator). The 4TH-order mechanism diagnosis is RETROACTIVELY VALIDATED: the non-differentiable indicator WAS the gradient-flow bottleneck. With differentiable tanh-proxy, the L2+tanh+symmetric-clip attractor IS escapable via loss-axis intervention.

## Sign-bitmap entropy

Range [0.918, 1.000] bits across 6 cells (Shannon limit 1.0 for binary). Maximum diversification at gc=0.05 k=20.0 (0.9999 bits) and gc=0.20 k=1.0 (0.9998 bits). This is FULL diversification at the indicator level.

## BPR1 sidecar bytes

- Variant A (int8): EXPLODED from 42B baseline to range [126066B, 1517079B]. Brotli cannot compress high-magnitude diversified-sign residuals.
- Variant B-d (sign-bitmap): grew from 149B baseline to range [166B, 70784B]. Sister RLE compresses the SIGN bitmap somewhat but the MAGNITUDE bytes blow up.

## Carmack-dissent verdict per Catalog #307

**PARADIGM-LEVEL REFUTATION**: NONE. Residual-correction hybrid stacking paradigm INTACT.

**IMPLEMENTATION-LEVEL VERDICT**: Variant C-i' PARTIAL-VALIDATED at this fixture surface. The 4TH-order mechanism diagnosis (non-differentiable indicator is the gradient-flow bottleneck) is RETROACTIVELY VALIDATED. NEW 5TH-ORDER DISCOVERY: differentiable penalty bypassed the gradient bottleneck BUT triggered residual magnitude explosion because soft sign-balance is achievable at any magnitude — the L2 term then drove residuals to clip saturation.

**5TH-ORDER DISCOVERY mechanism explanation**:
1. The soft proxy `(tanh(k*r) + 1) / 2` reaches 0.5 when `mean(tanh(k*r)) = 0`, which holds for ANY symmetric distribution including high-magnitude saturated outputs.
2. The L2-MSE objective drives the composed frame toward GT; with soft sign-balance achieved at any magnitude, the optimizer prefers HIGH-magnitude residuals because they enable more L2 reduction per step.
3. The clip operation `clip([-gain_clamp, +gain_clamp])` saturates the residuals at the boundary; sign distribution becomes ~50/50 at the boundary but magnitude distribution becomes bimodal at +/-gain_clamp.
4. Brotli encodes the int8-quantized boundary residuals as high-entropy bytes; sidecar bytes EXPLODE.

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion"**: substrate paradigm DEFERRED-pending-6TH-ORDER decomposition. Recommended 6TH-order nodes per Catalog #308:

- **Variant C-i''** `add_magnitude_regularizer_lambda_mag_mean_r_squared`: add `lambda_mag * mean(r^2)` to objective. Tests whether jointly penalizing sign-imbalance + magnitude solves the coupled-objective problem. HIGHEST-EV next step (~10 LOC change; directly addresses discovered phenomenon).
- **Variant C-vii** `per_pixel_or_per_channel_sign_diversity_penalty`: per-pixel/per-channel rather than global. Provides per-element differential signal.
- **Variant C-iii** `paired_positive_negative_residual_heads`: architectural sign-decomposition with separate gain_clamp per branch.
- **Variant C-v** `replace_tanh_with_asymmetric_activation`: e.g. shifted sigmoid; breaks tanh saturation symmetry.
- **Variant C-vi** `replace_L2_with_huber_loss`: softer gradient field may admit sign mixtures without magnitude explosion.

## Canonical equation #347 anchor IDs

Per Catalog #344 + `tools/append_boostnerv_variant_c_i_prime_differentiable_sign_diversity_empirical_anchors_20260526.py`:

- **Aggregate anchor**: `residual_hybrid_boosting_boostnerv_pr110_VARIANT_C_I_PRIME_differentiable_tanh_proxy_6cell_aggregate_5TH_ORDER_SOFT_VALIDATED_HARD_PARTIAL_byte_explosion_due_to_penalty_bypass_of_gradient_bottleneck_20260526`
- **Per-cell anchor**: `residual_hybrid_boosting_boostnerv_pr110_VARIANT_C_I_PRIME_differentiable_tanh_proxy_clamp_0p20_k_5p0_sweet_spot_cell_20260526` (sweet-spot k=5.0 gain_clamp=0.20; smallest B-d sidecar 5009B)
- **Registry state**: 7 -> 9 anchors (delta +2); equation status remains PROVISIONAL pending 6TH-ORDER empirical anchor.

## Drift surface declaration

All results stamped `[macOS-MLX research-signal]` per Catalog #192/#317/#341. Predicted MLX-CUDA drift band: O(1e-4) per canonical equation `mps_drift_architecture_class_dependent_v1`. NO CUDA dispatch in this work.

## Canonical-vs-frontier-push decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Fixture | CANONICAL-SISTER | Apples-to-apples with 5 sister sweeps |
| Codec sister-comparisons | CANONICAL-SISTER | Variant A + Variant B-d unchanged |
| Penalty formulation | FRONTIER-PUSH | 5TH-ORDER differentiable tanh-proxy replaces 4TH-ORDER indicator |
| Sweep axes | FRONTIER-PUSH | k x gain_clamp at fixed lambda=1.0 (max-pressure) |

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map: ACTIVE** (chain-rule sensitivity from penalty to residual params IS the diagnosed mechanism + empirically validated)
- hook #2 Pareto constraint: N/A (single-objective)
- hook #3 bit-allocator: N/A (no allocator change)
- **hook #4 cathedral autopilot dispatch: ACTIVE** (canonical equation #347 anchor consumed by autopilot ranker per Catalog #344)
- **hook #5 continual-learning posterior: ACTIVE** (2 anchors appended to canonical equation #347; registry state 7 -> 9)
- **hook #6 probe-disambiguator: ACTIVE** (empirical anchor IS the disambiguator between 4TH-order-mechanism-diagnosis-validation AND 5TH-order-magnitude-explosion-discovery)

## Operator-routable next step

**HIGHEST-EV NEXT-STEP per recursive doctrine**: 6TH-ORDER iteration with Variant C-i'' (add magnitude-regularizer `lambda_mag * mean(r^2)` to objective).
- Rationale: 5TH-ORDER empirically validated the differentiable-proxy mechanism but revealed the coupled-objective problem (sign-balance + magnitude-bounding). C-i'' directly addresses the discovered phenomenon with minimal change (~10 LOC).
- Cost: $0 (MLX-local) + 5-10 min wallclock + sister-cycle pattern.
- Predicted outcome: if `lambda_mag * mean(r^2)` bounds residual magnitudes, Variant B-d sidecar should drop below 157B baseline at multiple cells (joint sign-balance + magnitude-bounding); if NOT, the coupled-objective problem requires C-vii per-pixel OR C-iii architectural.

Sister candidates per Catalog #308: Variant C-vii per-pixel (provides per-element differential signal); Variant C-iii paired heads (architectural sign-decomposition); Variant C-vi Huber loss (softer gradient field).

Per CLAUDE.md "Long-burn score-lowering campaign default" + "Frontier target" + "Forbidden premature KILL": iteration continues. The 5TH-ORDER discovery REFINES our understanding rather than ending it.
