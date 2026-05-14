# Composition Cell EV Ranking — Top 5

- Generated at: 2026-05-14T04:16:15.013249+00:00
- Schema: tac_composition_cell_ev_ranking_v1
- Total ranked cells: 5181

**CLAUDE.md compliance**: every row is `[predicted; substrate × primitive matrix v1 × posterior reweight]`. `score_claim=False`. This is operator decision input, NOT a score-promotion artifact.

**Per-axis marginal weight (PR106 r2 operating point)**: pose=2.71, seg=1.00, rate=0.50, mixed=1.50 (per CLAUDE.md operating-point-dependent rule).

**Notes**: Smoke run from custody+variance+composition tools landing 2026-05-13

| Rank | Cell | Substrate | Class | Primitives | Predicted ΔS band | Cost (USD) | EV/$ | Anchors | Readiness | Source |
|------|------|-----------|-------|------------|-------------------|------------|------|---------|-----------|--------|
| 1 | `cell__c3_residual__cooperative_receiver_atick_redlich__predictive_coding_rao_ballard` | `c3_residual` | residual | cooperative_receiver_atick_redlich,predictive_coding_rao_ballard | [-0.02705, -0.02455] | $0.50 | 0.04257 | 2 | L2 | [predicted; C3 + temporal hyperprior] |
| 2 | `cell__c3_residual__compressai_cheng2020__cooperative_receiver_atick_redlich` | `c3_residual` | residual | compressai_cheng2020,cooperative_receiver_atick_redlich | [-0.02485, -0.02235] | $0.50 | 0.03894 | 2 | L2 | [predicted; C3 + temporal hyperprior] |
| 3 | `cell__c3_residual__compressai_balle_hyperprior__cooperative_receiver_atick_redlich` | `c3_residual` | residual | compressai_balle_hyperprior,cooperative_receiver_atick_redlich | [-0.02415, -0.02165] | $0.50 | 0.03778 | 2 | L2 | [predicted; C3 + temporal hyperprior] |
| 4 | `cell__c3_residual__compressai_factorized_prior__cooperative_receiver_atick_redlich` | `c3_residual` | residual | compressai_factorized_prior,cooperative_receiver_atick_redlich | [-0.02340, -0.02090] | $0.50 | 0.03655 | 2 | L2 | [predicted; C3 + temporal hyperprior] |
| 5 | `cell__c3_residual__lzma__cooperative_receiver_atick_redlich` | `c3_residual` | residual | lzma,cooperative_receiver_atick_redlich | [-0.02225, -0.01975] | $0.50 | 0.03465 | 2 | L2 | [predicted; C3 + temporal hyperprior] |

## Methodology

Per-cell EV/$:

```
EV_score    = abs(predicted_ΔS_midpoint × posterior_correction)
p_holds     = sigmoid(n_authoritative_anchors, midpoint=2)
EV_per_$    = (EV_score × p_holds × axis_weight)
              / max(estimated_dispatch_cost_usd, 0.05)
```

- `predicted_ΔS` band is from the alien-tech expert-team memos (2026-05-13 wave) cross-referenced against substrate anchors.
- `posterior_correction` ∈ [0.5, 1.5] derived from the most recent authoritative anchor for the architecture class.
- `p_holds` ∈ [0.05, 0.95] sigmoid in the anchor count.
- `axis_weight` per CLAUDE.md PR106-r2 operating point.
- `readiness` = max lane level across lanes whose `lane_id` contains the substrate token.

## Rows with blockers (excerpt)

| Cell | Blocker(s) |
|------|------------|
| `cell__c3_residual__compressai_cheng2020__cooperative_receiver_atick_redlich` | semantic_compatibility_warning: primitive 'compressai_cheng2020' is semantically redundant on substrate 'c3_residual' (substrate already provides this primitive's benefit; applying it is a no-op) |
| `cell__c3_residual__compressai_balle_hyperprior__cooperative_receiver_atick_redlich` | semantic_compatibility_warning: primitive 'compressai_balle_hyperprior' is semantically redundant on substrate 'c3_residual' (substrate already provides this primitive's benefit; applying it is a no-op) |
| `cell__c3_residual__compressai_factorized_prior__cooperative_receiver_atick_redlich` | semantic_compatibility_warning: primitive 'compressai_factorized_prior' is semantically redundant on substrate 'c3_residual' (substrate already provides this primitive's benefit; applying it is a no-op) |
| `cell__c3_residual__lzma__cooperative_receiver_atick_redlich` | semantic_compatibility_warning: primitive 'lzma' is semantically redundant on substrate 'c3_residual' (substrate already provides this primitive's benefit; applying it is a no-op) |
