# Codex Routing Directive — TOP-3 Arbitrariness Extinction: Per-pair Loss Weighting Uniform

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Value ID**: `per_pair_loss_weighting_uniform`
**Resolution path**: `learned`
**Predicted ΔS**: [-0.006, -0.001]
**Cost envelope**: $0
**Rank score per dollar**: 6.0

## Bug class

Every substrate trainer weights the 1199 (or 600) per-pair losses **uniformly**. But per-pair `pose_avg` + `seg_avg` vary by 100× across the 1200 pairs (per-pair difficulty distribution is highly skewed). Uniform weighting under-penalizes hard pairs → optimizer spends gradient on easy pairs already near zero.

## 5-path analysis

1. **experimental** — sweep static reweighting schemes. Slow.
2. **analytical_solve** — Pareto-frontier solver across 1200 pairs. Possible but overkill.
3. **formula** — Focal loss (Lin et al 2017 arxiv:1708.02002): `weight_pair_i = (1 - p_i)^γ` where `p_i = exp(-current_loss_i)`.
4. **learned** [RECOMMENDED] — Kendall et al 2018 multi-task uncertainty weighting: `weight_pair_i = 1/(2σ_i²)`; σ_i learned as nn.Parameter.
5. **self_alien_tech** — per-pair curriculum (Bengio et al 2009 'Curriculum Learning').

## Concrete next step ($0)

Land canonical extension to `tac.substrates._shared.score_aware_common.score_pair_components`:

```python
def score_pair_components(
    *,
    pred: Tensor,
    target: Tensor,
    per_pair_focal_gamma: float = 0.0,   # 0 = uniform; 2.0 = focal-default
    per_pair_uncertainty_sigma: nn.Parameter | None = None,  # learned weighting
    ...,
) -> ScorePairComponents:
    """Compute per-pair seg/pose/rate scores with optional focal or learned reweighting."""
```

Wire as opt-in default to `per_pair_focal_gamma=1.0` (light focal) across substrate trainers.

## Coupling

- Sister `score_pair_components_weights_static` (row #6) proposes Kendall uncertainty weighting at axis level
- This row applies SAME idea at per-pair level (one dimension deeper)

## Exit criteria

1. Focal + uncertainty weighting added to `score_pair_components`
2. Per-substrate smoke confirms per-pair loss variance reduction
3. Empirical anchor confirms predicted ΔS lower bound
