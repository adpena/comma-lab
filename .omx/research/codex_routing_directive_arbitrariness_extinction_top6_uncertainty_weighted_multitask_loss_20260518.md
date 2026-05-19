# Codex Routing Directive — TOP-6 Arbitrariness Extinction: score_pair_components Weights Static (Kendall Uncertainty)

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Value ID**: `score_pair_components_weights_static`
**Resolution path**: `learned`
**Predicted ΔS**: [-0.005, -0.001]
**Cost envelope**: $0
**Rank score per dollar**: 5.0

## Bug class

`src/tac/substrates/_shared/score_aware_common.py::score_pair_components` uses static λ per axis (seg/pose/pixel/rate). Even if TOP-1 (`lambda_seg_pose_rate_multipliers_unprincipled`) derives analytic baseline multipliers, the OPTIMAL per-batch weighting depends on per-batch gradient statistics that vary throughout training.

## 5-path analysis

1. **experimental** — sweep λ schedules. Expensive.
2. **analytical_solve** — Pareto front per-batch. Theoretically clean; expensive.
3. **formula** — KKT multipliers (TOP-1 canonical row). Static; baseline.
4. **learned** [RECOMMENDED] — Kendall et al 2018 "Multi-Task Learning Using Uncertainty to Weight Losses" arxiv:1705.07115: learn σ_i per axis; loss = Σ_i (1/(2σ_i²)) · L_i + log σ_i. The σ are LEARNED.
5. **self_alien_tech** — GradNorm (Chen et al 2018 arxiv:1711.02257) — alternative learnable scheme.

## Concrete next step ($0)

Land Kendall uncertainty extension:

```python
class UncertaintyWeightedScoreLoss(nn.Module):
    def __init__(self, n_axes: int = 3):  # seg, pose, rate
        super().__init__()
        # log(σ²) param; init to 0 (σ=1, neutral)
        self.log_sigma_sq = nn.Parameter(torch.zeros(n_axes))

    def forward(self, per_axis_losses: Tensor) -> Tensor:
        # loss = Σ_i (1/(2σ_i²)) · L_i + (1/2)log(σ_i²)
        return ((-self.log_sigma_sq).exp() * per_axis_losses / 2.0 + self.log_sigma_sq / 2.0).sum()
```

Add as opt-in mode in `score_pair_components(uncertainty_weighted=True)`.

## Coupling

- TOP-1 (`lambda_seg_pose_rate_multipliers_unprincipled`) provides ANALYTIC baseline λ — Kendall σ LEARNED relative to that baseline
- TOP-3 (`per_pair_loss_weighting_uniform`) extends Kendall to per-pair level — both compose

## Exit criteria

1. `UncertaintyWeightedScoreLoss` canonical helper
2. Wired as opt-in for ≥5 substrate trainers
3. Empirical anchor confirms predicted ΔS lower bound
