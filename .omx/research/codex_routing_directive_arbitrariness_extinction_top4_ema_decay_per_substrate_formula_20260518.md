# Codex Routing Directive — TOP-4 Arbitrariness Extinction: EMA Decay 0.997 Universal

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Value ID**: `ema_decay_0.997_hardcoded_all_substrate_trainers`
**Resolution path**: `formula`
**Predicted ΔS**: [-0.005, -0.001]
**Cost envelope**: $0
**Rank score per dollar**: 5.0

## Bug class

Per CLAUDE.md "EMA — NON-NEGOTIABLE, HIGHEST EMPHASIS": EMA decay = 0.997 across ALL 30+ substrate trainers. Quantizr 0.33 used 0.997 — but Quantizr's training was 5-stage (each stage ~50-200 epochs). Ours is mostly single-stage 200-2000 ep.

Effective-window formula:
```
τ_eff = 1 / (1 - decay)
0.997 → τ_eff = 333 steps
```

For a 2000-ep substrate × 600 pairs/ep = 1.2M training steps, τ_eff=333 means the EMA shadow is averaging only the LAST 0.028% of training. That's WAY too short — the shadow is essentially the last-few-batch weights, not a proper exponential average.

## 5-path analysis

1. **experimental** — sweep decay ∈ {0.99, 0.995, 0.997, 0.999, 0.9995, 0.9999}. Slow.
2. **formula** [RECOMMENDED] — derive from training-step count: `decay = 1 - 1/(0.2 * total_steps)`. So τ_eff = 20% of training, matching standard practice.
3. **analytical_solve** — Polyak-Juditsky 1992 (`Acceleration of stochastic approximation by averaging`) provides provably optimal asymptotic SGD averaging. Closed-form.
4. **learned** — Learn `decay` as a parameter (would need careful initialization). Overkill.
5. **self_alien_tech** — N/A.

## Concrete next step ($0)

Land canonical extension to `src/tac/training.py::EMA`:

```python
class EMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float | str = "auto",  # "auto" = formula-derived
        target_window_fraction: float = 0.2,  # 20% of training
        total_steps: int | None = None,
    ):
        if decay == "auto":
            if total_steps is None:
                raise ValueError("decay='auto' requires total_steps=int")
            decay = 1.0 - 1.0 / (target_window_fraction * total_steps)
        ...
```

Per-substrate trainer changes:
```python
total_steps = args.epochs * pairs_per_epoch
ema = EMA(model, decay="auto", total_steps=total_steps)
```

## Composability

- Couples with `epochs_wildly_varies_*` row (TOP-2): once per-substrate epochs are early-stopped, the EMA decay re-derives accordingly via the formula
- Does NOT change codebook EMA (van den Oord persistent buffer form; 0.99 default keeps adapting fast)

## Exit criteria

1. `tac.training.EMA` accepts `decay="auto"` + `total_steps=int`
2. ≥10 substrate trainers wire `decay="auto"` as new default
3. Empirical anchor on smallest substrate confirms predicted ΔS lower bound
