# Codex Routing Directive — TOP-2 Arbitrariness Extinction: Per-substrate Epochs Wildly Varies

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Value ID**: `epochs_wildly_varies_1_100_200_1000_2000`
**Resolution path**: `experimental` (with early-stopping formula sister)
**Predicted ΔS**: [-0.006, -0.001]
**Cost envelope**: **$0 (NET-NEGATIVE — saves money)**
**Rank score per dollar**: 6.0

## Bug class

Per-substrate `--epochs` defaults are wildly arbitrary:
- 1 epoch: nscs06 family
- 100: pr101_dp1
- 200: atw_codec_v1/v2, atw_v2_1, nscs02
- 1000: nscs01
- 2000: c1, c6, d4, cool_chic, pretrained_driving_prior, s2sbs, sane_hnerv...

No principled per-substrate convergence-aware stopping. Most substrates over-train (waste $); nscs06's 1-epoch under-trains (under-converged → CARGO-CULT per CLAUDE.md NSCS06 v6 → v7 44% rescue).

## 5-path analysis

1. **experimental** [RECOMMENDED] — convergence-aware early stopping: track val-score slope every N epochs; stop when slope < ε for K consecutive windows.
2. **formula** — Prechelt 1998 "Early Stopping — But When?" canonical GL_α / UP_K / PQ_α stopping criteria. Closed-form per validation curve.
3. **analytical_solve** — N/A (training curves rarely closed-form).
4. **learned** — Bayesian-optimal stopping (Murphy 2022 §9.3) overkill.
5. **self_alien_tech** — N/A.

## Concrete next step ($0)

Land canonical helper `tac.early_stopping.SlopeWatcher`:

```python
@dataclass(frozen=True)
class SlopeWatcherConfig:
    eval_interval_epochs: int = 50
    patience_windows: int = 3
    min_slope_improvement: float = -1e-4  # negative = improvement (lower score)
    smoothing_window: int = 5

class SlopeWatcher:
    def step(self, epoch: int, val_score: float) -> bool:
        """Returns True if training should stop."""
```

Wire into every substrate trainer's main training loop:

```python
watcher = SlopeWatcher(SlopeWatcherConfig())
for epoch in range(args.epochs):
    train_one_epoch(...)
    if epoch % watcher.config.eval_interval_epochs == 0:
        val_score = evaluate(...)
        if watcher.step(epoch, val_score):
            print(f"[early-stopping] stopped at epoch={epoch}")
            break
```

## Net effect

- Substrates over-training at 2000ep: save 30-70% of cost
- Substrates under-training at 1ep: catch convergence earlier
- Cost: $0; in fact NET-NEGATIVE (saves wall-clock GPU dollars)

## Sister coordination

- Couples with `epochs_wildly_varies_*` row + sister row `early_stopping_patience_undeclared`
- DP1 + PR101 + score-aware loss substrates benefit most

## Exit criteria

1. `tac.early_stopping.SlopeWatcher` canonical helper
2. Wired into ≥ 5 highest-cost substrate trainers first
3. Cost-band posterior shows wall-clock reduction
