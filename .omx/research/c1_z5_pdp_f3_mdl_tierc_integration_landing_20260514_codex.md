# C1/Z5 + PDP F3 + MDL Tier-C Integration Landing - 2026-05-14

## Scope

This landing promotes the sibling implementation batch that was already
present in the shared worktree after the HDM8 recovery closure. It is code, not
strategy-only work:

- C1 world-model foveation can opt into Z5's `HierarchicalPredictor` through
  `WorldModelConfig.route_compute_to_z5`.
- Time-Traveler L5 exposes a neutral `Z5RoutedLatentPredictor` adapter for
  future predictive-residual variants without replacing the v1 SIREN path.
- VQ-VAE and pre-trained-driving-prior trainers wire the canonical Tier-1/F3
  optimization context as opt-in/default-off surfaces.
- PDP score-aware loss accepts `GTScorerCache` kwargs and routes through the
  canonical cache-aware score dispatcher.
- Z1/MDL ablation adds DP1 grammar support and Tier-C perturbation for DP1's
  renderer state_dict plus per-pair residual.
- Cathedral autopilot halves, but does not remove, the C1-class literature
  reward pending Z5 empirical evidence.

## Evidence

Focused test surface:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/substrates/c1_world_model_foveation/tests/test_c1_z5_routing_and_autopilot_halve.py \
  src/tac/substrates/time_traveler_l5_autonomy/tests/test_z5_routed_latent_predictor.py \
  src/tac/substrates/pretrained_driving_prior/tests/test_score_aware_loss_f3_kwargs.py \
  src/tac/tests/test_f3_backport_vqvae_pdp_wired.py \
  src/tac/tests/test_mdl_ablation_tier_c_dp1.py \
  src/tac/tests/test_mdl_ablation_tier_c_pr106.py \
  src/tac/tests/test_cathedral_autopilot_tier_c_and_composition.py -q
# 154 passed in 6.97s
```

CLI import/help checks:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/mdl_scorer_conditional_ablation.py --help >/dev/null
PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/train_substrate_vq_vae.py --help >/dev/null
PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/train_substrate_pretrained_driving_prior.py --help >/dev/null
```

`git diff --check` passed before promotion.

## Classification

No score claim. This is infrastructure and substrate-composition enablement for
the next score-lowering sweep. Promotion gates remain:

- Z5/C1 routed variants need byte-closed archive/runtime packets and exact
  `[contest-CUDA]` or `[contest-CPU]` auth-eval custody before ranking.
- F3 cache is a trainer speed path only; it is not an eval-axis score claim.
- MDL Tier-C DP1/PR106 rows are diagnostic and must stay separate from
  leaderboard score claims.

## Next

Use the wired surfaces to reduce wall-clock for C1/Z5/PDP/VQ-VAE training and
to run apples-to-apples MDL Tier-C density comparisons across PR106, C6/IBPS1,
and DP1 before dispatch ranking changes.
