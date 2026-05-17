# L5 v2 Z6 Paired-Control Initialization Adversarial Review

- schema: `l5_v2_z6_paired_control_initialization_adversarial_review_v1`
- date: `2026-05-17`
- lane_id: `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- ready_for_paid_dispatch: `false`
- evidence_grade: `local_cpu_real_video_smoke_proxy_no_scorer`

## Finding

The Z6 full-FiLM versus identity-predictor controls were not apples-to-apples.
Both arms used the same seed, but the identity predictor has zero trainable
parameters. In the old constructor order, the full arm initialized predictor
parameters before shared `latent_init` and `residuals`, while the identity arm
skipped that RNG consumption. Same-seed paired smoke therefore compared
different shared latent/residual initial states.

This is an engineering-custody bug, not a Z6 paradigm result. It contaminated
the prior `identity_predictor_proxy_lower_loss` verdict.

## Fix

- `Z6PredictiveCodingSubstrate` now initializes shared encoder, decoder,
  `latent_init`, residuals, and ego-motion buffer before constructing the
  mode-specific predictor.
- Z6 smoke stats now emit
  `paired_control_initialization=shared_modules_seed_order_matched_v2`.
- The Z6 identity disambiguator refuses paired stats that lack that marker.
- The Z6 real-video ego-proxy sweep records the same marker in every row.
- A regression test asserts same-seed full and identity controls share all
  common state while preserving the predictor parameter-count difference.

## Recomputed Local Evidence

Commands:

```bash
.venv/bin/python -m pytest -q \
  src/tac/substrates/time_traveler_l5_z6/tests/test_z6.py \
  src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py \
  src/tac/tests/test_probe_z6_real_video_ego_proxy_sweep.py \
  src/tac/tests/test_probe_z6_predictor_liveness.py

.venv/bin/python tools/probe_z6_real_video_ego_proxy_sweep.py \
  --video-path upstream/videos/0.mkv --device cpu --epochs 3 --seed 0

.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/time_traveler_l5_z6/disambiguator_full_film_real_video_smoke_20260516_codex \
  --epochs 3 --device cpu --seed 0 \
  --smoke-target-mode real-video --smoke-ego-motion-mode real-video --smoke

.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/time_traveler_l5_z6/disambiguator_identity_real_video_smoke_20260516_codex \
  --epochs 3 --device cpu --seed 0 \
  --smoke-target-mode real-video --smoke-ego-motion-mode real-video --smoke \
  --identity-predictor

.venv/bin/python tools/probe_z6_predictive_coding_vs_identity_disambiguator.py \
  --full-stats experiments/results/time_traveler_l5_z6/disambiguator_full_film_real_video_smoke_20260516_codex/stats.json \
  --identity-stats experiments/results/time_traveler_l5_z6/disambiguator_identity_real_video_smoke_20260516_codex/stats.json \
  --output-json .omx/research/l5_v2_z6_identity_predictor_disambiguator_20260516_codex.json \
  --output-md .omx/research/l5_v2_z6_identity_predictor_disambiguator_20260516_codex.md
```

Observed:

- focused tests: `45 passed`
- paired disambiguator verdict:
  `full_film_predictor_proxy_lower_loss`
- real-video ego-proxy sweep verdict:
  `full_film_proxy_found_real_video_smoke`
- best cheap proxy: `random_control`
- best identity-minus-full loss proxy:
  `5.304813385009766e-06`

## Adversarial Classification

The fixed result is still proxy-only and weak. It shows the full-FiLM predictor
capacity is live under matched shared initialization. It does **not** show that
ego-motion semantics are useful, because the best proxy is `random_control` and
`semantic_ego_proxy_supported=false`.

Z6-v1 should therefore move from `blocked_identity_dominates` to
`proxy_candidate_found_requires_scorer_probe`, not to paid dispatch or paradigm
promotion.

## Required Next Gate

Before any paid Z6 full-main or exact-eval dispatch, run a scorer-bearing or
PoseNet-derived ego proxy probe. If the best proxy remains `zero` or
`random_control`, redesign the ego-conditioning objective before lifting the
Z6 `_full_main` gate.
