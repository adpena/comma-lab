# Arbitrariness Extinction Audit — Top 50 Ranked

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Captured at UTC**: `2026-05-19T02:13:29Z`
**Source**: operator standing directive 2026-05-18 (5-path resolution: experimental / analytical_solve / formula / learned / self_alien_tech)
**Rows audited**: 52

## Ranking metric

`rank_score_per_dollar = |predicted_ev_delta_s.lower_bound| / cost_envelope_usd` (with `cost=0` → 1000× boost)

All `predicted_ev_delta_s` are PREDICTIONS not measurements (`evidence_grade=predicted` per Catalog #323).

## Table

| Rank | EV/$ | EV [lo,hi] | $ | Path | Value ID | Replacement | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 12.0000 | [-0.0120, -0.0030] | $0 | analytical_solve | `lambda_seg_pose_rate_multipliers_unprincipled` | Per-substrate Pareto-frontier sweep + KKT-derived multipliers: λ are LAGRANGE mu... | HIGH-EV; closed-form derivation per CLAUDE.md operating-point-dependent rule |
| 2 | 6.0000 | [-0.0060, -0.0010] | $0 | experimental | `epochs_wildly_varies_1_100_200_1000_2000` | Per-substrate convergence-aware early stopping: track validation score every N e... | NET-NEGATIVE cost (saves money) + reduces over/under-training |
| 3 | 6.0000 | [-0.0060, -0.0010] | $0 | learned | `per_pair_loss_weighting_uniform` | Per-pair difficulty-aware loss weighting: weight pairs by their CURRENT loss (fo... | HIGH-EV; LEARNED resolution; per-pair focal weighting |
| 4 | 5.0000 | [-0.0050, -0.0010] | $0 | formula | `ema_decay_0.997_hardcoded_all_substrate_trainers` | Effective-window formula: τ_eff ≈ 1/(1-decay). decay=0.997 → τ_eff ≈ 333 steps. ... | $0 cost = infinite EV per dollar; sister effect compounds across 30 trainers |
| 5 | 5.0000 | [-0.0050, -0.0010] | $0 | formula | `inflate_device_fallback_policy_PACT_INFLATE_DEVICE_auto` | Per CLAUDE.md 'Forbidden device-selection defaults (the MPS-fallback trap)': alr... | HIGH-EV at boundary; reproduces A1 0.19285 [contest-CPU] vs 0.22635 [contest-CUD... |
| 6 | 5.0000 | [-0.0050, -0.0010] | $0 | learned | `score_pair_components_weights_static` | Uncertainty-weighted multi-task loss (Kendall et al 2018 'Multi-Task Learning Us... | HIGH-EV; LEARNED resolution path; sister to λ_seg/pose/rate row |
| 7 | 3.0000 | [-0.0030, -0.0005] | $0 | formula | `qint_max_grid_1_3_7_15_31_arbitrary_water_filling` | Water-filling is mathematically OPTIMAL for the GIVEN qint_max grid — but the gr... | HIGH-EV; closed-form derivation per CLAUDE.md Meta-Lagrangian directive |
| 8 | 3.0000 | [-0.0030, -0.0005] | $0 | learned | `saliency_threshold_0.5_default_multiple_callsites` | src/tac/learnable_saliency_threshold.py ALREADY EXISTS as learnable parameter. T... | META: the LEARNED resolution path already EXISTS but isn't wired — pure engineer... |
| 9 | 3.0000 | [-0.0030, -0.0005] | $0 | formula | `early_stopping_patience_undeclared` | Validation-loss-slope early stopping with patience = K windows where slope < eps... | NET-NEGATIVE COST (saves money) + improves score |
| 10 | 3.0000 | [-0.0030, -0.0005] | $0 | self_alien_tech | `vqvae_codebook_init_random_uniform_vs_kmeans` | K-means++ init on real data (van den Oord 2017 VQ-VAE arxiv:1711.00937 §3 explic... | Composes with K=64 row above |
| 11 | 3.0000 | [-0.0030, -0.0005] | $0 | learned | `per_layer_lr_multiplier_uniform_undeclared` | Per-layer LR scaling (LARS, You et al 2017 'Large Batch Training of Convolutiona... | HIGH-EV; LAMB drop-in replacement |
| 12 | 2.0000 | [-0.0020, -0.0003] | $0 | formula | `warmup_epochs_undefined_or_hardcoded` | Linear warmup over first 5-10% of training (Goyal et al 2017 §2.2; He et al 2016... | $0 formula |
| 13 | 2.0000 | [-0.0020, -0.0003] | $0 | experimental | `lzma_preset_9_hardcoded` | preset=9 is already max for lzma; the ARBITRARINESS is whether lzma vs zstd-22 v... | $0 cost |
| 14 | 2.0000 | [-0.0020, -0.0003] | $0 | analytical_solve | `vq_codebook_K_64_hardcoded_neural_weight_codec` | Rate-distortion theoretic K: K minimizes total_bits = N * log2(K) + K * d * byte... | Sister codec already has sweep — the ARBITRARINESS is not WIRING the sweep resul... |
| 15 | 2.0000 | [-0.0020, -0.0003] | $0 | analytical_solve | `block_fp_block_size_undeclared_default` | Optimal block_size minimizes header_overhead + within_block_variance_loss. Heade... | Closed-form; $0 |
| 16 | 2.0000 | [-0.0020, -0.0003] | $0 | experimental | `sigma_15_grayscale_lut_hardcoded_per_design` | sigma is the Gaussian-LUT bandwidth for grayscale-to-class scoring; OPTIMAL sigm... | Inherited from Selfcomp without re-derivation |
| 17 | 2.0000 | [-0.0020, -0.0005] | $0 | analytical_solve | `HIGH_PAIR_INVARIANT_threshold_Catalog_319` | Compute threshold via ROC analysis on labeled DeliverabilityProof corpus: find t... | $0 closed-form |
| 18 | 2.0000 | [-0.0020, -0.0003] | $0 | analytical_solve | `stack_of_stacks_langevin_t_final_1e-4` | Langevin dynamics convergence theory: t_final should depend on target distributi... | Closed-form; $0 |
| 19 | 1.0000 | [-0.0010, -0.0001] | $0 | formula | `validation_set_fraction_15_percent_unprincipled` | Contest defines the test set; internal val split must NOT leak from train. Use s... | Low-EV unless current val split has leakage |
| 20 | 1.0000 | [-0.0010, -0.0002] | $0 | experimental | `brotli_quality_10_vs_11_inconsistent` | Per-payload quality-vs-time tradeoff measurement: brotli quality=11 is ~3× slowe... | $0 cost — purely engineering |
| 21 | 1.0000 | [-0.0010, -0.0001] | $0 | analytical_solve | `vqvae_mask_codebook_K_256_hardcoded` | Same R-D solve as neural_weight_codec_K=64; couple with mask-codec compression r... | Lower-EV; mask codec less score-affecting at frontier |
| 22 | 1.0000 | [-0.0010, -0.0001] | $0 | analytical_solve | `inflate_per_frame_decode_priority_implicit` | Per-frame decode ordering affects ZIP locality and dictionary reuse for compress... | Low-EV unless current ordering is pathological |
| 23 | 1.0000 | [-0.0010, +0.0000] | $0 | experimental | `council_4_tier_cadence_3_per_day_3_per_week_arbitrary` | Empirically calibrate against actual decision throughput: track per-tier deliber... | Low-direct-EV; meta-level effect compounds |
| 24 | 1.0000 | [-0.0010, +0.0000] | $0 | experimental | `staleness_window_30_days_hardcoded_8_surfaces` | Per-surface empirical calibration: measure decay-rate of probe-outcome relevance... | Meta-level; affects dispatch routing efficiency not score directly |
| 25 | 1.0000 | [-0.0010, +0.0000] | $0 | formula | `sextet_quorum_5_of_6_arbitrary_threshold` | Bayesian-aggregation theoretic: quorum should depend on per-member calibration (... | Meta-level; current quorum may be over-rigorous (rejecting good deliberations) |
| 26 | 1.0000 | [-0.0010, -0.0002] | $0 | analytical_solve | `coupling_threshold_0.5_master_gradient_consumers` | Coupling threshold should derive from per-pair gradient inner-product statistics... | Closed-form $0 |
| 27 | 1.0000 | [-0.0010, +0.0000] | $0 | formula | `inflate_py_loc_budget_200_HNeRV_parity_L4` | LOC is a proxy for review-time + dependency-closure-risk. Per CLAUDE.md HNeRV pa... | Meta-level; affects review discipline not score directly |
| 28 | 1.0000 | [-0.0010, -0.0001] | $0 | formula | `gradient_clipping_undeclared_across_trainers` | Gradient clipping at norm 1.0 or 5.0 is standard for stability. Per Pascanu et a... | Low-EV unless gradients are pathological |
| 29 | 1.0000 | [-0.0010, -0.0001] | $0 | formula | `lr_warmup_init_lr_factor_default_undefined` | Per Smith 2017 §3.2: warmup should start at lr_base / 10 not 0. Zero start waste... | $0 formula tweak |
| 30 | 0.5000 | [-0.0005, +0.0000] | $0 | analytical_solve | `rashomon_ensemble_K_8_members_arbitrary` | Rashomon set size: K determined by bootstrap-confidence-interval width. Per Fish... | Meta-level; preflight Rashomon affects gate confidence not score directly |
| 31 | 0.5000 | [-0.0005, +0.0000] | $0 | analytical_solve | `per_pair_file_list_ordering_sequential` | Per-pair ordering optimization: order pairs to maximize ZIP dictionary reuse. So... | Low-EV unless current order is pathological |
| 32 | 0.2000 | [-0.0002, +0.0000] | $0 | experimental | `80_char_negation_window_Catalog_236` | Per-corpus negation-window calibration: measure false-positive vs false-negative... | Meta-meta; affects preflight false-positive/negative not score directly |
| 33 | 0.1000 | [-0.0001, +0.0000] | $0 | formula | `frontier_threshold_cpu_default_0.192_cathedral_autopilot` | Should auto-derive from canonical state via tac.frontier_scan (Catalog #316). Ha... | Already extincted by Catalog #316; verify wiring |
| 34 | 0.1000 | [-0.0001, +0.0000] | $0 | experimental | `memory_file_rotation_60_days_hardcoded` | Per-category empirical decay-rate calibration | Meta-meta; not score-affecting |
| 35 | 0.1000 | [-0.0001, +0.0000] | $0 | formula | `catalog_quota_400_arbitrary` | Catalog quota should derive from preflight execution-time budget: if each gate c... | Meta-meta; not score-affecting |
| 36 | 0.0040 | [-0.0200, -0.0050] | $5.00 | experimental | `c6_ibps_bottleneck_dim_24_falsified` | Per CLAUDE.md C6 IBPS 22× miss anchor: 24-dim destroys segmentation (score_seg=2... | HIGH-EV; reactivates per CLAUDE.md 'Forbidden premature KILL' |
| 37 | 0.0020 | [-0.0040, -0.0010] | $2.00 | analytical_solve | `batch_size_wildly_varies_1_4_8_16_32_per_substrate` | VRAM-aware analytical solve: batch_size = floor((vram_budget_gb - model_size_gb ... | Composable with lr sweep; net cost amortized |
| 38 | 0.0015 | [-0.0030, -0.0005] | $2.00 | experimental | `siren_omega_0_frequency_hyperparameter` | Per-resolution ω_0 sweep: Sitzmann et al 2020 'Implicit Neural Representations w... | SIREN-specific |
| 39 | 0.0010 | [-0.0030, -0.0005] | $3.00 | experimental | `kl_distill_temperature_T_2.0_hinton_default` | Per-distillation T sweep: Hinton 2014 §3 explicitly notes T is dataset-dependent... | May not apply if current operating point doesn't use KL distill |
| 40 | 0.0010 | [-0.0020, -0.0003] | $2.00 | experimental | `adamw_betas_eps_universal_pytorch_defaults` | Per Choi et al 2019 'On Empirical Comparisons of Optimizers for Deep Learning' a... | Sister-to lr_finder sweep |
| 41 | 0.0007 | [-0.0020, -0.0003] | $3.00 | learned | `default_activation_xavier_init_undocumented` | Per-substrate init sweep (kaiming-uniform/normal/xavier/orthogonal); for FiLM-co... | Coupled with lr_finder |
| 42 | 0.0006 | [-0.0030, -0.0005] | $5.00 | experimental | `weight_decay_1e-5_hardcoded_all_substrate_trainers` | Joint LR+weight_decay sweep (Loshchilov-Hutter 2017 §5): the AdamW paper explici... | Marginal cost is ~$0 if amortized with LR sweep |
| 43 | 0.0005 | [-0.0080, -0.0020] | $15.00 | experimental | `lr_5e-4_hardcoded_30_substrate_trainers` | Per-substrate LR finder (Smith 2017 cyclical LR range test): 1ep sweep over lr ∈... | EV per dollar: 0.000533 lower bound; HIGHEST-RATIO since affects ~30 trainers si... |
| 44 | 0.0005 | [-0.0010, -0.0002] | $2.00 | experimental | `non_arbitrariness_cos_thresholds_keep_0.30_prune_0.85` | Per-substrate cosine-similarity sweep + measure downstream pruning impact on sco... | META: the existing non-arbitrariness helper has arbitrary thresholds |
| 45 | 0.0004 | [-0.0080, -0.0020] | $20.00 | experimental | `composition_alpha_cascade_2_reward_bands_1.0_1.10_1.20` | Per-composition-pair empirical α measurement: run paired smoke + measure actual ... | HIGH-EV; many cargo-cult α values; couples with PR101 fec6 + PR106 format0d fron... |
| 46 | 0.0003 | [-0.0100, -0.0020] | $30.00 | learned | `quantizr_renderer_88K_94K_params_unprincipled` | Neural Architecture Search (Zoph-Le 2017 'Neural Architecture Search with Reinfo... | HIGH-EV; LTH at $1-2 per CLAUDE.md MEMORY.md is cheapest path |
| 47 | 0.0003 | [-0.0050, -0.0010] | $20.00 | experimental | `random_seed_single_per_run_no_seed_ensemble` | k-seed ensemble + take best (cheap noise-floor estimate). Reimers-Gurevych 2017 ... | HIGH-EV but expensive; better as cheap macOS-CPU advisory ranker |
| 48 | 0.0000 | [+0.0000, +0.0000] | $0 | contest_fixed | `contest_rate_term_25_factor_and_37545489_denom` | CONTEST-FIXED — cannot change. | OUT-OF-SCOPE — contest-fixed reference |
| 49 | 0.0000 | [+0.0000, +0.0000] | $0 | contest_fixed | `sqrt_10_pose_weight_contest_formula_inherited` | CONTEST-FIXED — formula sqrt(10·pose_avg) + 100·seg_avg + 25·archive_bytes/N | OUT-OF-SCOPE — used as reference for λ_pose derivation |
| 50 | 0.0000 | [+0.0000, +0.0000] | $0 | contest_fixed | `num_classes_5_segmentation_contest_fixed` | CONTEST-FIXED | REFERENCE |
| 51 | 0.0000 | [+0.0000, +0.0000] | $0 | contest_fixed | `pair_count_600_or_1199_contest_fixed` | CONTEST-FIXED | REFERENCE |
| 52 | 0.0000 | [+0.0000, +0.0000] | $0 | contest_fixed | `frame_resolution_384x512_contest_input` | CONTEST-FIXED at input; internal compressed-domain resolution is FREE | REFERENCE (sub-resolution compress fields are NOT contest-fixed per nscs02) |

## Clustering by resolution path

### `experimental` (17 rows)

- Total predicted ΔS envelope: [-0.0663, -0.0138]
- Total cost envelope: $74.00
- Rows: lr_5e-4_hardcoded_30_substrate_trainers, weight_decay_1e-5_hardcoded_all_substrate_trainers, epochs_wildly_varies_1_100_200_1000_2000, kl_distill_temperature_T_2.0_hinton_default, brotli_quality_10_vs_11_inconsistent, lzma_preset_9_hardcoded, sigma_15_grayscale_lut_hardcoded_per_design, council_4_tier_cadence_3_per_day_3_per_week_arbitrary, staleness_window_30_days_hardcoded_8_surfaces, 80_char_negation_window_Catalog_236

### `analytical_solve` (11 rows)

- Total predicted ΔS envelope: [-0.0280, -0.0058]
- Total cost envelope: $2.00
- Rows: batch_size_wildly_varies_1_4_8_16_32_per_substrate, lambda_seg_pose_rate_multipliers_unprincipled, vq_codebook_K_64_hardcoded_neural_weight_codec, vqvae_mask_codebook_K_256_hardcoded, block_fp_block_size_undeclared_default, inflate_per_frame_decode_priority_implicit, HIGH_PAIR_INVARIANT_threshold_Catalog_319, coupling_threshold_0.5_master_gradient_consumers, stack_of_stacks_langevin_t_final_1e-4, rashomon_ensemble_K_8_members_arbitrary

### `formula` (12 rows)

- Total predicted ΔS envelope: [-0.0232, -0.0036]
- Total cost envelope: $0.00
- Rows: ema_decay_0.997_hardcoded_all_substrate_trainers, warmup_epochs_undefined_or_hardcoded, validation_set_fraction_15_percent_unprincipled, qint_max_grid_1_3_7_15_31_arbitrary_water_filling, inflate_device_fallback_policy_PACT_INFLATE_DEVICE_auto, sextet_quorum_5_of_6_arbitrary_threshold, frontier_threshold_cpu_default_0.192_cathedral_autopilot, early_stopping_patience_undeclared, inflate_py_loc_budget_200_HNeRV_parity_L4, gradient_clipping_undeclared_across_trainers

### `learned` (6 rows)

- Total predicted ΔS envelope: [-0.0290, -0.0053]
- Total cost envelope: $33.00
- Rows: saliency_threshold_0.5_default_multiple_callsites, score_pair_components_weights_static, quantizr_renderer_88K_94K_params_unprincipled, default_activation_xavier_init_undocumented, per_pair_loss_weighting_uniform, per_layer_lr_multiplier_uniform_undeclared

### `self_alien_tech` (1 rows)

- Total predicted ΔS envelope: [-0.0030, -0.0005]
- Total cost envelope: $0.00
- Rows: vqvae_codebook_init_random_uniform_vs_kmeans

### `contest_fixed` (5 rows)

- Total predicted ΔS envelope: [+0.0000, +0.0000]
- Total cost envelope: $0.00
- Rows: contest_rate_term_25_factor_and_37545489_denom, sqrt_10_pose_weight_contest_formula_inherited, num_classes_5_segmentation_contest_fixed, pair_count_600_or_1199_contest_fixed, frame_resolution_384x512_contest_input


## Top-5 highest-EV-per-dollar (operator action priority)

### 1. `lambda_seg_pose_rate_multipliers_unprincipled` (EV/$ = 12.0000)

- **File**: `experiments/train_substrate_*.py — score-aware loss multipliers across all trainers`
- **Current**: Hand-tuned λ_seg, λ_pose, λ_rate per substrate (e.g. nscs02 args.seg_weight + args.pose_weight; cool_chic ar_rate_weight) without principled basis
- **Resolution path**: `analytical_solve`
- **Predicted ΔS**: [-0.0120, -0.0030]
- **Cost**: $0.00 (N/A — already $0)
- **Replacement**: Per-substrate Pareto-frontier sweep + KKT-derived multipliers: λ are LAGRANGE multipliers for constrained R-D-distortion optimization. The CONTEST FORMULA fixes their relationship: total_score = sqrt(10*pose_avg) + 100*seg_avg + 25*archive_bytes/N. So gradient-matching at the operating point: λ_seg / λ_pose = (∂score/∂seg_avg) / (∂score/∂pose_avg). Per CLAUDE.md SegNet-vs-PoseNet operating-point-dependent rule: at PR106 frontier (pose_avg ~3.4e-5), the marginal ratio FLIPS to pose 2.71× SegNet. So λ_pose should be ~2.71× λ_seg at frontier.
- **Literature**: Boyd-Vandenberghe 2004 'Convex Optimization' Ch.5; sister to Catalog #322 composition α / operator's existing 'Meta-Lagrangian/Pareto solver' NON-NEGOTIABLE per CLAUDE.md
- **Canonical helper**: `src/tac/sensitivity_map/ exists; tac.score_lagrangian proposed`
- **Notes**: HIGH-EV; closed-form derivation per CLAUDE.md operating-point-dependent rule

### 2. `epochs_wildly_varies_1_100_200_1000_2000` (EV/$ = 6.0000)

- **File**: `experiments/train_substrate_*.py (1=nscs06; 100=pr101_dp1; 200=atw_v1/v2; 1000=nscs01/nscs02; 2000=c1/c6/d4/cool_chic/etc)`
- **Current**: hand-picked per substrate
- **Resolution path**: `experimental`
- **Predicted ΔS**: [-0.0060, -0.0010]
- **Cost**: $0.00 (N/A — net-positive cost saving)
- **Replacement**: Per-substrate convergence-aware early stopping: track validation score every N epochs; stop when slope < epsilon for K consecutive windows. The 2000-ep default for most substrates is over-training (waste $); 1-ep for nscs06 is under-training (under-converged). Canonical helper: tac.early_stopping.SlopeWatcher.
- **Literature**: Prechelt 1998 'Early Stopping — But When?' Neural Networks: Tricks of the Trade
- **Canonical helper**: `N/A — propose tac.early_stopping.SlopeWatcher`
- **Notes**: NET-NEGATIVE cost (saves money) + reduces over/under-training

### 3. `per_pair_loss_weighting_uniform` (EV/$ = 6.0000)

- **File**: `experiments/train_substrate_*.py — uniform pair weights in score-aware loss`
- **Current**: uniform across 1199 (or 600) pairs
- **Resolution path**: `learned`
- **Predicted ΔS**: [-0.0060, -0.0010]
- **Cost**: $0.00 (N/A)
- **Replacement**: Per-pair difficulty-aware loss weighting: weight pairs by their CURRENT loss (focal loss, Lin et al 2017 'Focal Loss for Dense Object Detection' arxiv:1708.02002) OR by their LEARNED uncertainty σ_pair. Per-pair pose+seg vary by 100x across the 1200 pairs; uniform weighting under-penalizes hard pairs.
- **Literature**: Lin et al 2017 arxiv:1708.02002 / Kendall et al 2018
- **Canonical helper**: `src/tac/substrates/_shared/score_aware_common.py`
- **Notes**: HIGH-EV; LEARNED resolution; per-pair focal weighting

### 4. `ema_decay_0.997_hardcoded_all_substrate_trainers` (EV/$ = 5.0000)

- **File**: `experiments/train_substrate_*.py (ALL 30+ trainers)`
- **Current**: 0.997
- **Resolution path**: `formula`
- **Predicted ΔS**: [-0.0050, -0.0010]
- **Cost**: $0.00 (N/A — already $0)
- **Replacement**: Effective-window formula: τ_eff ≈ 1/(1-decay). decay=0.997 → τ_eff ≈ 333 steps. OPTIMAL decay is per-substrate: τ_eff should = ~10-30% of total training steps. For 2000ep × 600 pairs/ep = 1.2M steps, 0.997 → τ_eff=333 = 0.028% of training (WAY too short). Per-substrate formula: decay = 1 - 1/(0.2 * total_steps). ALTERNATIVE: Polyak-Ruppert averaging (Polyak-Juditsky 1992) — provably optimal asymptotic SGD.
- **Literature**: Polyak-Juditsky 1992 'Acceleration of stochastic approximation by averaging' SIAM J. Control Optim. 30(4); Quantizr 0.33 used 0.997 empirically but his training was 5-stage; ours is mostly single-stage
- **Canonical helper**: `src/tac/training.py::EMA (would extend to accept decay='auto'/formula)`
- **Notes**: $0 cost = infinite EV per dollar; sister effect compounds across 30 trainers

### 5. `inflate_device_fallback_policy_PACT_INFLATE_DEVICE_auto` (EV/$ = 5.0000)

- **File**: `submissions/*/inflate.py via tac.substrates._shared.inflate_runtime.select_inflate_device`
- **Current**: auto (CUDA if available; else CPU); MPS explicitly refused per Catalog #205
- **Resolution path**: `formula`
- **Predicted ΔS**: [-0.0050, -0.0010]
- **Cost**: $0.00 (N/A)
- **Replacement**: Per CLAUDE.md 'Forbidden device-selection defaults (the MPS-fallback trap)': already canonical. However, the AUTO selection per-archive can yield different floating-point results (CPU vs CUDA bicubic kernel; per CLAUDE.md A1 PR Council Round 1 F1/F11 anchor +0.0335 gap). OPTIMAL: per-submission archive declares its training-time device + inflate-time pins to that device for byte-deterministic reproduction.
- **Literature**: A1 PR Council F1 anchor 2026-05-13
- **Canonical helper**: `src/tac/substrates/_shared/inflate_runtime.py exists`
- **Notes**: HIGH-EV at boundary; reproduces A1 0.19285 [contest-CPU] vs 0.22635 [contest-CUDA] gap


## Total predicted envelope

- All actionable rows (excluding contest-fixed): 47
- Total predicted ΔS envelope: **[-0.1495, -0.0290]**
- Total cost envelope: **$109.00**
- Frontier today (per CLAUDE.md FRONTIER section): 0.19205 [contest-CPU] / 0.20533 [contest-CUDA]
- If ALL extinctions PROCEED with lower-bound EV realized: 0.19205 - 0.1495 = 0.0426 predicted [contest-CPU]

## Provenance + custody

Every row carries `provenance` per Catalog #323 (`tac.provenance.build_provenance_for_predicted`).
All rows: `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, `evidence_grade=predicted`.
PREDICTED EV ranges are HYPOTHESES, not measurements. Empirical anchors required before any score claim.
