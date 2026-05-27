# Codex Findings: Repair Campaign Palette Action Functional Integration

UTC: 2026-05-27T19:53:51Z

Lane: `lane_repair_campaign_palette_action_functional_integration_20260527`

## Finding

The repair campaign scorer already had a multiscale action ledger, entropy-position weights, first-class repair families, MLX advisory custody gates, stackability replay bundles, and posterior learning. The remaining signal-loss gap was narrower: the live FEC6 K=16 palette asymmetry was present in the upstream frontier feedback/materializer path but was not a first-class scoring feature inside the default repair campaign scorer.

The live empirical palette is:

```text
none
frame0_blue_chroma_amp_1
frame0_blue_chroma_amp_3
frame0_luma_bias_+1
frame0_luma_bias_-1
frame0_luma_bias_-2
frame0_luma_bias_-4
frame0_rgb_bias_m2_p1_p1
frame0_rgb_bias_m4_p2_p2
frame0_rgb_bias_p0_m1_p1
frame0_rgb_bias_p0_m2_p2
frame0_rgb_bias_p0_p1_m1
frame0_rgb_bias_p0_p2_m2
frame0_rgb_bias_p2_m1_m1
frame0_rgb_bias_p4_m2_m2
frame0_roll_dx+0_dy+1
```

That is one identity mode, fifteen non-identity frame0 modes, and zero frame1 modes. The correct mathematical interpretation is not "try more leaf modes"; it is a global frame0 color/geometry interaction prior spanning pixel, boundary, region, frame, pair, batch, and scorer axes.

## Landing

Implemented:

- `palette_frame_asymmetry_prior` now carries the empirical K=16 palette dynamics as a false-authority mathematical prior.
- `score_repair_campaign(...)` derives a `palette_dynamics_context` from explicit row priors, or from the first-class palette family prior when the row is a palette repair.
- Campaign score now includes a bounded `palette_frame_asymmetry_multiplier` so the optimizer can prefer measured frame0-concentrated palette repair without granting score, dispatch, or budget authority.
- Palette rows require `repair_dynamics_palette_probe_matrix_path` as local advisory custody before they are ready for local stackability execution.
- Hard constraints now record that frame0 palette repairs are global interaction terms and that frame1 palette repair requires counterfactual probing when the live palette has zero frame1 modes.
- Optimizer allocations, stackability probes, and learning-signal planner feature vectors preserve palette dynamics fields so the posterior can learn from the signal.
- Fixed a scale-classification bug where false-authority metadata such as `charged_bits_changed` inside `interaction_scope` could be misread as a real bit-scale signal.

## Authority

All new surfaces remain false-authority:

- no score claim
- no promotion eligibility
- no rank or kill authority
- no budget spend authority
- no exact dispatch authority

The output is local MLX advisory planning signal only. Exact score movement still requires receiver-runtime materialization, component-response replay, stackability remeasurement, and contest CPU/CUDA auth eval.

## Verification

Focused verification:

- `ruff check` on touched scheduler and repair campaign files
- `pytest src/tac/tests/test_repair_campaign_scorer.py -q`
- `pytest src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_repair_campaign_stackability_queue.py src/tac/tests/test_repair_cascade_mlx_probe_queue.py -q`
- `pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`

Broader validation remains required before any submission or exact dispatch.
