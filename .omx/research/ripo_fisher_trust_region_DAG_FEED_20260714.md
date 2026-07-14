# DAG FEED: categorical Fisher trust-region head update

**UTC date:** 2026-07-14  
**lane:** `lane_ripo_fisher_isometric_trust_region_500_20260714`  
**research_only:** `true`  
**pointer_delta:** `0`  
**verdict_scope:** output-space law built; raw-sum cross-space candidate v1 falsified;
one-step mean-loss v3 stopped on source TOCTOU with no verdict; few-step exact-delta sweep
building; full pullback family open

## Nodes

| node | kind | status | authority |
|---|---|---|---|
| `ripo_action_ratio_eq10` | paper law | VERIFIED | full paper text |
| `categorical_softmax_fisher_k5` | derived equation | DERIVED | exact finite KL + Taylor expansion |
| `winner_rival_margin_curvature` | derived equation | DERIVED | full K=5 probability vector |
| `delta_exact_flip_spill_quantiles` | cross-space measurement law | MEASURED_N600 | analytic scorer-output tie-KL after actual R; not a head perturb/replay flip budget |
| `categorical_fisher_output_space_clip` | NumPy/MLX build | LOCAL-VERIFIED | 78 passed, 2 no-Metal skips; exact-KL primary |
| `sequential_cpu_scorer_axis_v3` | measurement custody | COMPLETE | exact archive bytes; fresh batch4/threads1 vector = 3,970,482 errors; old concurrent count non-transferable |
| `raw_sum_cross_space_candidate_v1` | fixed-head formulation | FALSIFIED_INSTANCE | summed CE gradient used without `/N`; stopped at invalid 60/600, no verdict |
| `mean_loss_cross_space_candidate_v3` | fixed-head formulation | BLOCKED_INPUT_TOCTOU_NO_VERDICT | exact `/ce_sample_count`; source changed at 165/600; pair-zero restart required |
| `iterated_baseline_policy_candidate_v4` | fixed-head formulation | BUILDING | >=3 fixed-trunk head steps; exact-KL q10/q25/q50 plus controls; no optimizer-equivalence |
| `uniform_ratio_ppo_control` | control formulation | BUILT_NOT_N600_MEASURED | constant categorical ratio budget; uniform L2 retained separately |
| `postdeploy_head_constraint_audit` | measurement custody | BUILT_NOT_N600_MEASURED | replay final reprojected + int8 deployed `H_i Delta theta`, not temporary field |
| `receiver_ste_or_secant_jvp_vjp` | tangent operator | BLOCKED | exact R is discontinuous; declared STE/secant + adjoint/secant receipt absent |
| `full_pullback_head_metric` | reformulation | BLOCKED_NO_RECEIVER_PULLBACK | requires matrix-free `J^T F J` custody |
| `v9_dsl_trust_region_lever` | live integration | HELD | exclusive owner + complete receipt |

## Derivation edges

```text
ripo_action_ratio_eq10
  -> [reject direct scalar p1-to-logit transfer]
categorical_softmax_fisher_k5
  -> winner_rival_margin_curvature
  -> categorical_fisher_output_space_clip
actual_R + frozen_CPU_SegNet + current_EMA
  -> sequential_cpu_scorer_axis_v3
  -> delta_exact_flip_spill_quantiles
delta_exact_flip_spill_quantiles + categorical_fisher_output_space_clip
  -> raw_sum_cross_space_candidate_v1 [FALSIFIED: missing /N]
  -> mean_loss_cross_space_candidate_v3 [BLOCKED: source TOCTOU, no verdict]
  -> iterated_baseline_policy_candidate_v4
iterated_baseline_policy_candidate_v4
  -> uniform_ratio_ppo_control
  -> postdeploy_head_constraint_audit
postdeploy_head_constraint_audit + full_n600_controls
  -> {FORMULATION_GO, FORMULATION_NO_GO, NO_VERDICT_CUSTODY}
receiver_ste_or_secant_jvp_vjp
  -> full_pullback_head_metric
  -> future receiver-aware head-law candidate
favorable complete receipt + Pose + archive parseback
  -> held v9_dsl_trust_region_lever
```

## Triality

- **Equation:** `categorical_fisher_logit_trust_region_v1`, with explicit `delta_kl` and
  `delta_quad=2*delta_kl` conventions.
- **DAG:** this feed; negative edges carry explicit INSTANCE/FORMULATION/FAMILY scope.
- **DSL:** held typed Lever/LawRef/consumer spec in
  `codex_findings_ripo_fisher_trust_region_20260714_codex.md`; no live/hot edit by this lane.

## Gates

1. exact all-int8 receiver byte reproduction: archive SHA `81a4c516...613b7`, 63,664 bytes;
   the historical Task #336 count 3,970,488 is not reused across its concurrent scorer process;
2. fresh sequential batch4/threads1 n600 baseline vector, independently replayed by candidate
   fit/evaluation; upstream batch16 and the fp32 training row are separate substrates;
3. exact correction/spill KL quantile custody;
4. mean-loss vanilla vs PPO uniform-ratio vs separately labeled uniform-L2 vs local-Fisher vs
   exact-KL, same head direction/data/order;
5. strict 485-parameter only-head-changed proof plus realized post-reprojection/int8 constraint
   violation receipt; pre-reprojection clipping is not sufficient;
6. per-class/overall/fix/spill/confidence-band table; confidence-band variance is formulation
   stability only, not RIPO Proposition 4.1;
7. Pose and archive parse-back before launch/pointer authority;
8. sole typed V9 DSL path, non-null content-hashed receipt, and provenance bijection closure.

Any failed cross-space projection is a `FORMULATION` negative.  It does not kill full-block
Fisher, `J^T F J`, per-class delta, or KL-proximal families.
