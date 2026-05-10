# Check #126 lane-registry backfill

Generated: `2026-05-10T09:20:00Z`

## Verdict

Dev preflight had regressed on Catalog #126:
`check_lane_pre_registered_before_work_starts` reported `138` references across
`115` unique unregistered lane tokens in recently touched source scripts.

The durable fix was two-part:

1. Treat `lane_internal_name` as a metadata key, not a lane id.
2. Backfill all real source-referenced lanes as Level-0 registry sketches via
   `tools/lane_maturity.py add-lane`, not by hand-editing the registry.

This is registration-only. It does not promote any lane, claim any score, or
change dispatch readiness.

## Commands

Backfill command shape:

```bash
.venv/bin/python tools/lane_maturity.py add-lane <lane_id> \
  --name "Legacy lane backfill: <name>" \
  --phase 1 \
  --notes "Backfilled 2026-05-10 by Codex via lane_maturity CLI to satisfy strict Check #126 for existing source references; registration only, Level 0 sketch, no score claim or promotion."
```

False-positive fix:

```text
src/tac/preflight.py: add lane_internal_name to _LANE_ID_REFERENCE_BLOCKLIST
```

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_preflight_all_clean_cache.py src/tac/tests/test_check_lane_pre_registered_before_work_starts.py`
  -> `138 passed in 15.35s`
- `.venv/bin/python tools/lane_maturity.py validate`
  -> `OK - 283 lane(s) validated cleanly.`
- `.venv/bin/python -m tac.preflight --scope dev --timings-json experiments/results/preflight_dev_timing_swarm_check126_backfill_20260510_codex.json`
  -> `PREFLIGHT PASSED`, `wall_elapsed_s=9.223678`, `serial_elapsed_s=4.320585`

## Backfilled lane ids

`117` Level-0 sketch registrations were added to the local lane registry:

```text
lane_12_alpha_geo0_pose_regen
lane_12_nerv
lane_12_owv3_0120_nerv_stack
lane_19_logit_margin
lane_19_premise
lane_19_score_snapshot
lane_8_multipass_g_v3
lane_a
lane_a_optimized
lane_a_pose_tto
lane_a_sweep
lane_ac_archive_codec
lane_al_analog_latent
lane_cg
lane_d_halfframe
lane_d_retrofit
lane_d_v2_halfframe
lane_d_v3_annealed_kldistill
lane_d_v3_mismatch
lane_d_v3_premise
lane_darts_s_segmap_arch_sweep
lane_ea_entropy_archive
lane_ebr_entropy_bottleneck
lane_ec_engineered_corrections
lane_ec_v2_greedy_waterfill
lane_f_v2_fp4_qat_on_lane_a
lane_f_v3_fp4_qat_int8warmup_on_lane_a
lane_f_v4_mixed_precision_fp4_on_lane_a
lane_f_v5
lane_f_v5_hardware_fp8
lane_f_v5_premise
lane_fc_film_canvas
lane_fl_raft_derived_poses
lane_fr_mm_sigma_sweep
lane_fr_omega_fridrich_block_fp
lane_g_v3_anchor
lane_g_v3_masks
lane_g_v3_omega_w_v2_stack
lane_g_v3_pfp16_a_plus_plus_t4
lane_g_v3_pfp16_a_plus_plus_t4_20260430
lane_g_v3_pfp16_stack
lane_g_v3_poses
lane_g_v3_renderer
lane_ge_geodesic_pose
lane_gh_ghost_renderer
lane_gh_premise
lane_gp_gaussian_process_pose
lane_h_v3_joint_halfframe
lane_h_v3_premise
lane_hm
lane_hm_s_segmap_homography
lane_i_coolchic
lane_i_coolchic_renderer_on_lane_a_masks_poses
lane_j_imp_iterative_magnitude_pruning
lane_j_nwc_premise
lane_j_nwcs
lane_j_nwcs_ec_stack
lane_k_dsconv_quantizr_killer
lane_line_search_c067
lane_lm_a_zero_cost_poses
lane_lm_v2_endpoint_tracking
lane_lr
lane_lr_v2
lane_m_plus_zero_cost_poses_eval
lane_m_pose_mode
lane_m_v2_radial_zoom_proper_padding
lane_m_v3_pose_from_embedding_path_a
lane_mae_v
lane_mm_baseline_score
lane_mm_grayscale_lut
lane_mm_sigma_sweep
lane_n_linf_pose_budget
lane_n_linf_pose_weight
lane_omega_hessian_qat
lane_omega_hessian_qat_on_lane_a
lane_omega_v2
lane_omega_v2_lagrangian
lane_omega_v2_lagrangian_on_lane_a
lane_omega_v3_rate_frontier
lane_omega_v3_rate_frontier_on_lane_a
lane_omega_w_water_filling
lane_pa_pose_as_affine
lane_pd_pose_deltas
lane_ps_per_class_segnet_weighting
lane_ps_v2_learnable_segnet_class_weights
lane_psd_standard
lane_psd_std
lane_q_faithful_jointgen_88k
lane_q_faithful_premise
lane_qat_fp4
lane_qat_sweep
lane_rm
lane_s_self_compress
lane_s_self_compress_on_lane_a
lane_s_v2_auto_warmup
lane_s_v2_auto_warmup_on_lane_a
lane_sa_segmap_clone
lane_saug_v2
lane_sc_plus_plus_kl_distill
lane_sh_shannon_arithmetic
lane_si_v2_learnable_threshold
lane_so_hessian_block_fp
lane_sq_semantic_quantization
lane_stc_boundary_coding
lane_sz_phase2
lane_sz_phase2_szabolcs_no_masks_paradigm
lane_tr_temporal_residual
lane_uniward_texture
lane_v_channel_bug
lane_v_premise
lane_v_quantizr_replica_88k_halfframe
lane_v_v2_premise
lane_v_v2_quantizr_replica_88k_halfframe_annealed
lane_w_hard_pair
lane_w_v2_learnable_pair_weights
lane_wc
lane_wc_s_curator_weighted
```

## Notes

`.omx/state/lane_registry.json` and `.omx/state/lane_maturity_audit.log` are
local state surfaces and remain ignored. This ledger records the reproducible
backfill operation so the strict-preflight repair is not lost as chat-only
state.
