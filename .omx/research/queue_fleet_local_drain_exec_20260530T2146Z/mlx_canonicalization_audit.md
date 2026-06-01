# MLX Canonicalization Audit

- Schema: `mlx_canonicalization_audit.v1`
- Ready: `False`
- MLX files: `84`
- Routed/canonical: `47`
- Unique waivers: `0`
- Review required: `37`

## Review Required

- `src/tac/substrates/boost_nerv_pr110_residual/architecture.py` (bilinear_resize, upsample)
- `src/tac/substrates/boost_nerv_pr110_residual/long_training_adapter.py` (mlx import)
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/trainer.py` (mx.eval)
- `src/tac/substrates/coin_pp_implicit_neural_representation/mlx_renderer.py` (bilinear_resize, upsample)
- `src/tac/substrates/faiss_ivf_pq_residual/mlx_renderer.py` (bilinear_resize, upsample)
- `src/tac/substrates/hinton_distilled_scorer_surrogate/catalyst_cascade.py` (mlx import)
- `src/tac/substrates/hinton_distilled_scorer_surrogate/mlx_loss.py` (bilinear_resize)
- `src/tac/substrates/mdl_ibps_j_discrete_categorical_mine_hybrid/long_training_adapter.py` (mlx import)
- `src/tac/substrates/mdl_ibps_j_discrete_categorical_mine_hybrid/mlx_renderer.py` (mlx import)
- `src/tac/substrates/nirvana_cascading_nerv/long_training_adapter.py` (mlx import)
- `src/tac/substrates/nscs06_v8_chroma_lut/long_training_adapter.py` (mlx import)
- `src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py` (upsample)
- `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py` (mx.eval)
- `src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py` (pixel_shuffle, bilinear_resize, upsample)
- `src/tac/substrates/wyner_ziv_pipeline_stage_codec/trainer.py` (mlx import)
- `src/tac/substrates/z7_mamba2_v2_fresh_substrate/long_training_adapter.py` (mlx import)
- `tools/cascade_b_catalyst_cascade_composition_5th_order_mlx_local_anchor.py` (mlx import)
- `tools/cascade_b_catalyst_sister_wave_2_production_scale_post_train_qat_real_segnet_20260526.py` (mx.eval, bilinear_resize)
- `tools/cascade_b_path_a_learnable_head_smoke.py` (mx.eval, bilinear_resize)
- `tools/export_pact_nerv_ia3_mlx_to_pytorch_state_dict.py` (upsample)
- `tools/export_pact_nerv_selector_v2_mlx_to_pytorch_state_dict.py` (upsample)
- `tools/export_pact_nerv_selector_v3_mlx_to_pytorch_state_dict.py` (upsample)
- `tools/export_pact_nerv_selector_v4_mlx_to_pytorch_state_dict.py` (upsample)
- `tools/export_pact_nerv_vq_mlx_to_pytorch_state_dict.py` (upsample)
- `tools/export_z6_v2_cargo_cult_unwind_mlx_to_pytorch_state_dict.py` (upsample)
- `tools/extract_master_gradient_mlx.py` (mlx import)
- `tools/gate_mlx_candidate_contest_equivalence_pact_nerv_ia3.py` (upsample)
- `tools/gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v3.py` (upsample)
- `tools/gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v4.py` (upsample)
- `tools/gate_mlx_candidate_contest_equivalence_pact_nerv_vq.py` (upsample)
- `tools/gate_mlx_candidate_contest_equivalence_z6.py` (upsample)
- `tools/gate_mlx_candidate_contest_equivalence_z6_v2.py` (upsample)
- `tools/mlx_bitnet_158_pilot.py` (mx.eval)
- `tools/nscs06_v8_chroma_lut_hinton_distill_600pair_long_mlx.py` (mlx import)
- `tools/smoke_kahan_ema_vs_naive_z6.py` (upsample)
- `tools/uniward_6th_order_into_boostnerv_capacity_constrained_sweep_20260526.py` (mx.eval)
- `tools/uniward_per_pixel_n_plus_1_real_scorer_anchored_sweep_20260526.py` (mx.eval, bilinear_resize)
