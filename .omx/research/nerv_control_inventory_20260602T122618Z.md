# NeRV control inventory

Schema: `nerv_control_inventory.v1`
Authority: `false_authority_control_inventory_no_score_claim`

## Spend Rule

admit a control only when measured delta_nonrate_score + contest_byte_price_score_per_byte * delta_archive_bytes < 0 on the matching evidence axis; MLX rows can route training budget but never claim score or promotion

## Controls

| control | applies to | binding status | missing binding count |
|---|---|---|---|
| hi_nerv_hierarchical_capacity | hi_nerv | partially_wired_needs_measured_ladder | 3 |
| hi_nerv_bitstream_quantization | hi_nerv | partially_wired_needs_hi_nerv_codec_sweep_replay | 1 |
| hi_nerv_sr_resolution_axis | hi_nerv | design_knob_needs_trained_sr_receiver | 1 |
| sr_nerv_lowres_receiver_axis | cross_stack | design_knob_needs_receiver_closed_training | 1 |
| hi_nerv_inverse_steg_decoder_weight_fit | hi_nerv | not_wired_into_real_trainer | 1 |
| snerv_frequency_split | snerv | partially_wired_cpu_advisory_mlx_missing | 2 |
| snerv_lf_modelsize_and_stepmap | snerv | needs_representation_change | 2 |
| snerv_pose_guarded_hf_restoration | snerv | advisory_only_needs_scorer_loop_training | 1 |
| hnerv_modelsize_control | cross_stack | control_baseline_available | 1 |
| rnerv_config_optimizer | cross_stack | planner_missing | 1 |
| ffnerv_flow_temporal_redundancy | cross_stack | not_wired_into_compact_carrier_training | 1 |
| nervplusplus_decoder_efficiency_blocks | cross_stack | architecture_candidate_needs_rate_priced_ladder | 1 |
| vq_c3_cool_chic_latent_codebook | cross_stack | partially_wired_needs_full_video_section_pricing | 3 |
| inverse_steganalysis_saliency_stack | cross_stack | available_needs_carrier_domain_binding | 1 |
| master_gradient_xray_stack | cross_stack | available_needs_nerv_control_consumer | 1 |
| full_video_vjp_master_gradient_authority | cross_stack | available_needs_exact_reduced_bundle_consumer | 3 |
| bitmask_and_zero_packing | cross_stack | available_needs_family_specific_packet_layout | 1 |
| receiver_exact_custody_gate | cross_stack | available_must_remain_hard_gate | 2 |

## Recommended Work Orders

- `cache_quality_gate_required_before_profile_or_spend` from `hi_nerv_hierarchical_capacity`
- `measured_hi_nerv_modelsize_budget_ladder` from `hi_nerv_hierarchical_capacity`
- `decoder_weight_vjp_or_saliency_proxy_in_hi_nerv_full_main` from `hi_nerv_inverse_steg_decoder_weight_fit`
- `mlx_native_snerv_train_export` from `snerv_frequency_split`
- `snerv_measured_modelsize_ladder` from `snerv_lf_modelsize_and_stepmap`
- `full_video_section_value_for_vq_codebook_indices` from `vq_c3_cool_chic_latent_codebook`
- `full_video_vjp_bundle_as_budget_spend_prerequisite` from `full_video_vjp_master_gradient_authority`
- `push_saliency_into_hi_nerv_weight_groups_and_snerv_wavelet_groups` from `inverse_steganalysis_saliency_stack`
- `runnable_rnerv_style_config_search_over_hi_nerv_snerv_controls` from `rnerv_config_optimizer`
- `all_compact_carrier_emitters_on_shared_archive_bound_contract` from `receiver_exact_custody_gate`

## Implementation Sweep

- `hi_nerv`: `local_implementation_has_blocking_gaps_no_method_negative` (17 blocking gaps)
- `snerv`: `local_implementation_has_blocking_gaps_no_method_negative` (16 blocking gaps)

## Model-Size Ladder

- `hi_nerv`: 4 rows, 9 marginal gates
- `snerv`: 6 rows, 8 marginal gates

## Sources

- [hnerv_official](https://github.com/haochen-rye/HNeRV): architecture size, pruning, quantization, model bpp
- [hinerv_official](https://github.com/hmkx/HiNeRV): HiNeRV S/M/L configs, patch size, bitstream-q
- [hinerv_paper](https://arxiv.org/abs/2306.09818): hierarchical encodings plus pruning and quantization pipeline
- [snerv_official](https://github.com/qwertja/SNeRV): DWT LF/HF, enc/dec strides, fc_dim, emb_size, temporal extension
- [snerv_paper](https://arxiv.org/abs/2501.01681): spectral split, HFR/MFU/TUB controls
- [sr_nerv_paper](https://arxiv.org/abs/2505.00046): low-detail INR representation plus super-resolution
- [vq_nerv_paper](https://arxiv.org/abs/2403.12401): shallow residual/inter-frame VQ codebook controls
- [rnerv_paper](https://arxiv.org/abs/2506.24127): per-video design and training search over NeRV components
- [ffnerv_paper](https://arxiv.org/abs/2212.12294): flow-guided temporal redundancy and compact convolutional architecture
- [nervplusplus_paper](https://arxiv.org/abs/2402.18305): separable residual decoder blocks and skip-layer capacity
- [c3_paper](https://arxiv.org/abs/2312.02753): overfitted hierarchical latents and entropy-model bytes
- [cool_chic_docs](https://orange-opensource.github.io/Cool-Chic/encoding/architecture.html): hierarchical latent grids and autoregressive entropy model
- [nvrc_paper](https://arxiv.org/abs/2409.07414): end-to-end neural representation quantization and entropy coding
