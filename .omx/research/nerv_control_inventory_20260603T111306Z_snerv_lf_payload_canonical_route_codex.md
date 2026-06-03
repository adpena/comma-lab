# NeRV control inventory

Schema: `nerv_control_inventory.v1`
Authority: `false_authority_control_inventory_no_score_claim`

## Spend Rule

admit a control only when measured delta_nonrate_score + contest_byte_price_score_per_byte * delta_archive_bytes < 0 on the matching evidence axis; MLX rows can route training budget but never claim score or promotion

## Controls

| control | applies to | binding status | missing binding count |
|---|---|---|---|
| sr_nerv_lowres_receiver_axis | cross_stack | design_knob_needs_receiver_closed_training | 1 |
| snerv_frequency_split | snerv | receiver_hfr_mfu_t_ready_mlx_export_surfaces_unproven | 2 |
| snerv_lf_modelsize_and_stepmap | snerv | needs_representation_change | 3 |
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

- `snerv_mlx_native_full_video_scoreaware_train_export_proof` from `snerv_frequency_split`
- `snerv_real_segnet_posenet_teacher_loop_for_mlx_native_export` from `snerv_frequency_split`
- `snerv_measured_modelsize_ladder` from `snerv_lf_modelsize_and_stepmap`
- `decoder_weight_waterfill_plan_for_snerv_receiver_rows` from `snerv_lf_modelsize_and_stepmap`
- `full_video_section_value_for_vq_codebook_indices` from `vq_c3_cool_chic_latent_codebook`
- `full_video_vjp_bundle_as_budget_spend_prerequisite` from `full_video_vjp_master_gradient_authority`
- `push_saliency_into_hi_nerv_weight_groups_and_snerv_wavelet_groups` from `inverse_steganalysis_saliency_stack`
- `runnable_rnerv_style_config_search_over_hi_nerv_snerv_controls` from `rnerv_config_optimizer`
- `all_compact_carrier_emitters_on_shared_archive_bound_contract` from `receiver_exact_custody_gate`

## Implementation Sweep

- `snerv`: `local_implementation_has_blocking_gaps_no_method_negative` (14 blocking gaps)

## Model-Size Ladder

- `snerv`: 8 rows, 12 marginal gates

## Measured Archive Size Ladders


## Decoder Weight Waterfill

- `snerv`: `decoder_weight_waterfill_rows_available_false_authority` (1 rows, 3 section values)

## Archive Ladder Replay Actuators


## Archive Backend Drift


## Decoder-Weight Saliency Replays


## Decoder Mode Assignment

- `snerv`: `decoder_mode_assignment_rows_available_false_authority` (1 rows, 1 local-probe ready)

## Decoder Mode Probe

- `snerv`: best `explicit_fp163` score `3.6133971405465926` (1 candidates)

## SNeRV Scorer-Loop QAT

- `snerv`: `snerv_scorer_loop_qat_report_available_false_authority` (1 pairs, accepted=False, history=3, delta=0.0)

## SNeRV LF Payload Codec

- `snerv`: `snerv_lf_payload_codec_sweep_rows_available_false_authority` (6 reports, selected=int64_lzma, bytes=666556)

## Source Review Policy

- paper/OSS evidence is research context only until tiny forward parity and receiver byte grammar pass
- SNeRV means spectra-preserving DWT/MFU/HFR/TUB carrier unless a row explicitly says scalable-layered SNeRV
- HiNeRV and HNeRV are distinct; HNeRV/PR95 transfers must be priced as controls, not source parity
- PR95 is the same-axis public control arm; reproduce its archive/runtime/eval axis before beat claims
- CPU, CUDA, and MLX observations stay separate even when the archive and report text look identical

## Sources

- [comma_leaderboard_video_compression](https://comma.ai/leaderboard): public frontier ordering; PR95/98/101/103/105/106 remain same-axis control inputs, not inferred-equivalence authority
- [pr95_public_control](https://github.com/commaai/comma_video_compression_challenge/pull/95): HNeRV-shaped contest specialization: 178417-byte archive, 8-stage scorer/rate curriculum, C1a entropy shaping, QAT, sigma sweep, Muon, and documented CPU/CUDA discrepancy
- [hnerv_official](https://github.com/haochen-rye/HNeRV): architecture size, pruning, quantization, model bpp
- [hinerv_official](https://github.com/hmkx/HiNeRV): HiNeRV S/M/L configs, patch/frame flexibility, deepspeed/timm/torchac code path, pruning, quantization, and bitstream-q
- [hinerv_paper](https://arxiv.org/abs/2306.09818): hierarchical encodings plus pruning and quantization pipeline
- [snerv_official](https://github.com/qwertja/SNeRV): spectra-preserving SNeRV: Haar/DWT LF/HF, enc/dec strides, fc_dim, emb_size, MFU, HFR, and temporal extension
- [snerv_paper](https://arxiv.org/abs/2501.01681): spectral split, HFR/MFU/TUB controls
- [snerv_scalable_disambiguation](https://openreview.net/forum?id=ZqN4bnXSSY): separate Scalable Neural Representation paper; useful for layered bitstream thinking, not the spectra-preserving DWT/MFU/HFR carrier target
- [sr_nerv_paper](https://arxiv.org/abs/2505.00046): low-detail INR representation plus super-resolution
- [vq_nerv_paper](https://arxiv.org/abs/2403.12401): shallow residual/inter-frame VQ codebook controls
- [rnerv_paper](https://arxiv.org/abs/2506.24127): per-video design and training search over NeRV components
- [rnerv_vinrb_oss](https://github.com/mgwillia/vinrb): official VINRB/RNeRV implementation and ablation framework; use as source-parity input for NeRV-family component controls
- [ffnerv_paper](https://arxiv.org/abs/2212.12294): flow-guided temporal redundancy and compact convolutional architecture
- [ffnerv_project](https://maincold2.github.io/ffnerv/): project/code surface for flow-guided pose-channel enhancer and compact grouped/pointwise convolution ideas
- [boost_nerv_official](https://github.com/Xinjie-Q/Boosting-NeRV): conditional decoder and temporal-aware affine enhancer; treat as a carrier enhancer only after receiver-byte custody
- [nervplusplus_paper](https://arxiv.org/abs/2402.18305): separable residual decoder blocks and skip-layer capacity
- [c3_paper](https://arxiv.org/abs/2312.02753): overfitted hierarchical latents and entropy-model bytes
- [c3_project](https://c3-neural-compression.github.io/): per-video small-model latent/entropy-model codec; useful for section-value-priced latent/codebook controls
- [cool_chic_docs](https://orange-opensource.github.io/Cool-Chic/encoding/architecture.html): hierarchical latent grids and autoregressive entropy model
- [nvrc_paper](https://arxiv.org/abs/2409.07414): end-to-end neural representation quantization and entropy coding
