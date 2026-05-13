# Cooperative-Receiver Solver Integration

- schema: `tac_cooperative_receiver_solver_integration_v1`
- campaign_count: `14`
- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`
- meta_lagrangian_rows: `14`
- xray_grammars: `7`

| rank | campaign | predicted delta | cost | compiler/xray surface | next gate |
|---:|---|---:|---:|---|---|
| 1 | `darts_confirmed_time_traveler_config` | `-0.052500` | `$5.50` | `planning_only_no_packet_magic_yet` | DARTS config materialized + TT5L archive roundtrip + paired [contest-CUDA]/[contest-CPU] exact eval |
| 2 | `time_traveler_world_model_substrate` | `-0.033000` | `$5.50` | `TT5L` / `time_traveler_l5_packet` | TT5L archive roundtrip + macOS advisory smoke + paired exact eval |
| 3 | `sabor_boundary_only_renderer` | `-0.018000` | `$1.50` | `SBO1` / `sabor_boundary_only_renderer_packet` | SBO1 archive custody + scorer-free inflate + paired exact eval |
| 4 | `s2sbs_hf_byte_stuffing` | `-0.015000` | `$2.00` | `S2SB` / `s2sbs_byte_stuffing_packet` | S2S1 archive custody + byte-consumption proof + paired exact eval |
| 5 | `driving_prior_pretrained_renderer_2032` | `-0.008500` | `$17.50` | `planning_only_no_packet_magic_yet` | dataset/license manifest + scorer penultimate-hook readiness + deterministic renderer export + paired [contest-CUDA]/[contest-CPU] exact eval |
| 6 | `driving_prior_world_model_substrate` | `-0.007500` | `$2.00` | `DPW1` / `driving_prior_world_model_packet` | DPW1 byte-consumption proof + trained prior export + paired [contest-CUDA]/[contest-CPU] exact eval |
| 7 | `h15_coord_mlp_residual_sidecar_pr103_on_pr106` | `-0.012500` | `$2.00` | `CMLR` / `coord_mlp_residual_sidecar_packet` | sidecar bytes changed and consumed + full-frame parity baseline + paired [contest-CUDA]/[contest-CPU] exact eval |
| 8 | `s7_donoho_hnerv_wavelet_threshold` | `-0.015000` | `$0.10` | `planning_only_no_packet_magic_yet` | state_dict_parity + full-frame inflate parity + paired exact eval |
| 9 | `l2_sar_coherent_pose_spectrum` | `-0.004000` | `$0.50` | `planning_only_no_packet_magic_yet` | spectral concentration smoke + pose codec roundtrip + paired exact eval |
| 10 | `n4_video_keyed_codebook` | `-0.003000` | `$3.00` | `planning_only_no_packet_magic_yet` | deterministic codebook regeneration + archive-byte reduction + exact eval |
| 11 | `b1_atick_redlich_segmap_eigenmodes` | `-0.010000` | `$0.75` | `planning_only_no_packet_magic_yet` | basis custody + decoded mask/pose coupling proof + paired exact eval |
| 12 | `g5_mallat_scattering_decoder` | `-0.030000` | `$4.00` | `planning_only_no_packet_magic_yet` | trainer + archive grammar + runtime dependency closure + paired exact eval |
| 13 | `a1_plus_lapose_composition` | `-0.008000` | `$1.65` | `LPA1` / `a1_plus_lapose_composition_packet` | sidecar byte-consumption proof + predicted packet below 0.19 target + paired [contest-CUDA]/[contest-CPU] exact eval |
| 14 | `a1_plus_wavelet_residual_retarget` | `-0.003500` | `$0.55` | `WAV1` / `a1_plus_wavelet_residual_composition_packet` | score-sensitive wavelet atom selection + packet predicted below 0.19 + paired [contest-CUDA]/[contest-CPU] exact eval |
