# Architecture Sweep Recommendations - architecture_sweep_recommendations_20260508_codex

## Evidence Boundary

- `score_claim=false`; this is CPU/MPS research-signal planning only.
- `ready_for_exact_eval_dispatch=false`; rows cannot authorize GPU dispatch.
- Exact CUDA promotion still requires the gates listed in the JSON manifest.

## Curves

| family | curve | device | points | metric | best research row | dispatch |
|---|---|---|---:|---|---|---|
| arch_shrink | width_x0_4 | mps | 3 | proxy_loss | epoch_020 | do_not_dispatch_from_cpu_mps_research_signal |
| hnerv_arch_shrink_deltaepszeta | hnerv_arch_shrink_deltaepszeta | cpu | 4 | expected_archive_bytes | stage_c_width075_int6_qat_dez | do_not_dispatch_from_cpu_mps_research_signal |
| imp_plus_self_compress_hnerv | imp_plus_self_compress_hnerv | cpu | 1 | expected_archive_bytes | stage_e_high_risk_int4_sparse | do_not_dispatch_from_cpu_mps_research_signal |
| pr101_sparsity | post_hoc_sparsity_alpha | cpu | 7 | archive_bytes | alpha_0_9 | do_not_dispatch_from_cpu_mps_research_signal |
| renderer_bin_recompression | general_compressor_headroom | cpu | 2 | payload_bytes | brotli_q11_renderer_bin | do_not_dispatch_from_cpu_mps_research_signal |
| renderer_self_compress | c3_residual | cpu | 2 | payload_bytes | c3_residual:uniform_i4lz_bytes | do_not_dispatch_from_cpu_mps_research_signal |
| renderer_self_compress | coolchic | cpu | 2 | payload_bytes | coolchic:uniform_i4lz_bytes | do_not_dispatch_from_cpu_mps_research_signal |
| self_compress_width_precision_hnerv | self_compress_width_precision_hnerv | cpu | 1 | expected_archive_bytes | stage_d_zeta_width_precision_int4 | do_not_dispatch_from_cpu_mps_research_signal |

## Lightning State

- Active arch-shrink job `arch-shrink-x0-4-lightning-20260508T024304Z` submitted `2026-05-08T02:43:10Z` terminal_status `None`.
- Active claim `arch_shrink_x0.4_lightning` job `arch-shrink-x0-4-lightning-20260508T024304Z` status `active_dispatching` expires `2026-05-08T20:43:09Z`.

## Recommendations

- `do_not_duplicate_active_arch_shrink_lightning_dispatch`: Local Lightning state or claim ledger already has an active arch-shrink lane.
- `use_top_research_curves_for_local_build_order_only`: Cheapest normalized curves can choose CPU/MPS follow-up order, not score rank or GPU dispatch.
- `preserve_exact_cuda_promotion_gates`: Every normalized row is CPU/MPS or prediction evidence and must pass archive, compliance, lane-claim, and full CUDA auth-eval gates before score use.
