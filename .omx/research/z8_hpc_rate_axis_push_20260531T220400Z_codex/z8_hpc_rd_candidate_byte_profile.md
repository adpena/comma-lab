# Z8/HPC Archive Byte Profile

- archive: `/Users/adpena/Projects/pact/.omx/research/z8_full_video_mlx_vjp_live_20260531T181115Z/per_subband_delta_schedule_codex/materialized_rd_waterfill_full600_max_weighted_mse_5e-5/0.bin`
- 0.bin bytes: `24,475,266`
- archive sha256: `a725297c92f164857c6e76f04bd28e268e06888c86b781610ed129d3e77601e3`
- archive.zip bytes: `24573973`
- pairs / levels: `600` / `3`
- evidence: `[macOS-CPU advisory]`; score_claim=`False`

## Sections

| section | bytes | archive % | brotli probe | entropy floor |
|---|---:|---:|---:|---:|
| `wavelet_blob` | 24,444,771 | 99.875% | 24,444,847 | 24,443,660 |
| `wyner_ziv_blob` | 22,804 | 0.093% | 2,019 | 10,637 |
| `indices_blob` | 5,400 | 0.022% | 13 | -0 |
| `meta_blob` | 2,164 | 0.009% | 715 | 1,293 |
| `z8hpc1_header` | 62 | 0.000% | 60 | 24 |
| `decoder_blob` | 34 | 0.000% | 38 | 21 |
| `dreamer_state_blob` | 31 | 0.000% | 35 | 19 |

## Wavelet Blob

- wavelet blob bytes: `24,444,771`
- pair blob compressed bytes: `24,442,367`
- top-LL raw payload inside pair blobs: `11,059,200`
- detail payload bytes inside pair blobs: `20,570,658`
- detail coefficients: `41,472,000`
- detail codec methods: `{'qi16_constriction_range': 5780, 'zigzag_u16_byteplane': 1420}`
- solid raw-pair brotli probe: `24,895,413` bytes (delta `453,046`)

## Detail Headroom

| Δ | v2 bytes | floor bytes | headroom | MSE |
|---:|---:|---:|---:|---:|
| 0.03125 | 9,392,460 | 9,389,418 | 56.0% | 8.145e-05 |
| 0.0625 | 4,538,450 | 4,536,526 | 78.8% | 2.111e-04 |
| 0.125 | 1,204,737 | 1,401,246 | 94.4% | 3.801e-04 |
| 0.25 | 293,287 | 471,717 | 98.6% | 5.448e-04 |
| 0.5 | 84,611 | 139,387 | 99.6% | 7.618e-04 |
| 1.0 | 14,944 | 25,124 | 99.9% | 9.809e-04 |

## Ranked Opportunities

1. `wavelet_blob_dominance` — `binding_rate_axis`
   - position: `ARCHIVE/PAYLOAD_BEFORE_OUTER_ZIP`
   - next: continue Z8 work only through wavelet/top-state byte reductions; other sections are not current rate binders
2. `detail_coeff_quantize_entropy_code` — `real_large_lever_but_distortion_operating_point_must_be_full_replay_gated`
   - position: `AT_DETAIL_SYMBOL_CODER`
   - next: materialize RD schedules from full-video headroom rows, replay locally, and let exact archive projection accept or reject
3. `top_ll_float_payload` — `next_binding_after_detail_collapse`
   - position: `BEFORE_DETAIL_ENTROPY_CODER`
   - next: build top-LL RD curves: delta/DC quantization, predictive top-LL from frame0, Wyner-Ziv conditional residual, and entropy-code accepted residuals
4. `solid_pair_blob_coding` — `not_a_generic_win_at_measured_quality`
   - position: `PACKET_LAYOUT / OUTER_CONTEXT`
   - next: if positive, replace independent per-pair brotli members with global section coding plus indexed seek table; otherwise keep as demoted layout branch
5. `non_wavelet_sections` — `secondary_until_wavelet_rate_axis_moves`
   - position: `ARCHIVE_CONTROL_AND_STACK_CUSTODY`
   - next: receiver-proof elision/proceduralization only for sections not consumed by runtime; do not spend primary effort here before top-LL/detail collapse
6. `outer_zip_and_repack` — `minor_unless_runtime_members_or_headers_are_large`
   - position: `AFTER_PRIMARY_PAYLOAD_ENTROPY`
   - next: run deterministic min-zip/rebrotli only after payload grammar changes; after-entropy transforms cannot fix float payload entropy
7. `contest_rate_distance` — `must_get_to_same_order_of_magnitude_before_exact_auth`
   - position: `OBJECTIVE_RATE_TERM`
   - next: gate Z8 exact-auth only after byte-closed local archive is near frontier-byte scale and MLX/CPU distortion remains plausible
