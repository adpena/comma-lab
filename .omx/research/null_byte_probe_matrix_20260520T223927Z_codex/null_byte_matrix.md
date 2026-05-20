# Null-byte master-gradient probe matrix (11 anchors)

- Generated UTC: 2026-05-20T22:40:27.763602+00:00
- Anchors scanned: 11
- Probed OK: 11
- Epsilon: 1e-09
- Axis tag: [predicted] (observability-only per Catalog #323)

## Per-anchor results

| # | substrate | codec_family | axis | hardware | n_pairs | n_bytes | n_null | null_frac | seg_zero | pose_zero | rate_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | fec6_subject_sha | hnerv_family | [contest-CPU] | darwin_arm64_m5_max_macos_cpu_advisory | None | 178417 | 0 | 0.00% | 16638 | 16292 | 0 |
| 2 | fec6_subject_sha | hnerv_family | [macOS-CPU advisory] | darwin_arm64_m5_max_macos_cpu_advisory | 8 | 178417 | 0 | 0.00% | 16638 | 16292 | 0 |
| 3 | a1_finetuned | hnerv_family | [macOS-CPU advisory] | darwin_arm64_m5_max_macos_cpu_advisory | 8 | 178162 | 16037 | 9.00% | 16383 | 16037 | 178162 |
| 4 | pr101_lc_v2 | hnerv_family | [macOS-CPU advisory] | darwin_arm64_m5_max_macos_cpu_advisory | 8 | 178158 | 16033 | 9.00% | 16379 | 16033 | 178158 |
| 5 | pr101_fec6_frontier | hnerv_family | [macOS-CPU advisory] | darwin_arm64_m5_max_macos_cpu_advisory | 8 | 178417 | 16292 | 9.13% | 16638 | 16292 | 178417 |
| 6 | pr101_lc_v2 | hnerv_family | [macOS-CPU advisory] | darwin_arm64_m5_max_macos_cpu_advisory | 8 | 178158 | 16033 | 9.00% | 16379 | 16033 | 178158 |
| 7 | a1_finetuned | hnerv_family | [macOS-CPU advisory] | darwin_arm64_local_cpu_advisory | 8 | 178162 | 16037 | 9.00% | 16383 | 16037 | 178162 |
| 8 | pr101_fec6_frontier | hnerv_family | [macOS-CPU advisory] | darwin_arm64_local_cpu_advisory | 8 | 178417 | 16292 | 9.13% | 16638 | 16292 | 178417 |
| 9 | pr106_format0d | pr106_format0d_family | [macOS-CPU advisory] | darwin_arm64_local_cpu_advisory | 8 | 186776 | 16909 | 9.05% | 17273 | 16909 | 186776 |
| 10 | pr107_apogee | pr107_apogee_family | [macOS-CPU advisory] | darwin_arm64_local_cpu_advisory | 8 | 178284 | 15987 | 8.97% | 16335 | 15987 | 178284 |
| 11 | pr101_fec6_frontier | hnerv_family | [contest-CUDA] | linux_x86_64_t4_modal | 600 | 178417 | 16292 | 9.13% | 16638 | 16292 | 178417 |

## Codec-family rollups

| family | n_anchors | null_frac_mean | null_frac_stddev | null_frac_min | null_frac_max | null_bytes_total |
|---|---|---|---|---|---|---|
| hnerv_family | 9 | 7.04% | 3.9940% | 0.00% | 9.13% | 113016 |
| pr106_format0d_family | 1 | 9.05% | 0.0000% | 9.05% | 9.05% | 16909 |
| pr107_apogee_family | 1 | 8.97% | 0.0000% | 8.97% | 8.97% | 15987 |

## Cross-hardware drift per archive

| substrate | axes_present | per_axis_mean_null_fraction | abs_spread | rel_spread |
|---|---|---|---|---|
| fec6_subject_sha | [contest-CPU], [macOS-CPU advisory] | [contest-CPU]=0.00%, [macOS-CPU advisory]=0.00% | 0.0000pp | 0.00% |
| pr101_fec6_frontier | [contest-CUDA], [macOS-CPU advisory] | [macOS-CPU advisory]=9.13%, [contest-CUDA]=9.13% | 0.0000pp | 0.00% |

## Top-5 replacement candidates (deduplicated per substrate)

| rank | substrate | family | n_null_bytes | null_frac | ΔS@K=16 | ΔS@K=32 | ΔS@K=64 | ΔS@K=128 | ΔS@K=256 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | pr106_format0d | pr106_format0d_family | 16909 | 9.05% | -0.011248 | -0.011238 | -0.011216 | -0.011174 | -0.011089 |
| 2 | pr101_fec6_frontier | hnerv_family | 16292 | 9.13% | -0.010838 | -0.010827 | -0.010806 | -0.010763 | -0.010678 |
| 3 | a1_finetuned | hnerv_family | 16037 | 9.00% | -0.010668 | -0.010657 | -0.010636 | -0.010593 | -0.010508 |
| 4 | pr101_lc_v2 | hnerv_family | 16033 | 9.00% | -0.010665 | -0.010654 | -0.010633 | -0.010590 | -0.010505 |
| 5 | pr107_apogee | pr107_apogee_family | 15987 | 8.97% | -0.010634 | -0.010624 | -0.010602 | -0.010560 | -0.010475 |

## Provenance (per Catalog #323)

- `score_claim`: False
- `promotion_eligible`: False
- `rank_or_kill_eligible`: False
- `promotable`: False
- `axis_tag`: `[predicted]`
- All ΔS predictions assume substituting null bytes with a K-byte PRNG seed
  + the inflate-side deterministic codebook regeneration; actual contest-CUDA
  measurement still required per CLAUDE.md "Apples-to-apples evidence discipline".
