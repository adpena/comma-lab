# promotion accounting

This table tracks every score the lab has promoted to the live floor, across both eras. The Era 2 entries are the live frontier; the Era 1 entries are preserved for the writeup arc.

## Era 2 — neural renderer (contest-CUDA, current floor)

| Run | Date | Lane | Recipe | Archive bytes | Score | Source |
|---|---|---|---|---:|---:|---|
| `dilated_h64_baseline_crf50` | 2026-04-25 | baseline | dilated-h64 renderer + CRF=50 + matched poses | ~293,000 | **0.90** | `contest-CUDA` (Vast.ai 4090) |
| `lane_a_landed` | 2026-04-27 | Lane A | baseline + pose TTO from baseline poses (rank-1 init) | 694,074 | **1.15** | `contest-CUDA` |
| `lane_g_v3_landed` | 2026-04-28 | Lane G v3 | Lane A + KL distill weight=0.002 + pose TTO retry | 694,074 | **1.05** | `contest-CUDA` |
| `modal_auth_eval_9b20bdfca246` | 2026-04-29 | Lane G v3 (Modal repro) | identical archive bytes | 694,074 | **1.04** | `Modal-T4-CUDA` |

The 0.90 → 1.05 trajectory is driven by Lane A's pose-TTO + Lane G v3's KL-distill weight=0.002. The 1.15 entry sits between the two because Lane A introduced a 401KB rate increase from the larger pose tensor; Lane G v3 keeps the rate flat and improves both PoseNet and SegNet.

## Era 1 — codec + tiny CNN post-filter (Track B, historical)

| Run | Scale | Filters | Current workflow score | Current workflow bytes | Rule-faithful estimate | Rule-faithful bytes |
|---|---:|---|---:|---:|---:|---:|
| `robust_current-medium23-cpu-2026-04-03` | 512x384 | lanczos/bicubic | 3.62 | 2,819,374 | 3.618 | 2,822,418 |
| `robust_current-448x336-medium23-cpu-2026-04-03` | 448x336 | lanczos/bicubic | 3.56 | 1,978,141 | 3.563 | 1,981,185 |
| `robust_current-keyint48-cpu-2026-04-04` | 448x336 | lanczos/bicubic | 3.56 | 1,901,606 | 3.562 | 1,904,650 |
| `robust_current-lanczos-lanczos-cpu-2026-04-04` | 448x336 | lanczos/lanczos | 3.54 | 1,901,606 | 3.546 | 1,904,650 |
| `robust_current-432x324-cpu-2026-04-04` | 432x324 | lanczos/lanczos | 3.33 | 1,781,129 | 3.330 | 1,787,266 |
| `robust_current-cand-424x318-crf23-g48-b4-r4-cpu-2026-04-05` | 424x318 | lanczos/lanczos | 3.25 | 1,669,984 | 3.275 | 1,704,163 |
| `robust_current-av1-524x394-rgb24-promoted-cpu-2026-04-05` | 524x394 | lanczos/bicubic | 2.20 | 920,457 | 2.228 | 956,424 |
| `robust_current-av1-524x394-crf34-promoted-cpu-2026-04-06` | 524x394 | lanczos/bicubic | 2.19 | 864,455 | 2.215 | 900,954 |
| `robust_current-av1-524x394-upscale-lanczos-promoted-cpu-2026-04-06` | 524x394 | lanczos/lanczos | 2.18 | 864,455 | 2.196 | 892,472 |
| `robust_current-av1-524x394-colorspace-hardening-promoted-cpu-2026-04-06` | 524x394 | lanczos/lanczos + explicit bt709/tv->rgb(pc) | 2.12 | 864,486 | 2.142 | 897,745 |
| `robust_current-sharpness1-promoted-cpu-2026-04-06` | 524x394 | lanczos/lanczos | 2.08 | 864,168 | 2.124 | 922,416 |
| `robust_current-av1-522x392-postfilter-promoted-cpu-2026-04-07` | 522x392 | lanczos/learned-postfilter | 2.05 | 861,986 | 2.078 | 896,432 |
| `robust_current-long500-h16-promoted-cpu-2026-04-08` | 524x394 | lanczos/long500-qat-ema-postfilter-h16 | 1.99 | 864,167 | 2.027 | 926,090 |
| `robust_current-long500-h32-promoted-cpu-2026-04-08` | 524x394 | lanczos/long500-qat-ema-postfilter-h32 | 1.95 | 864,168 | 1.992 | 935,149 |
| `robust_current-long1000-h16-promoted-cpu-2026-04-08` | 524x394 | lanczos/long1000-qat-ema-postfilter-h16 | 1.92 | 864,167 | 1.964 | 927,230 |
| `robust_current-long1000-h32-promoted-cpu-2026-04-08` | 524x394 | lanczos/long1000-qat-ema-postfilter-h32 | 1.85 | 864,167 | 1.893 | 935,166 |
| `robust_current-ensemble-h32-mc75-25-promoted-cpu-2026-04-09` | 524x394 | lanczos/ensemble-postfilter-h32-mc75-25 | 1.84 | 864,168 | 1.889 | 936,886 |
| `robust_current-long1000-h64-promoted-cpu-2026-04-09` | 524x394 | lanczos/long1000-qat-ema-postfilter-h64 | 1.73 | 864,167 | 1.795 | 966,071 |

## leaderboard context (live)

| Rank | Score | Entry | Notes |
|------|------:|-------|-------|
| 1 | 0.33 | Quantizr | FiLM CNN 88K + KL-T2 + AV1 |
| 2 | 0.38 | Selfcomp | self-compression ~1.017 bpw + analytical-pose affine |
| 3 | 0.60 | Mask2mask | obfuscated arch |
| ours | 1.05 | Lane G v3 | not submitted; would rank ~4th |

Sub-0.30 is the live target. None of the live work has produced a contest-CUDA score below 1.05 yet — every prediction in `next_experiments.md` is advisory only.
