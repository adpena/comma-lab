# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/archive.zip`
- bytes: `276386`
- sha256: `9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a`
- rate contribution: `0.184034092618`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `2287`
- buffered planning bytes at unchanged distortion: `2288`
- buffered target archive bytes: `274098`
- buffered target score: `0.3139992066919859`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276286`
- payload format: `public_pr75_qzs3_qp1_segactions_p3`
- payload internal overhead bytes: `10`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.999101298667 | 219471 | 0.010420463658 | 0 |
| renderer.bin | 55965 | 59288 | brotli_qzs3 | 0.037264796311 | 7.993942574843 | 55956 | 0.040864826231 | 0 |
| optimized_poses.qp1 | 677 | 1140 | public_qp1_brotli | 0.000450786511 | 7.668248567183 | 677 | 3.378138847858 | 0 |
| seg_tile_actions.bin | 162 | 160 | brotli_seg_tile_actions_v1 | 0.00010786915 | 6.799666206562 | 161 | 14.117283950617 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
