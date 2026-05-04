# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c082_p6_stream_resweep_276333_t4_20260503T0705Z/archive.zip`
- bytes: `276333`
- sha256: `30932c684c6b09a7bd2bd248e17fd66a4bc448ed2fe5747f92e74bcceec66681`
- rate contribution: `0.183998802093`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `2234`
- buffered planning bytes at unchanged distortion: `2235`
- buffered target archive bytes: `274098`
- buffered target score: `0.31399924721650135`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276233`
- payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- payload internal overhead bytes: `12`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219465 | 223385 | brotli_av1_obu | 0.146132735147 | 7.99895679064 | 219464 | 0.010179299661 | 0 |
| renderer.bin | 55965 | 59288 | brotli_qzs3 | 0.037264796311 | 7.99397958021 | 55957 | 0.039917805771 | 0 |
| optimized_poses.qp1 | 676 | 1140 | public_qp1_brotli | 0.000450120652 | 7.646701291478 | 676 | 3.304733727811 | 0 |
| seg_tile_actions.bin | 115 | 160 | brotli_seg_tile_actions_delta_varint_v1 | 7.657378e-05 | 6.530145159584 | 115 | 19.426086956522 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
