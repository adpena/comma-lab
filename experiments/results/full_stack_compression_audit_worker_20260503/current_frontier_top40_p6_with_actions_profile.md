# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
- bytes: `276342`
- sha256: `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- rate contribution: `0.184004794824`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `2209`
- buffered planning bytes at unchanged distortion: `2210`
- buffered target archive bytes: `274132`
- buffered target score: `0.3139991791089505`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276242`
- payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- payload internal overhead bytes: `12`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.999101909606 | 219471 | 0.010065065248 | 0 |
| renderer.bin | 55965 | 59288 | brotli_qzs3 | 0.037264796311 | 7.993924528845 | 55956 | 0.039471098008 | 0 |
| optimized_poses.qp1 | 677 | 1140 | public_qp1_brotli | 0.000450786511 | 7.675934096635 | 677 | 3.262924667651 | 0 |
| seg_tile_actions.bin | 116 | 160 | brotli_seg_tile_actions_delta_varint_v1 | 7.7239639e-05 | 6.454921620109 | 116 | 19.043103448276 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
