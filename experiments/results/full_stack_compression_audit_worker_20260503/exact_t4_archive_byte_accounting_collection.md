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


# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c082_qp1_p6_delta_varint_actions_stream_resweep_t4_20260503T0626Z/archive.zip`
- bytes: `276394`
- sha256: `9b78333dd39c12c986ca7fc02a4bb35a6a61ef5a1cc0c9c9c840820eb840058a`
- rate contribution: `0.184039419489`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `2237`
- buffered planning bytes at unchanged distortion: `2238`
- buffered target archive bytes: `274156`
- buffered target score: `0.3139988014182773`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276294`
- payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- payload internal overhead bytes: `12`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219465 | 223385 | brotli_av1_obu | 0.146132735147 | 7.99895679064 | 219464 | 0.010192969266 | 0 |
| renderer.bin | 55965 | 59288 | brotli_qzs3 | 0.037264796311 | 7.993982321305 | 55957 | 0.039971410703 | 0 |
| optimized_poses.qp1 | 676 | 1140 | public_qp1_brotli | 0.000450120652 | 7.639151878685 | 676 | 3.309171597633 | 0 |
| seg_tile_actions.bin | 176 | 268 | brotli_seg_tile_actions_delta_varint_v1 | 0.000117191176 | 6.804407683892 | 176 | 12.710227272727 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.


# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_lag_eval_top67_p6_t4_20260503T0608Z/archive.zip`
- bytes: `276352`
- sha256: `d73f0957cf2d8da0526c1786613443331ccb913f5648131cfaaac5e6f7eae972`
- rate contribution: `0.184011453413`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `2250`
- buffered planning bytes at unchanged distortion: `2251`
- buffered target archive bytes: `274101`
- buffered target score: `0.3139991165579473`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276252`
- payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- payload internal overhead bytes: `12`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.999101909606 | 219471 | 0.010251877233 | 0 |
| renderer.bin | 55965 | 59288 | brotli_qzs3 | 0.037264796311 | 7.993926314542 | 55956 | 0.04020369874 | 0 |
| optimized_poses.qp1 | 677 | 1140 | public_qp1_brotli | 0.000450786511 | 7.673718504483 | 677 | 3.323485967504 | 0 |
| seg_tile_actions.bin | 126 | 176 | brotli_seg_tile_actions_delta_varint_v1 | 8.3898228e-05 | 6.46724291948 | 126 | 17.857142857143 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.


# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_lag_eval_pose2_top67_p6_t4_20260503T0608Z/archive.zip`
- bytes: `276338`
- sha256: `af7a34cb1c051b1accebe2768245a44f55280e2596b315f8e4809a73a23926cd`
- rate contribution: `0.184002131388`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `2287`
- buffered planning bytes at unchanged distortion: `2288`
- buffered target archive bytes: `274050`
- buffered target score: `0.31399926093703057`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276238`
- payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- payload internal overhead bytes: `12`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.999101909606 | 219471 | 0.010420463658 | 0 |
| renderer.bin | 55965 | 59288 | brotli_qzs3 | 0.037264796311 | 7.993925437395 | 55956 | 0.040864826231 | 0 |
| optimized_poses.qp1 | 677 | 1140 | public_qp1_brotli | 0.000450786511 | 7.662436058418 | 677 | 3.378138847858 | 0 |
| seg_tile_actions.bin | 112 | 152 | brotli_seg_tile_actions_delta_varint_v1 | 7.4576203e-05 | 6.329589073805 | 112 | 20.419642857143 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.


# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_lag_eval_pose4_top67_p6_t4_20260503T0626Z/archive.zip`
- bytes: `276338`
- sha256: `de7d549ec4437b3ed9508c1f62f8a79bef88440ba3db0f4a7de5a2c83fe160ef`
- rate contribution: `0.184002131388`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `2295`
- buffered planning bytes at unchanged distortion: `2296`
- buffered target archive bytes: `274042`
- buffered target score: `0.3139987734452681`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276238`
- payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- payload internal overhead bytes: `12`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.999101909606 | 219471 | 0.010456914777 | 0 |
| renderer.bin | 55965 | 59288 | brotli_qzs3 | 0.037264796311 | 7.993925437395 | 55956 | 0.041007772715 | 0 |
| optimized_poses.qp1 | 677 | 1140 | public_qp1_brotli | 0.000450786511 | 7.676682106245 | 677 | 3.389955686854 | 0 |
| seg_tile_actions.bin | 112 | 152 | brotli_seg_tile_actions_delta_varint_v1 | 7.4576203e-05 | 6.425614855074 | 112 | 20.491071428571 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.


# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_pose_safe_positive_ampminus1_p6_t4_20260503T0632Z/archive.zip`
- bytes: `276317`
- sha256: `6e6ec4609c6da581b4113c5c06e67970542a9d8e22c4866959fe618aaff2c796`
- rate contribution: `0.18398814835`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `2346`
- buffered planning bytes at unchanged distortion: `2347`
- buffered target archive bytes: `273970`
- buffered target score: `0.31399919663273`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276217`
- payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- payload internal overhead bytes: `12`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.999101909606 | 219471 | 0.010689290661 | 0 |
| renderer.bin | 55965 | 59288 | brotli_qzs3 | 0.037264796311 | 7.993928681503 | 55956 | 0.041919056553 | 0 |
| optimized_poses.qp1 | 677 | 1140 | public_qp1_brotli | 0.000450786511 | 7.672417788795 | 677 | 3.465288035451 | 0 |
| seg_tile_actions.bin | 91 | 116 | brotli_seg_tile_actions_delta_varint_v1 | 6.0593165e-05 | 6.197193953314 | 91 | 25.78021978022 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.


# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_pr75_qp1_public_replay_t4_20260503T0608Z/archive.zip`
- bytes: `276741`
- sha256: `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd`
- rate contribution: `0.184270472546`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `2255`
- buffered planning bytes at unchanged distortion: `2256`
- buffered target archive bytes: `274485`
- buffered target score: `0.3139988034863083`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276641`
- payload format: `public_pr75_qzs3_qp1_segactions_fixed_slices`
- payload internal overhead bytes: `0`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.99911223334 | 219471 | 0.010274659182 | 0 |
| renderer.bin | 56034 | 59288 | brotli_qzs3 | 0.037310740579 | 7.994023674622 | 56026 | 0.040243423636 | 0 |
| optimized_poses.qp1 | 899 | 1140 | public_qp1_brotli | 0.000598607199 | 7.797470302303 | 899 | 2.508342602892 | 0 |
| seg_tile_actions.bin | 236 | 268 | brotli_seg_tile_actions_v1 | 0.000157142713 | 7.14252317436 | 235 | 9.555084745763 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
