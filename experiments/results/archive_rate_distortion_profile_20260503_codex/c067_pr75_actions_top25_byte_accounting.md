# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top25_p3_diag_20260503T0405Z/archive.zip`
- bytes: `276328`
- sha256: `2ee07ed8069b3f3cba668f509e71f20f887f501ad1d4e6dc37c23b99ac377747`
- rate contribution: `0.183995472798`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `22881`
- buffered planning bytes at unchanged distortion: `22882`
- buffered target archive bytes: `253446`
- buffered target score: `0.299998866249844`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276228`
- payload format: `public_pr75_qzs3_qp1_segactions_p3`
- payload internal overhead bytes: `10`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.999101298667 | 219471 | 0.104254756871 | 0 |
| renderer.bin | 55965 | 59288 | brotli_qzs3 | 0.037264796311 | 7.993942692553 | 55956 | 0.408844813723 | 0 |
| optimized_poses.bin | 677 | 7200 | public_qp1_brotli | 0.000450786511 | 7.659904785068 | 677 | 33.797636632201 | 0 |
| seg_tile_actions.bin | 104 | 100 | brotli_seg_tile_actions_v1 | 6.9249331e-05 | 5.506017511558 | 97 | 220.009615384615 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
