# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- bytes: `276214`
- sha256: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- rate contribution: `0.183919564878`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `23454`
- buffered planning bytes at unchanged distortion: `23455`
- buffered target archive bytes: `252759`
- buffered target score: `0.2999993090390018`
- scored reference matches profiled archive: `True`

## Container

- zip overhead bytes: `100`
- payload bytes: `276114`
- payload format: `public_pr67_qzs3_qp1_fixed_slices`
- payload internal overhead bytes: `0`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.998591976459 | 219465 | 0.106865568273 | 0 |
| renderer.bin | 55965 | 59288 | brotli_qzs3 | 0.037264796311 | 7.993832581821 | 55951 | 0.419083355669 | 0 |
| optimized_poses.bin | 677 | 7200 | public_qp1_brotli | 0.000450786511 | 7.673436926071 | 677 | 34.644017725258 | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
