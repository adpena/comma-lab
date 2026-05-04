# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/renderer_parity_shrink_search_20260503_worker/zero_fp4_frame1_head_0.1/archive.zip`
- bytes: `275900`
- sha256: `bb8d1835303478183465e15b28fd9af8776b4ea1c07cdd6cfe0895032a267b64`
- rate contribution: `0.183710485166`
- score claim: `False`
- promotion eligible: `False`

## Container

- zip overhead bytes: `100`
- payload bytes: `275800`
- payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- payload internal overhead bytes: `12`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.999096085337 | 219471 | None | 0 |
| renderer.bin | 55513 | 59288 | brotli_qzs3 | 0.036963828065 | 7.993866141958 | 55504 | None | 0 |
| optimized_poses.qp1 | 677 | 1140 | public_qp1_brotli | 0.000450786511 | 7.673718504483 | 677 | None | 0 |
| seg_tile_actions.bin | 126 | 176 | brotli_seg_tile_actions_delta_varint_v1 | 8.3898228e-05 | 6.46724291948 | 126 | None | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.


# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/renderer_parity_shrink_search_20260503_worker/zero_fp4_all_fp4_0.1/archive.zip`
- bytes: `272069`
- sha256: `623e764ae1feeb11aba31d3be8c825ecf4e74948c8a5f5bb756ac92e8f91905a`
- rate contribution: `0.181159579517`
- score claim: `False`
- promotion eligible: `False`

## Container

- zip overhead bytes: `100`
- payload bytes: `271969`
- payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- payload internal overhead bytes: `12`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.999187046059 | 219471 | None | 0 |
| renderer.bin | 51682 | 59288 | brotli_qzs3 | 0.034412922415 | 7.993672918138 | 51674 | None | 0 |
| optimized_poses.qp1 | 677 | 1140 | public_qp1_brotli | 0.000450786511 | 7.673718504483 | 677 | None | 0 |
| seg_tile_actions.bin | 126 | 176 | brotli_seg_tile_actions_delta_varint_v1 | 8.3898228e-05 | 6.445378732955 | 126 | None | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.


# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/renderer_parity_shrink_search_20260503_worker/zero_fp4_shared_trunk_0.1/archive.zip`
- bytes: `273951`
- sha256: `9cf86ac92f3d7a97190a0abbb86b7d65277e3333b6f168ff547cd934c38c7ce9`
- rate contribution: `0.182412726067`
- score claim: `False`
- promotion eligible: `False`

## Container

- zip overhead bytes: `100`
- payload bytes: `273851`
- payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- payload internal overhead bytes: `12`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.999092093209 | 219471 | None | 0 |
| renderer.bin | 53564 | 59288 | brotli_qzs3 | 0.035666068965 | 7.993859611923 | 53556 | None | 0 |
| optimized_poses.qp1 | 677 | 1140 | public_qp1_brotli | 0.000450786511 | 7.673718504483 | 677 | None | 0 |
| seg_tile_actions.bin | 126 | 176 | brotli_seg_tile_actions_delta_varint_v1 | 8.3898228e-05 | 6.46724291948 | 126 | None | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
