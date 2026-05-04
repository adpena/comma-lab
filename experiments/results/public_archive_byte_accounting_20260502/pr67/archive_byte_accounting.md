# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/reports/raw/leaderboard_intel_20260501/pr67_archive.zip`
- bytes: `276564`
- sha256: `a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765`
- rate contribution: `0.184152615511`
- score claim: `False`
- promotion eligible: `False`

## Container

- zip overhead bytes: `100`
- payload bytes: `276464`
- payload format: `public_pr67_qzs3_qp1_fixed_slices`
- payload internal overhead bytes: `0`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.998591976459 | 219465 | None | 0 |
| renderer.bin | 56093 | 59288 | brotli_qzs3 | 0.037350026257 | 7.994011401262 | 56075 | None | 0 |
| optimized_poses.bin | 899 | 7200 | public_qp1_brotli | 0.000598607199 | 7.797470302303 | 899 | None | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
