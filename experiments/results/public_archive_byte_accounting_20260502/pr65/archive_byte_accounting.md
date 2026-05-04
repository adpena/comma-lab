# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/reports/raw/leaderboard_intel_20260501/pr65_archive.zip`
- bytes: `284425`
- sha256: `b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68`
- rate contribution: `0.189386932742`
- score claim: `False`
- promotion eligible: `False`

## Container

- zip overhead bytes: `100`
- payload bytes: `284325`
- payload format: `public_pr65_qpost_compact_v4`
- payload internal overhead bytes: `30`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.mkv | 219472 | 223385 | brotli_av1_obu | 0.14613739616 | 7.998591976459 | 219465 | None | 0 |
| renderer.bin | 57074 | 61590 | brotli_pr65_custom_model | 0.03800323389 | 7.995683133524 | 57064 | None | 0 |
| qpost.randmulti | 3731 | 6265 | brotli_qpost_randmulti | 0.002484319754 | 7.953992758377 | 3731 | None | 0 |
| optimized_poses.bin | 1487 | 1806 | brotli_pr65_pose | 0.000990132263 | 7.86188685951 | 1487 | None | 0 |
| qpost.post | 1400 | 2400 | brotli_qpost_post | 0.000932202534 | 7.847004918663 | 1400 | None | 0 |
| qpost.region | 273 | 603 | brotli_qpost_region | 0.000181779494 | 7.102950673806 | 273 | None | 0 |
| qpost.shift | 226 | 603 | brotli_qpost_shift | 0.000150484123 | 6.904546426986 | 221 | None | 0 |
| qpost.bias | 223 | 603 | brotli_qpost_bias | 0.000148486547 | 7.076465482783 | 223 | None | 0 |
| qpost.frac3 | 154 | 603 | brotli_qpost_frac3 | 0.000102542279 | 6.464399323712 | 151 | None | 0 |
| qpost.frac2 | 149 | 603 | brotli_qpost_frac2 | 9.9212984e-05 | 6.479620109104 | 145 | None | 0 |
| qpost.frac | 106 | 179 | brotli_qpost_frac | 7.0581049e-05 | 6.298582954522 | 105 | None | 0 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
