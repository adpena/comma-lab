# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/pmg_hotspot_candidate_c067_atomtop4068_20260502/archive.zip`
- bytes: `195762`
- sha256: `2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497`
- rate contribution: `0.130349880381`
- score claim: `False`
- promotion eligible: `False`

## Container

- zip overhead bytes: `100`
- payload bytes: `195657`
- payload format: `None`
- payload internal overhead bytes: `877`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.cmg3 | 134352 | 134352 | raw | 0.08945948207 | 7.997925889544 | 134349 | None | 200 |
| renderer.bin | 59288 | 59288 | raw | 0.039477445613 | 7.861381999313 | 59182 | None | 3323 |
| optimized_poses.bin | 1140 | 7200 | pose_qp1_v1 | 0.000759079207 | 5.229017915936 | 1071 | None | 463 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
