# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/pmg_hotspot_candidate_c067_20260502/archive.zip`
- bytes: `187144`
- sha256: `3ab4d0c85ae15325b61b7838c375383a460e7ad1b911b677101fda50e2611c53`
- rate contribution: `0.124611507923`
- score claim: `False`
- promotion eligible: `False`

## Container

- zip overhead bytes: `100`
- payload bytes: `187039`
- payload format: `None`
- payload internal overhead bytes: `877`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.cmg3 | 125734 | 125734 | raw | 0.083721109612 | 7.992436447016 | 125731 | None | 427 |
| renderer.bin | 59288 | 59288 | raw | 0.039477445613 | 7.861381999313 | 59182 | None | 3323 |
| optimized_poses.bin | 1140 | 7200 | pose_qp1_v1 | 0.000759079207 | 5.229017915936 | 1071 | None | 463 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
