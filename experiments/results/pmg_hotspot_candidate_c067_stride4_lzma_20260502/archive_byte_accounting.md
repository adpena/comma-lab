# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/pmg_hotspot_candidate_c067_stride4_lzma_20260502/archive.zip`
- bytes: `136967`
- sha256: `145c7c2358badac72f53c396ba3a456ebe8f35e7cdc88514289cc58df3d5cad4`
- rate contribution: `0.091200703232`
- score claim: `False`
- promotion eligible: `False`

## Sub-0.300 Byte Gap

- exact crossing bytes at unchanged distortion: `23454`
- buffered planning bytes at unchanged distortion: `23455`
- buffered target archive bytes: `252759`
- buffered target score: `0.2999993090390018`
- scored reference matches profiled archive: `False`
- reference warning: `target_gap was computed from eval_json's scored archive, which does not match the profiled archive bytes/SHA; treat it as a reference gap only`

## Container

- zip overhead bytes: `100`
- payload bytes: `136862`
- payload format: `None`
- payload internal overhead bytes: `876`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masks.cmg3 | 75558 | 75558 | raw | 0.05031097078 | 7.996278643931 | 75554 | 0.310410545541 | 232 |
| renderer.bin | 59288 | 59288 | raw | 0.039477445613 | 7.861381999313 | 59182 | 0.395594386722 | 3323 |
| optimized_poses.bin | 1140 | 7200 | pose_qp1_v1 | 0.000759079207 | 5.229017915936 | 1071 | 20.573684210526 | 463 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
