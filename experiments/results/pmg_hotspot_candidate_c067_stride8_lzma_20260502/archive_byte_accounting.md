# Archive Byte Accounting

This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.

## Archive

- path: `/Users/adpena/Projects/pact/experiments/results/pmg_hotspot_candidate_c067_stride8_lzma_20260502/archive.zip`
- bytes: `102456`
- sha256: `48e70211fb621ae7919924b5eb57002473e1834dd7fbf758d673003597c1608a`
- rate contribution: `0.068221244901`
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
- payload bytes: `107164`
- payload format: `None`
- payload internal overhead bytes: `876`

## Streams

| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| renderer.bin | 59288 | 59288 | raw | 0.039477445613 | 7.861381999313 | 59182 | 0.395594386722 | 3323 |
| masks.cmg3 | 45860 | 45860 | raw | 0.03053629159 | 7.99192080435 | 45853 | 0.511426079372 | 619 |
| optimized_poses.bin | 1140 | 7200 | pose_qp1_v1 | 0.000759079207 | 5.229017915936 | 1071 | 20.573684210526 | 463 |

## Dispatch Interpretation

- Direct nested compression is exhausted for the current byte streams.
- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.
- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.
- Pose bytes are already too small to close the sub-0.300 gap alone.
- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.
