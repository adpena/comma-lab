# Full-Pipeline Self-Compression Nextwave - 2026-05-03

Evidence grade: empirical planning only. Score claim: false. Remote dispatch: none.

## C089 Break-Even

- Archive bytes: `276342`
- Current score: `0.3154707273953505`
- Target score: `0.314`
- Bytes needed at unchanged distortion: `2209`
- Rate score per byte: `6.658589531221714e-07`

## Stream Screen

| stream | bytes | generic savings | max bytes if solo crossing |
| --- | ---: | ---: | ---: |
| masks.mkv | 219472 | 0 | 217263 |
| renderer.bin | 55965 | 0 | 53756 |
| seg_tile_actions.bin | 116 | 0 | -2093 |
| optimized_poses.qp1 | 677 | 0 | -1532 |
| payload_internal_header | 12 | 0 | -2197 |
| archive_overhead | 100 | 0 | -2109 |

## Top Recommendations

1. `renderer_trained_self_compression_c089_transplant`: Rebuild the best trained/self-compressed renderer candidate against the C089 P6/QP1 slices, then run renderer-transplant pose-safety and byte-closure preflights. Recommendation: Do not dispatch from this planner. If C089-local transplant preflight passes, claim the lane before exact CUDA eval.
2. `renderer_zero_fp4_recovery_training`: Use the byte-crossing zero-FP4 transforms as masks for a short recovery/QAT pass; rebuild only after local pose-safety passes. Recommendation: Do not exact-eval this raw shrink candidate; it is a training/repair seed.
3. `mask_exact_lossless_transcoder_target_217263`: Build a decoded-mask-lossless transcoder that proves the PR75 mask decoded SHA before packaging; target <=217263 charged mask bytes or combine with renderer savings. Recommendation: No dispatch until decoded mask SHA parity, archive byte closure, and non-noop payload proof exist.
4. `pr75_p6_action_dictionary_v2_micro_stack`: Use the P6 action candidate only as a cheap component probe or as a stack member with a larger renderer/mask byte move. Recommendation: local_cuda_exact_eval_optional_no_remote_dispatch
5. `renderer_pose_safe_micro_shrink`: Keep the largest pose-safe shrink as a regression fixture and stack only behind a larger byte move. Recommendation: do_not_dispatch_yet_safe_but_too_small
