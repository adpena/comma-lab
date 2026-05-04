# Renderer Shrink PR75/C088 Local Screen - Worker RENDERER-SHRINK - 2026-05-03

Scope: local renderer self-compression/readiness only. No remote GPU, training,
or exact-eval dispatch was performed, so no dispatch claim was required or
opened in `tools/claim_lane_dispatch.py`.

## Source Custody

- Source archive: `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/archive.zip`
- Source bytes: `276386`
- Source SHA-256: `9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a`
- Source payload layout: one stored `p` member, PR75-style self-describing body
- Logical members preserved by candidate builder:
  - `masks.mkv`: `223385` bytes, SHA-256 `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
  - `optimized_poses.qp1`: `1140` bytes, SHA-256 `1d2a6c31e836aa138bd09b3448db7e066f29f0cfcbf71b00e13357242655b583`
  - `seg_tile_actions.bin`: `160` bytes, SHA-256 `8b34ef134e6551c83d170f07c8f6672870f09df5d38551ab24320f4a2f20a373`

## Builder And Preflight Changes

- Added `experiments/build_renderer_shrink_candidate.py` as a deterministic
  local-only builder. It rewrites only `renderer.bin`, preserves every
  non-renderer logical member byte-for-byte, rebuilds the PR75 `p` payload with
  fixed zip metadata, and emits per-candidate manifests plus `summary.json`.
- Hardened `experiments/preflight_renderer_transplant_pose_safety.py` for PR75
  renderer-transplant custody:
  - Accepts either `optimized_poses.bin` or `optimized_poses.qp1`.
  - Allows unchanged PR75 auxiliary members such as `seg_tile_actions.bin`.
  - Decodes QP1 poses for runtime parity checks.
  - Fails closed when non-renderer member sets or bytes differ.

## Candidate Byte Screen

All candidates below preserve masks, QP1 poses, and tile actions byte-for-byte.
`promotion_eligible=false`; this is byte custody plus local preflight only, not
score evidence.

| Candidate | Archive bytes | Delta vs source | SHA-256 |
| --- | ---: | ---: | --- |
| `qzs3_b0033_pr75_preserved_slices` | `276349` | `-37` | `908fa78bb7f9f26cff53935a6cbd9d575dbc505cb7be5d941aa9992c87a0d086` |
| `qzs3_b0036_pr75_preserved_slices` | `276085` | `-301` | `ab90c6c78dd180f27b18d57c7529d844e3c965265a8df93ae9605c804475366d` |
| `qzs3_b0040_pr75_preserved_slices` | `275828` | `-558` | `df5a8414cc8211f6b03545c29370eae93d209882188424aac943e739e630ddc4` |
| `qzs3_b0044_pr75_preserved_slices` | `275572` | `-814` | `15417c3161d19222ec2f3ef0d1e69883e89399da58bbc9816f57e64ca1a8ea88` |
| `qzs3_b0046_pr75_preserved_slices` | `275481` | `-905` | `38dc790ca0df41af333db3785b1c0623449398ef71565c3db74450dfe6ea4f92` |
| `qzs3_b0047_pr75_preserved_slices` | `275395` | `-991` | `3623c37b5843a0433725b314c960c63d8aa7dd4a42c02111d8f01e383d0be9e6` |
| `qzs3_b0048_pr75_preserved_slices` | `275313` | `-1073` | `368387a5d360951fe4df002219d86c2fd9e6643681322743edd66ed829fdbdab` |
| `qzs3_b0064_pr75_preserved_slices` | `274580` | `-1806` | `35b7400b4fae8f64bf41b7af2c03ffe86290fe30833adf191e46ae702eb9dfbd` |
| `qzs3_b0096_pr75_preserved_slices` | `273821` | `-2565` | `fcb5945f1865a342f29c48cbe7ad1a86625e81b447256f8a35aabb5511354c64` |
| `qzs3_b0128_pr75_preserved_slices` | `273415` | `-2971` | `9d2148772ded7c21d30db0033c33c0cd951ac32a0c828a110cc44a80e0ca2978` |

Primary artifact directory:
`experiments/results/renderer_shrink_pr75_c088_20260503_worker/summary.json`

## Pose-Safety Preflight

The useful byte-save candidates fail local renderer output parity despite clean
transplant contracts. Thresholds were mean_abs_delta <= `3.0`, rms_delta <=
`8.0`, max_abs_delta <= `80.0`; sampled pair indices were `[0, 300, 599]`.

| Candidate | Contract | Safe for exact eval | mean_abs_delta | rms_delta | max_abs_delta | Failure |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `qzs3_b0047_pr75_preserved_slices` | OK | false | `7.750818252563477` | `13.035503324529826` | `202.64881896972656` | `render_output_parity_unsafe` |
| `qzs3_b0048_pr75_preserved_slices` | OK | false | `8.46407699584961` | `14.358474919083612` | `192.08627319335938` | `render_output_parity_unsafe` |
| `qzs3_b0064_pr75_preserved_slices` | OK | false | `7.192920684814453` | `11.907402723685104` | `198.67037963867188` | `render_output_parity_unsafe` |
| `qzs3_b0096_pr75_preserved_slices` | OK | false | `8.684568405151367` | `14.235236835874474` | `230.6109619140625` | `render_output_parity_unsafe` |
| `qzs3_b0128_pr75_preserved_slices` | OK | false | `8.712523460388184` | `13.976724076763078` | `222.70811462402344` | `render_output_parity_unsafe` |

Saved reports:

- `experiments/results/renderer_shrink_pr75_c088_20260503_worker/qzs3_b0047_pr75_preserved_slices/pose_safety_preflight.json`
- `experiments/results/renderer_shrink_pr75_c088_20260503_worker/qzs3_b0048_pr75_preserved_slices/pose_safety_preflight.json`
- `experiments/results/renderer_shrink_pr75_c088_20260503_worker/qzs3_b0064_pr75_preserved_slices/pose_safety_preflight.json`
- `experiments/results/renderer_shrink_pr75_c088_20260503_worker/qzs3_b0096_pr75_preserved_slices/pose_safety_preflight.json`
- `experiments/results/renderer_shrink_pr75_c088_20260503_worker/qzs3_b0128_pr75_preserved_slices/pose_safety_preflight.json`

## Decision

Do not dispatch these naive QZS3 reblock candidates to exact CUDA eval. The
archive byte savings are real and byte-closed, but the local parity gate reports
renderer-transplant pose-safety failure before any score truth run. Evidence
grade is `empirical_local_preflight_no_score`.

Highest-EV next renderer-shrink path is not another unconditioned QZS3 block
sweep. It is either:

1. Consume a trained/fixed-renderer burn artifact that already improves raw
   renderer parity, then use this builder/preflight path to preserve PR75 QP1
   and actions.
2. Add a constrained renderer encoder search objective that includes local
   output-parity loss, not just compressed renderer bytes.

No exact-eval command is recommended for the current artifacts.
