# NeRV decoder-weight waterfill

Schema: `nerv_decoder_weight_waterfill.v1`
Family: `hi_nerv`
Authority: `false_authority_decoder_weight_waterfill_no_score_claim`

| group | action | fp32 bytes | selected bytes | delta rate | delta non-rate |
|---|---:|---:|---:|---:|---:|
| blocks.0.conv.bias | fp32_protect | 192 | 192 | 0.000000 | 0.000000 |
| blocks.0.conv.weight | fp32_protect | 20736 | 20736 | 0.000000 | 0.000000 |
| blocks.1.conv.bias | fp32_protect | 192 | 192 | 0.000000 | 0.000000 |
| blocks.1.conv.weight | fp32_protect | 20736 | 20736 | 0.000000 | 0.000000 |
| blocks.2.conv.bias | fp32_protect | 192 | 192 | 0.000000 | 0.000000 |
| blocks.2.conv.weight | fp32_protect | 20736 | 20736 | 0.000000 | 0.000000 |
| blocks.3.conv.bias | fp32_protect | 192 | 192 | 0.000000 | 0.000000 |
| blocks.3.conv.weight | fp32_protect | 20736 | 20736 | 0.000000 | 0.000000 |
| blocks.4.conv.bias | fp32_protect | 192 | 192 | 0.000000 | 0.000000 |
| blocks.4.conv.weight | fp32_protect | 20736 | 20736 | 0.000000 | 0.000000 |
| blocks.5.conv.bias | fp32_protect | 192 | 192 | 0.000000 | 0.000000 |
| blocks.5.conv.weight | fp32_protect | 20736 | 20736 | 0.000000 | 0.000000 |
| blocks.6.conv.bias | fp32_protect | 192 | 192 | 0.000000 | 0.000000 |
| blocks.6.conv.weight | fp32_protect | 20736 | 20736 | 0.000000 | 0.000000 |
| fine_injector.proj.bias | fp32_protect | 48 | 48 | 0.000000 | 0.000000 |
| fine_injector.proj.weight | fp32_protect | 384 | 384 | 0.000000 | 0.000000 |
| head_rgb_0.bias | fp32_protect | 12 | 12 | 0.000000 | 0.000000 |
| head_rgb_0.weight | fp32_protect | 1296 | 1296 | 0.000000 | 0.000000 |
| head_rgb_1.bias | fp32_protect | 12 | 12 | 0.000000 | 0.000000 |
| head_rgb_1.weight | fp32_protect | 1296 | 1296 | 0.000000 | 0.000000 |
| latent_embed.bias | fp32_protect | 576 | 576 | 0.000000 | 0.000000 |
| latent_embed.weight | fp32_protect | 1152 | 1152 | 0.000000 | 0.000000 |
| mid_injector.proj.bias | fp32_protect | 48 | 48 | 0.000000 | 0.000000 |
| mid_injector.proj.weight | fp32_protect | 192 | 192 | 0.000000 | 0.000000 |

## Blockers

- `full_video_coverage_missing`
- `contest_cpu_cuda_exact_eval_not_executed`
