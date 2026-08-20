# NeRV decoder-weight waterfill

Schema: `nerv_decoder_weight_waterfill.v1`
Family: `hi_nerv`
Authority: `false_authority_decoder_weight_waterfill_no_score_claim`

| group | action | fp32 bytes | selected bytes | delta rate | delta non-rate |
|---|---:|---:|---:|---:|---:|
| blocks.0.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.0.conv.weight | zero_rle | 20736 | 2 | -0.013806 | 0.000000 |
| blocks.1.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.1.conv.weight | zero_rle | 20736 | 2 | -0.013806 | 0.000000 |
| blocks.2.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.2.conv.weight | zero_rle | 20736 | 2 | -0.013806 | 0.000000 |
| blocks.3.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.3.conv.weight | zero_rle | 20736 | 2 | -0.013806 | 0.000000 |
| blocks.4.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.4.conv.weight | zero_rle | 20736 | 2 | -0.013806 | 0.000000 |
| blocks.5.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.5.conv.weight | zero_rle | 20736 | 2 | -0.013806 | 0.000000 |
| blocks.6.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.6.conv.weight | zero_rle | 20736 | 2 | -0.013806 | 0.000000 |
| fine_injector.proj.bias | zero_rle | 48 | 2 | -0.000031 | 0.000000 |
| fine_injector.proj.weight | zero_rle | 384 | 2 | -0.000254 | 0.000000 |
| head_rgb_0.bias | int2 | 12 | 1 | -0.000007 | 0.000000 |
| head_rgb_0.weight | zero_rle | 1296 | 2 | -0.000862 | 0.000000 |
| head_rgb_1.bias | int2 | 12 | 1 | -0.000007 | 0.000000 |
| head_rgb_1.weight | zero_rle | 1296 | 2 | -0.000862 | 0.000000 |
| latent_embed.bias | zero_rle | 576 | 2 | -0.000382 | 0.000000 |
| latent_embed.weight | zero_rle | 1152 | 2 | -0.000766 | 0.000000 |
| mid_injector.proj.bias | zero_rle | 48 | 2 | -0.000031 | 0.000000 |
| mid_injector.proj.weight | zero_rle | 192 | 2 | -0.000127 | 0.000000 |

## Blockers

- `full_video_coverage_missing`
- `contest_cpu_cuda_exact_eval_not_executed`
