# NeRV decoder-weight waterfill

Schema: `nerv_decoder_weight_waterfill.v1`
Family: `hi_nerv`
Authority: `false_authority_decoder_weight_waterfill_no_score_claim`

| group | action | fp32 bytes | selected bytes | delta rate | delta non-rate |
|---|---:|---:|---:|---:|---:|
| blocks.0.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.0.conv.weight | int4 | 20736 | 2592 | -0.012081 | 0.000047 |
| blocks.1.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.1.conv.weight | int4 | 20736 | 2592 | -0.012081 | 0.000070 |
| blocks.2.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.2.conv.weight | int4 | 20736 | 2592 | -0.012081 | 0.000096 |
| blocks.3.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.3.conv.weight | int4 | 20736 | 2592 | -0.012081 | 0.000116 |
| blocks.4.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.4.conv.weight | int4 | 20736 | 2592 | -0.012081 | 0.000139 |
| blocks.5.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.5.conv.weight | int4 | 20736 | 2592 | -0.012081 | 0.000185 |
| blocks.6.conv.bias | zero_rle | 192 | 2 | -0.000127 | 0.000000 |
| blocks.6.conv.weight | int4 | 20736 | 2592 | -0.012081 | 0.000211 |
| fine_injector.proj.bias | zero_rle | 48 | 2 | -0.000031 | 0.000000 |
| fine_injector.proj.weight | int8 | 384 | 96 | -0.000192 | 0.000000 |
| head_rgb_0.bias | int2 | 12 | 1 | -0.000007 | 0.000000 |
| head_rgb_0.weight | int4 | 1296 | 162 | -0.000755 | 0.000015 |
| head_rgb_1.bias | int2 | 12 | 1 | -0.000007 | 0.000000 |
| head_rgb_1.weight | int4 | 1296 | 162 | -0.000755 | 0.000018 |
| latent_embed.bias | zero_rle | 576 | 2 | -0.000382 | 0.000001 |
| latent_embed.weight | int4 | 1152 | 144 | -0.000671 | 0.000087 |
| mid_injector.proj.bias | zero_rle | 48 | 2 | -0.000031 | 0.000000 |
| mid_injector.proj.weight | int8 | 192 | 48 | -0.000096 | 0.000000 |

## Blockers

- `full_video_coverage_missing`
- `receiver_proof_not_satisfied`
- `contest_cpu_cuda_exact_eval_not_executed`
