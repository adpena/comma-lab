# NeRV decoder-weight waterfill

Schema: `nerv_decoder_weight_waterfill.v1`
Family: `hi_nerv`
Authority: `false_authority_decoder_weight_waterfill_no_score_claim`

| group | action | fp32 bytes | selected bytes | delta rate | delta non-rate |
|---|---:|---:|---:|---:|---:|
| blocks.0.conv.bias | zero_rle | 128 | 2 | -0.000084 | 0.000000 |
| blocks.0.conv.weight | int4 | 13824 | 1728 | -0.008054 | 0.000045 |
| blocks.1.conv.bias | zero_rle | 128 | 2 | -0.000084 | 0.000000 |
| blocks.1.conv.weight | int4 | 9216 | 1152 | -0.005369 | 0.000031 |
| blocks.2.conv.bias | zero_rle | 128 | 2 | -0.000084 | 0.000000 |
| blocks.2.conv.weight | int4 | 9216 | 1152 | -0.005369 | 0.000090 |
| blocks.3.conv.bias | zero_rle | 128 | 2 | -0.000084 | 0.000000 |
| blocks.3.conv.weight | int4 | 9216 | 1152 | -0.005369 | 0.000060 |
| blocks.4.conv.bias | zero_rle | 128 | 2 | -0.000084 | 0.000000 |
| blocks.4.conv.weight | int4 | 9216 | 1152 | -0.005369 | 0.000076 |
| blocks.5.conv.bias | zero_rle | 128 | 2 | -0.000084 | 0.000000 |
| blocks.5.conv.weight | int4 | 9216 | 1152 | -0.005369 | 0.000102 |
| blocks.6.conv.bias | zero_rle | 128 | 2 | -0.000084 | 0.000000 |
| blocks.6.conv.weight | int4 | 9216 | 1152 | -0.005369 | 0.000134 |
| fine_injector.proj.bias | int2 | 32 | 2 | -0.000020 | 0.000000 |
| fine_injector.proj.weight | int8 | 256 | 64 | -0.000128 | 0.000000 |
| head_rgb_0.bias | int2 | 12 | 1 | -0.000007 | 0.000000 |
| head_rgb_0.weight | int4 | 864 | 108 | -0.000503 | 0.000016 |
| head_rgb_1.bias | int2 | 12 | 1 | -0.000007 | 0.000000 |
| head_rgb_1.weight | int4 | 864 | 108 | -0.000503 | 0.000017 |
| latent_embed.bias | zero_rle | 576 | 2 | -0.000382 | 0.000000 |
| latent_embed.weight | int4 | 1152 | 144 | -0.000671 | 0.000081 |
| mid_injector.proj.bias | int2 | 32 | 2 | -0.000020 | 0.000000 |
| mid_injector.proj.weight | int8 | 128 | 32 | -0.000064 | 0.000000 |

## Blockers

- `full_video_coverage_missing`
- `receiver_proof_not_satisfied`
- `archive_sha256_missing`
- `contest_cpu_cuda_exact_eval_not_executed`
