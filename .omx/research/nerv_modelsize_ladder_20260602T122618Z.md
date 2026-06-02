# NeRV model-size ladder

Schema: `nerv_modelsize_ladder.v1`
Authority: `false_authority_modelsize_ladder_no_score_claim`
Contest byte price: `6.658589531221714e-07`

## hi_nerv

| row | main control | fp32 bytes | int8 bytes | int4 bytes | int2 bytes |
|---|---|---:|---:|---:|---:|
| hi_nerv_local_tiny | modelsize_scale=0.5 | 379056 | 97730 | 50348 | 26658 |
| hi_nerv_local_small | modelsize_scale=0.75 | 795492 | 205091 | 105655 | 55937 |
| hi_nerv_local_base | modelsize_scale=1 | 1363208 | 351456 | 181055 | 95855 |
| hi_nerv_local_wide | modelsize_scale=1.5 | 2952480 | 761190 | 392130 | 207601 |

Marginal gates:

- `int8` `hi_nerv_local_tiny` -> `hi_nerv_local_small` needs non-rate drop >= `0.07148728306614945`
- `int8` `hi_nerv_local_small` -> `hi_nerv_local_base` needs non-rate drop >= `0.09745844567372662`
- `int8` `hi_nerv_local_base` -> `hi_nerv_local_wide` needs non-rate drop >= `0.2728250522985598`
- `int4` `hi_nerv_local_tiny` -> `hi_nerv_local_small` needs non-rate drop >= `0.03682666112032793`
- `int4` `hi_nerv_local_small` -> `hi_nerv_local_base` needs non-rate drop >= `0.050205765065411724`
- `int4` `hi_nerv_local_base` -> `hi_nerv_local_wide` needs non-rate drop >= `0.14054617853026233`
- `int2` `hi_nerv_local_tiny` -> `hi_nerv_local_small` needs non-rate drop >= `0.019495684288464057`
- `int2` `hi_nerv_local_small` -> `hi_nerv_local_base` needs non-rate drop >= `0.026579757690730836`
- `int2` `hi_nerv_local_base` -> `hi_nerv_local_wide` needs non-rate drop >= `0.07440707457559016`

## snerv

| row | main control | fp32 bytes | int8 bytes | int4 bytes | int2 bytes |
|---|---|---:|---:|---:|---:|
| snerv_l4_lf2_decoder_int4 | levels=4, lf_bits=2, decoder_bits=4 | 11089296 | 2858962 | 1472800 | 779719 |
| snerv_l3_lf2_decoder_int4 | levels=3, lf_bits=2, decoder_bits=4 | 44266572 | 11412477 | 5879156 | 3112495 |
| snerv_l3_lf4_decoder_int4 | levels=3, lf_bits=4, decoder_bits=4 | 44266572 | 11412477 | 5879156 | 3112495 |
| snerv_l2_lf4_decoder_int4 | levels=2, lf_bits=4, decoder_bits=4 | 176976648 | 45626794 | 23504713 | 12443673 |
| snerv_l2_lf8_decoder_int8 | levels=2, lf_bits=8, decoder_bits=8 | 176976648 | 45626794 | 23504713 | 12443673 |
| snerv_l1_lf8_decoder_int8 | levels=1, lf_bits=8, decoder_bits=8 | 707817924 | 182484311 | 94007071 | 49768451 |

Marginal gates:

- `configured` `snerv_l4_lf2_decoder_int4` -> `snerv_l3_lf2_decoder_int4` needs non-rate drop >= `1.553286468049464`
- `configured` `snerv_l3_lf2_decoder_int4` -> `snerv_l3_lf4_decoder_int4` needs non-rate drop >= `1.8409668335921794`
- `configured` `snerv_l3_lf4_decoder_int4` -> `snerv_l2_lf4_decoder_int4` needs non-rate drop >= `11.73613493221516`
- `configured` `snerv_l2_lf4_decoder_int4` -> `snerv_l2_lf8_decoder_int8` needs non-rate drop >= `14.72778860331264`
- `configured` `snerv_l2_lf8_decoder_int8` -> `snerv_l1_lf8_decoder_int8` needs non-rate drop >= `91.12780299651978`
- `int4` `snerv_l4_lf2_decoder_int4` -> `snerv_l3_lf2_decoder_int4` needs non-rate drop >= `2.9340115932435986`
- `int4` `snerv_l3_lf4_decoder_int4` -> `snerv_l2_lf4_decoder_int4` needs non-rate drop >= `11.73613493221516`
- `int4` `snerv_l2_lf8_decoder_int8` -> `snerv_l1_lf8_decoder_int8` needs non-rate drop >= `46.944626290524546`

## Blockers

- `hi_nerv_measured_nonrate_modelsize_ladder_missing`
- `hi_nerv_byte_closed_modelsize_ladder_missing`
- `snerv_measured_nonrate_modelsize_ladder_missing`
- `snerv_byte_closed_modelsize_ladder_missing`
- `modelsize_ladder_false_authority_no_nonrate_score`
- `archive_zip_runtime_overhead_not_in_payload_projection`
