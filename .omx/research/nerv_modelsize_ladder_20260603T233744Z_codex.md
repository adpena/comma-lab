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
| snerv_l4_lf2_fc4e0_decoder_int4 | levels=4, lf_bits=2, decoder_bits=4 | 11088192 | 2858676 | 1472652 | 779640 |
| snerv_l4_lf2_fc9e0_decoder_int4 | levels=4, lf_bits=2, decoder_bits=4 | 11088432 | 2858738 | 1472684 | 779657 |
| snerv_l4_lf2_fc12e4_decoder_int4 | levels=4, lf_bits=2, decoder_bits=4 | 11088768 | 2858824 | 1472728 | 779680 |
| snerv_l3_lf2_fc9e0_decoder_int4 | levels=3, lf_bits=2, decoder_bits=4 | 44265924 | 11412311 | 5879071 | 3112451 |
| snerv_l3_lf4_fc9e0_decoder_int4 | levels=3, lf_bits=4, decoder_bits=4 | 44265924 | 11412311 | 5879071 | 3112451 |
| snerv_l2_lf4_fc9e0_decoder_int4 | levels=2, lf_bits=4, decoder_bits=4 | 176976216 | 45626682 | 23504655 | 12443642 |
| snerv_l2_lf8_fc9e0_decoder_int8 | levels=2, lf_bits=8, decoder_bits=8 | 176976216 | 45626682 | 23504655 | 12443642 |
| snerv_l1_lf8_fc9e0_decoder_int8 | levels=1, lf_bits=8, decoder_bits=8 | 707817708 | 182484255 | 94007042 | 49768435 |

Marginal gates:

- `configured` `snerv_l4_lf2_fc4e0_decoder_int4` -> `snerv_l4_lf2_fc9e0_decoder_int4` needs non-rate drop >= `2.1307486499909485e-05`
- `configured` `snerv_l4_lf2_fc9e0_decoder_int4` -> `snerv_l4_lf2_fc12e4_decoder_int4` needs non-rate drop >= `2.929779393737554e-05`
- `configured` `snerv_l4_lf2_fc12e4_decoder_int4` -> `snerv_l3_lf2_fc9e0_decoder_int4` needs non-rate drop >= `1.5532778118830735`
- `configured` `snerv_l3_lf2_fc9e0_decoder_int4` -> `snerv_l3_lf4_fc9e0_decoder_int4` needs non-rate drop >= `1.8409668335921794`
- `configured` `snerv_l3_lf4_fc9e0_decoder_int4` -> `snerv_l2_lf4_fc9e0_decoder_int4` needs non-rate drop >= `11.736152910406894`
- `configured` `snerv_l2_lf4_fc9e0_decoder_int4` -> `snerv_l2_lf8_fc9e0_decoder_int8` needs non-rate drop >= `14.72775264692917`
- `configured` `snerv_l2_lf8_fc9e0_decoder_int8` -> `snerv_l1_lf8_fc9e0_decoder_int8` needs non-rate drop >= `91.12784028462114`
- `int4` `snerv_l4_lf2_fc4e0_decoder_int4` -> `snerv_l4_lf2_fc9e0_decoder_int4` needs non-rate drop >= `2.1307486499909485e-05`
- `int4` `snerv_l4_lf2_fc9e0_decoder_int4` -> `snerv_l4_lf2_fc12e4_decoder_int4` needs non-rate drop >= `2.929779393737554e-05`
- `int4` `snerv_l4_lf2_fc12e4_decoder_int4` -> `snerv_l3_lf2_fc9e0_decoder_int4` needs non-rate drop >= `2.9340029370772083`
- `int4` `snerv_l3_lf4_fc9e0_decoder_int4` -> `snerv_l2_lf4_fc9e0_decoder_int4` needs non-rate drop >= `11.736152910406894`
- `int4` `snerv_l2_lf8_fc9e0_decoder_int8` -> `snerv_l1_lf8_fc9e0_decoder_int8` needs non-rate drop >= `46.944645600434185`

## Blockers

- `hi_nerv_measured_nonrate_modelsize_ladder_missing`
- `hi_nerv_byte_closed_modelsize_ladder_missing`
- `snerv_measured_nonrate_modelsize_ladder_missing`
- `snerv_byte_closed_modelsize_ladder_missing`
- `modelsize_ladder_false_authority_no_nonrate_score`
- `archive_zip_runtime_overhead_not_in_payload_projection`
