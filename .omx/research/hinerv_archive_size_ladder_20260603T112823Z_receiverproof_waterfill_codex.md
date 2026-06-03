# HiNeRV archive-size ladder

Schema: `hinerv_archive_size_ladder.v1`
Authority: `false_authority_archive_size_ladder_no_score_claim`
Axis: `[planning/control]`
Decoder codec: `int8_mixed`
Decoder codec policy: `modelsize_budget_candidate_decoder_codec_overrides_top_level_default`
Modelsize budget schema: `nerv_modelsize_budget.v1`

| row | params | nominal bytes | archive bytes | measured-minus-nominal | rate score [planning/control] | proof ready |
|---|---:|---:|---:|---:|---:|---:|
| hinerv_np600_ld4_ed32_dc4_cnx_int2_mixed_ceil36000_tgtmp0p02 | 20022 | 19706 | 45834 | 26128 | 0.030519 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_int2_mixed_ceil216000_tgtmp0p02 | 20022 | 19706 | 45933 | 26227 | 0.030585 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_int2_mixed_ceil285000_tgtmp0p02 | 20022 | 19706 | 46055 | 26349 | 0.030666 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_int4_mixed_ceil285000_tgtmp0p02 | 20022 | 22611 | 49109 | 26498 | 0.032700 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_int4_mixed_ceil216000_tgtmp0p02 | 20022 | 22611 | 49197 | 26586 | 0.032758 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_int4_mixed_ceil36000_tgtmp0p02 | 20022 | 22611 | 49234 | 26623 | 0.032783 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil216000_tgtmp0p02 | 20022 | 28422 | 54432 | 26010 | 0.036244 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_portfolio_auto_ceil285000_tgtmp0p02 | 20022 | 28422 | 54441 | 26019 | 0.036250 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_portfolio_auto_ceil216000_tgtmp0p02 | 20022 | 28422 | 54456 | 26034 | 0.036260 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil285000_tgtmp0p02 | 20022 | 28422 | 54458 | 26036 | 0.036261 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_portfolio_auto_ceil36000_tgtmp0p02 | 20022 | 28422 | 54461 | 26039 | 0.036263 | True |
| hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil36000_tgtmp0p02 | 20022 | 28422 | 54466 | 26044 | 0.036267 | True |
| hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000 | 26886 | 35286 | 61056 | 25770 | 0.040655 | True |
| hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000 | 26886 | 35286 | 61112 | 25826 | 0.040692 | True |
| hinerv_np600_ld4_ed24_dc24_hfg_int8_mixed_ceil216000 | 206814 | 215214 | 247142 | 31928 | 0.164562 | True |
| hinerv_np600_ld4_ed24_dc24_hfg_portfolio_auto_ceil216000 | 206814 | 215214 | 247162 | 31948 | 0.164575 | True |
| hinerv_np600_ld4_ed32_dc32_portfolio_auto_ceil285000 | 270678 | 279078 | 304383 | 25305 | 0.202676 | True |
| hinerv_np600_ld4_ed32_dc32_int8_mixed_ceil285000 | 270678 | 279078 | 304403 | 25325 | 0.202689 | True |

## Marginal Gates

- `hinerv_np600_ld4_ed32_dc4_cnx_int2_mixed_ceil36000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_int2_mixed_ceil216000_tgtmp0p02` adds `99` B; requires non-rate drop >= `6.592003635909497e-05`
- `hinerv_np600_ld4_ed32_dc4_cnx_int2_mixed_ceil216000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_int2_mixed_ceil285000_tgtmp0p02` adds `122` B; requires non-rate drop >= `8.12347922809049e-05`
- `hinerv_np600_ld4_ed32_dc4_cnx_int2_mixed_ceil285000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_int4_mixed_ceil285000_tgtmp0p02` adds `3054` B; requires non-rate drop >= `0.0020335332428351115`
- `hinerv_np600_ld4_ed32_dc4_cnx_int4_mixed_ceil285000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_int4_mixed_ceil216000_tgtmp0p02` adds `88` B; requires non-rate drop >= `5.859558787475108e-05`
- `hinerv_np600_ld4_ed32_dc4_cnx_int4_mixed_ceil216000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_int4_mixed_ceil36000_tgtmp0p02` adds `37` B; requires non-rate drop >= `2.4636781265520342e-05`
- `hinerv_np600_ld4_ed32_dc4_cnx_int4_mixed_ceil36000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil216000_tgtmp0p02` adds `5198` B; requires non-rate drop >= `0.003461134838329047`
- `hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil216000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_portfolio_auto_ceil285000_tgtmp0p02` adds `9` B; requires non-rate drop >= `5.9927305780995425e-06`
- `hinerv_np600_ld4_ed32_dc4_cnx_portfolio_auto_ceil285000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_portfolio_auto_ceil216000_tgtmp0p02` adds `15` B; requires non-rate drop >= `9.98788429683257e-06`
- `hinerv_np600_ld4_ed32_dc4_cnx_portfolio_auto_ceil216000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil285000_tgtmp0p02` adds `2` B; requires non-rate drop >= `1.3317179062443428e-06`
- `hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil285000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_portfolio_auto_ceil36000_tgtmp0p02` adds `3` B; requires non-rate drop >= `1.9975768593665143e-06`
- `hinerv_np600_ld4_ed32_dc4_cnx_portfolio_auto_ceil36000_tgtmp0p02` -> `hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil36000_tgtmp0p02` adds `5` B; requires non-rate drop >= `3.329294765610857e-06`
- `hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil36000_tgtmp0p02` -> `hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000` adds `6590` B; requires non-rate drop >= `0.00438801050107511`
- `hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000` -> `hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000` adds `56` B; requires non-rate drop >= `3.72881013748416e-05`
- `hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000` -> `hinerv_np600_ld4_ed24_dc24_hfg_int8_mixed_ceil216000` adds `186030` B; requires non-rate drop >= `0.12386974104931754`
- `hinerv_np600_ld4_ed24_dc24_hfg_int8_mixed_ceil216000` -> `hinerv_np600_ld4_ed24_dc24_hfg_portfolio_auto_ceil216000` adds `20` B; requires non-rate drop >= `1.3317179062443427e-05`
- `hinerv_np600_ld4_ed24_dc24_hfg_portfolio_auto_ceil216000` -> `hinerv_np600_ld4_ed32_dc32_portfolio_auto_ceil285000` adds `57221` B; requires non-rate drop >= `0.03810111515660377`
- `hinerv_np600_ld4_ed32_dc32_portfolio_auto_ceil285000` -> `hinerv_np600_ld4_ed32_dc32_int8_mixed_ceil285000` adds `20` B; requires non-rate drop >= `1.3317179062443427e-05`

## Blockers

- `hinerv_archive_size_ladder_false_authority_no_nonrate_score`
- `contest_cpu_cuda_exact_eval_not_executed`
