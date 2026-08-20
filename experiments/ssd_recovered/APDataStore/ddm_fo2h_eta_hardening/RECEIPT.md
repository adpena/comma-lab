# ddm_fo2h — RECEIPT

UTC 2026-08-17T17:39:44Z · git `eff98477134164e9a735c9c42f1b43bf52f0a1e4` · host Primary

Axis `[macOS-CPU advisory]` · `score_claim=false` · `promotable=false` · pointer UNMOVED.

All paths relative to `/Volumes/APDataStore/pact/ddm_fo2h_eta_hardening/`.

## verdict + sweep artifacts

| artifact | bytes | sha256 |
|---|---:|---|
| `FO2H_ETA_ADJUDICATION.json` | 46,891 | `7e154b2c7a4cda3c2d1e472954bfc4ff` |
| `FO2H_SAMPLE.json` | 955 | `3ac20abf39f5b8b0e7636abcf31f2fd8` |
| `FO2H_WATERFILL_MEASURED.json` | 164,916 | `b9dd1a1adcac9275e643c0bed625afe3` |
| `FO2H_WATERFILL_MEASURED.run1_coarse.json` | 66,641 | `2514b3669de7e4d1c381dae37bfca99f` |
| `FO2H_WATERFILL_MEASURED.run2.json` | 164,916 | `f4c321e63fdf1ec1cc4a288c05d7a9b8` |
| `FO2H_WATERFILL_RERANK.json` | 58,891 | `416d4459cba95a3f4ef5566d509f1f99` |
| `PROGRESS.jsonl` | 116,544 | `98b11a038129189c406026ce6ee2331f` |

## LEG 1 -- eta gate rows and receipts (projected arm shards + matched unprojected arm)

| artifact | bytes | sha256 |
|---|---:|---|
| `null_shardA/ETA_GATE_ROWS.jsonl` | 13,346 | `b1485992ddcfff4a79a0f8422ef978e2` |
| `null_shardA/ETA_GATE_VERDICT.json` | 19,229 | `30ac2381b7a4e5fc1ff300b0f2b90a62` |
| `null_shardB/ETA_GATE_ROWS.jsonl` | 13,306 | `1c0fc5ed2bf47697c80ea26d95481708` |
| `null_shardB/ETA_GATE_VERDICT.json` | 19,190 | `91e7296530e3a4e60832a03fbfc022a3` |
| `free_matched16/ETA_GATE_ROWS.jsonl` | 8,888 | `c5a96514e60a7ee7789eba3a43c15af3` |
| `free_matched16/ETA_GATE_VERDICT.json` | 13,428 | `1199ea1cb231cc939bbde84048f0170a` |
| `free_matched16/launch_manifest.json` | 3,600 | `8cdecbc07131039e25b42c2d98b0a091` |
| `null_shardA/launch_manifest.json` | 3,622 | `4429c5a67e43a75ff9898b0ee390f25a` |
| `null_shardB/launch_manifest.json` | 3,624 | `c0c8d79bbc80ceaaba66b23062aa79cb` |

## LEG 1 -- provenance control (arm-B snap reproduction)

| artifact | bytes | sha256 |
|---|---:|---|
| `verify_snap_arm_pair33/ETA_GATE_ROWS.jsonl` | 550 | `1d7cf9368c6c065194e0be68d89e73dd` |
| `verify_snap_arm_pair33/ETA_GATE_VERDICT.json` | 2,547 | `7633c1da7274a2847f2090a1cb86e2bd` |

## LEG 2 -- retained coder payloads (one per candidate level, 74 levels)

| artifact | bytes | sha256 |
|---|---:|---|
| `retained/fo2h_mask_m001.rc` | 5 | `d397b9ab75a9e866ae24f4b735467b92` |
| `retained/fo2h_mask_m002.rc` | 6 | `c1dca78a3311ac774da9ef93c0181970` |
| `retained/fo2h_mask_m003.rc` | 8 | `39db2aa087da5388f4a98960f5e04425` |
| `retained/fo2h_mask_m004.rc` | 32 | `306fd17bbcd6c083bac29ab85052f99e` |
| `retained/fo2h_mask_m005.rc` | 32 | `a9b97972717376b482f3a9a882054277` |
| `retained/fo2h_mask_m006.rc` | 32 | `f8ad67fbee1843c59a8a4003935273cb` |
| `retained/fo2h_mask_m007.rc` | 49 | `33b95b4b277951bb05efa7c1b13133d1` |
| `retained/fo2h_mask_m008.rc` | 49 | `0925b222567e2091e36c782378ced2ef` |
| `retained/fo2h_mask_m009.rc` | 61 | `91e9578118fefc8c40d6fa92aebc6e04` |
| `retained/fo2h_mask_m010.rc` | 66 | `06326be40d6293bfa044ab46aaf0bd67` |
| `retained/fo2h_mask_m011.rc` | 69 | `4180d59989669d4aec968369bfb6b7c2` |
| `retained/fo2h_mask_m012.rc` | 71 | `7f04365395a167d933e42b1e56d26a20` |
| `retained/fo2h_mask_m013.rc` | 93 | `354bc75219ec65b5097e3e8465170d73` |
| `retained/fo2h_mask_m014.rc` | 95 | `41201ceeeb054730b51384bd88fc9d88` |
| `retained/fo2h_mask_m015.rc` | 101 | `2688224f00ba46b8fc10c9e7bc4135f7` |
| `retained/fo2h_mask_m016.rc` | 196 | `669dab6b39aada71f09200c75121e34f` |
| `retained/fo2h_mask_m017.rc` | 359 | `0e83085a6c2fbf7722a64b1ea9c18d3f` |
| `retained/fo2h_mask_m018.rc` | 381 | `fd69d11c88e36aff78bcceaf7547c63a` |
| `retained/fo2h_mask_m019.rc` | 644 | `e5d329f87853a32a7c9a67ec5cde7a3b` |
| `retained/fo2h_mask_m020.rc` | 675 | `10d79a75b423da5aa5fe200897c0b1f5` |
| … 108 more of the same kind … | | |
| `retained/fo2h_target_m055.rc` | 468 | `82eacdd9d4c5fa0782294d0cfd5b97fa` |
| `retained/fo2h_target_m056.rc` | 505 | `022d70aaa17d1908b3f2652a82d235a8` |
| `retained/fo2h_target_m057.rc` | 506 | `45ef186a726b78aa9924d6680aed496b` |
| `retained/fo2h_target_m058.rc` | 588 | `7c4a38deea47d24b7154afce4dc35f9a` |
| `retained/fo2h_target_m059.rc` | 627 | `b6b9b7f4cab0ee048b8b81b3c01450c2` |
| `retained/fo2h_target_m060.rc` | 628 | `652063a69fceec1e4f6d78af542a7048` |
| `retained/fo2h_target_m061.rc` | 773 | `bed7daf5c68e2c58aab5a68dd5260222` |
| `retained/fo2h_target_m062.rc` | 886 | `59ab7bacb3b4bc9bb34afb9b565d458e` |
| `retained/fo2h_target_m063.rc` | 886 | `13d93f19918910c2b42869a993be98ad` |
| `retained/fo2h_target_m064.rc` | 939 | `636700a5a8df7742176a4828bbd5f338` |
| `retained/fo2h_target_m065.rc` | 966 | `bb40e9bef4858b81e0ee234faec5a1e0` |
| `retained/fo2h_target_m066.rc` | 1,020 | `289d8ea759815b6ddf0ea7a78a902c37` |
| `retained/fo2h_target_m067.rc` | 1,102 | `194b925f9f07154d39fa0d4257a39e56` |
| `retained/fo2h_target_m068.rc` | 1,194 | `c9fff34c4f8413f574b6213f1da190dc` |
| `retained/fo2h_target_m069.rc` | 1,218 | `86cbfe0b6c70f66f20435ddf6e7a256b` |
| `retained/fo2h_target_m070.rc` | 1,226 | `81dbabed9253feb0cefe99b9caedb4e3` |
| `retained/fo2h_target_m071.rc` | 1,226 | `81dbabed9253feb0cefe99b9caedb4e3` |
| `retained/fo2h_target_m072.rc` | 1,226 | `81dbabed9253feb0cefe99b9caedb4e3` |
| `retained/fo2h_target_m073.rc` | 1,226 | `81dbabed9253feb0cefe99b9caedb4e3` |
| `retained/fo2h_target_m074.rc` | 1,226 | `81dbabed9253feb0cefe99b9caedb4e3` |

## LEG 2 -- selections + round-trip proof

| artifact | bytes | sha256 |
|---|---:|---|
| `retained/fo2h_selected_cells_m001.npy` | 136 | `0b71695815cefb729847f800b09ea90b` |
| `retained/fo2h_selected_cells_m002.npy` | 144 | `4649d6e2efbddeb3214278c2e9066d05` |
| `retained/fo2h_selected_cells_m003.npy` | 152 | `47cacbe88f36fe7cd949663967288492` |
| `retained/fo2h_selected_cells_m004.npy` | 160 | `5870c1b165171ee7a2ab5e080380d7f2` |
| `retained/fo2h_selected_cells_m005.npy` | 168 | `398305869e66fe29db56a8bbaaf5a720` |
| `retained/fo2h_selected_cells_m006.npy` | 176 | `a05dd8a91e4eab88a66ede1534d2d6a4` |
| `retained/fo2h_selected_cells_m007.npy` | 184 | `dab98357fa2994b2bf6f3c39ba843d27` |
| `retained/fo2h_selected_cells_m008.npy` | 192 | `d5409f354f6793e885da9a073c6bb9ec` |
| `retained/fo2h_selected_cells_m009.npy` | 200 | `92369b9c5576024f99f1778d9b384432` |
| `retained/fo2h_selected_cells_m010.npy` | 208 | `5df7bfa1266d68c7636c185888053b32` |
| `retained/fo2h_selected_cells_m011.npy` | 216 | `bd43b6ac116864677c6a784f75152e7d` |
| `retained/fo2h_selected_cells_m012.npy` | 224 | `62eb3ea858260fab3d7639f45e101783` |
| `retained/fo2h_selected_cells_m013.npy` | 232 | `e21b7e5902c388384d2be34d1e015525` |
| `retained/fo2h_selected_cells_m014.npy` | 240 | `73847f37600dc850616e4d6ba785bb62` |
| `retained/fo2h_selected_cells_m015.npy` | 248 | `aaa69b121f03f23b04f6e7fdef178eed` |
| `retained/fo2h_selected_cells_m016.npy` | 256 | `b7b0df33268d1521ef28a20f32cbbf7e` |
| `retained/fo2h_selected_cells_m017.npy` | 264 | `b20c5e4308bd3f43bda4540f81b66734` |
| `retained/fo2h_selected_cells_m018.npy` | 272 | `2e94e08bcef7ddcbf17ef1571cdf6ec4` |
| `retained/fo2h_selected_cells_m019.npy` | 280 | `e8d51c4872661bcce387ca7713ebce7b` |
| `retained/fo2h_selected_cells_m020.npy` | 288 | `4df798b62ea9e8fd33024404a0018f0e` |
| … 38 more of the same kind … | | |
| `retained/fo2h_selected_cells_m059.npy` | 600 | `f602c25a04ce325f481d90134e6faf04` |
| `retained/fo2h_selected_cells_m060.npy` | 608 | `6dc2d6d4adcc2db884221e13ab1a30ab` |
| `retained/fo2h_selected_cells_m061.npy` | 616 | `c138c5b3ee7a52141ba9cc8a8d69f212` |
| `retained/fo2h_selected_cells_m062.npy` | 624 | `e1acf56a1ede8e0a6c2fe00ca24a8343` |
| `retained/fo2h_selected_cells_m063.npy` | 632 | `70dd9e09cab957e2b42d703d91bf475b` |
| `retained/fo2h_selected_cells_m064.npy` | 640 | `f21b1a8fb00e211da9fcac93482e47bb` |
| `retained/fo2h_selected_cells_m065.npy` | 648 | `18bae5c743d45aa36d435a1e4bc0b564` |
| `retained/fo2h_selected_cells_m066.npy` | 656 | `964d1a0acbc29359c31df4767da7e2be` |
| `retained/fo2h_selected_cells_m067.npy` | 664 | `3b786c40893f62995827fb20a99202e6` |
| `retained/fo2h_selected_cells_m068.npy` | 672 | `6295e080cec37b0c17b40cb491a6e3a1` |
| `retained/fo2h_selected_cells_m069.npy` | 680 | `2f6236de60fe19c7c17fe9203e2321db` |
| `retained/fo2h_selected_cells_m070.npy` | 688 | `c8f780fb56c4b3a23c43f6507bd6ea71` |
| `retained/fo2h_selected_cells_m071.npy` | 696 | `c73991bdbf45eb63a93f8aa5a1b8738b` |
| `retained/fo2h_selected_cells_m072.npy` | 704 | `382cad366919278ebf8638ee99b1e04e` |
| `retained/fo2h_selected_cells_m073.npy` | 712 | `4f46ed359de75292559ca77948c15749` |
| `retained/fo2h_selected_cells_m074.npy` | 720 | `e3256cb701190390b497c0bea778da27` |
| `retained/fo2h_roundtrip_decoded_m009.npy` | 624 | `1054dd9a191c73255f20e0ea76ec0a26` |
| `retained/fo2h_roundtrip_decoded_m041.npy` | 94,252 | `8bca66ec89eb830a2c3ee0fe4df55149` |
| `retained/fo2h_restricted_truth_m009.npy` | 624 | `1054dd9a191c73255f20e0ea76ec0a26` |
| `retained/fo2h_restricted_truth_m041.npy` | 94,252 | `8bca66ec89eb830a2c3ee0fe4df55149` |

## run logs

| artifact | bytes | sha256 |
|---|---:|---|
| `logs/leg2.log` | 18,917 | `b07e4807dde9fe49a06ddd98084b7bed` |
| `logs/leg2_final.log` | 39,699 | `df97a894b9fa7a88c427f0da438cb16f` |
| `logs/leg2_full.log` | 22,376 | `ec561cb681cf74599475ddf79116cd9c` |
| `logs/leg2_full_retain.log` | 23,973 | `08d24afdf76a608c03cbaecc8df82b62` |
| `logs/leg2_repeat3.log` | 39,699 | `da92b50d10d78b2b35c31080ed2e8d84` |
| `logs/leg2b_final.log` | 8,730 | `db48833caac266f5a91b1ab7cb16b0ea` |
| `logs/leg2b_rerank.log` | 8,197 | `e7d613bb804c45c8f06919689d949d00` |
| `logs/smoke.log` | 2,601 | `31a617afba257cb6f9e6919e4bd4a9ea` |
| `logs/verify_snap.log` | 2,519 | `dcbce79bebf19ddbcd27008b47bf4a40` |
| `free_matched16/run.log` | 3,756 | `c98031b862d7a478b05c3a78223d47a8` |
| `null_shardA/run.log` | 4,395 | `1975a805e99338a3fc063df94d1d0c62` |
| `null_shardB/run.log` | 4,396 | `f341ad99edf0a393dc196cc4b6a21bd3` |

**256 retained files, 1,791,263 bytes total.**

