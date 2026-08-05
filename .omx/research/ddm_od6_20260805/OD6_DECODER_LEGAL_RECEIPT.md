# OD6 decoder-legal context targeter receipt - 2026-08-05

Status: `SCORER_FREE_DECODER_LEGAL_CONTEXT_PRICED / NO FRONTIER MOVE`.

Axis: `[macOS-CPU cache-derived advisory / scorer-free mask-domain replay]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`, `scorer_forwards_run=0`.

## Answer First

OD6 rebuilt OD5's bucket-context targeter with decoder-legal feature columns only. The best row is `base_rgb_generator_geometry_b1024`: `7334` exact n32 OD5 packet bytes, `76304` projected n600 packet bytes, `6111` retained fixes out of the fixed denominator `6,177`, eta `0.548514`, and projected `S = 0.743600771` with OD2 pose credit.

That is `below` the live own line by `0.010379959` (`delta_vs_live=-0.010379959`). It is not a score and not promotion-eligible: OD6 ran no scorer, no `upstream/evaluate.py`, no receiver-closed RGB/inflate candidate, and no full n600 dispatch.

Against OD5's scorer-native selected index (`S=0.743783052`, `78,010` projected bytes, eta `0.553900`, ratio `0.530`), the best decoder-legal row has `delta_S=-0.000182281`, `delta_bytes=-1706`, `eta_lost=0.005386`, and `retained_fixes_lost=60`. OD4's sparse baseline ratio is printed as `0.711`; OD5's scorer-native ratio is printed as `0.530`.

## Price Table

| surface | bucket_count | exact n32 bytes | projected n600 bytes | retained fixes | eta | S w/ OD2 pose credit | rate/seg ratio | vs OD5 0.530 | vs OD4 0.711 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| base_rgb_local_b512 | 512 | 6623 | 75601 | 5195 | 0.466296 | 0.757692100 | 0.610 | worse | better |
| base_rgb_local_b1024 | 1024 | 7294 | 76272 | 5024 | 0.450947 | 0.760856863 | 0.636 | worse | better |
| base_rgb_local_b2048 | 2048 | 8079 | 77098 | 5799 | 0.520510 | 0.749088569 | 0.557 | worse | better |
| base_rgb_local_b4096 | 4096 | 9084 | 78119 | 5649 | 0.507046 | 0.752152597 | 0.579 | worse | better |
| base_rgb_local_b8192 | 8192 | 10409 | 79446 | 5477 | 0.491608 | 0.755770058 | 0.608 | worse | better |
| base_rgb_local_b16384 | 16384 | 12113 | 81137 | 4920 | 0.441612 | 0.765749302 | 0.691 | worse | better |
| base_rgb_generator_b512 | 512 | 6633 | 75613 | 5153 | 0.462526 | 0.758367662 | 0.615 | worse | better |
| base_rgb_generator_b1024 | 1024 | 7305 | 76287 | 5301 | 0.475810 | 0.756464054 | 0.603 | worse | better |
| base_rgb_generator_b2048 | 2048 | 8049 | 77082 | 4519 | 0.405619 | 0.769422967 | 0.715 | worse | worse |
| base_rgb_generator_b4096 | 4096 | 9096 | 78142 | 5006 | 0.449331 | 0.762388121 | 0.654 | worse | better |
| base_rgb_generator_b8192 | 8192 | 10481 | 79517 | 5605 | 0.503097 | 0.753782829 | 0.594 | worse | better |
| base_rgb_generator_b16384 | 16384 | 12234 | 81289 | 5802 | 0.520779 | 0.751831500 | 0.587 | worse | better |
| base_rgb_generator_geometry_b512 | 512 | 6671 | 75656 | 5723 | 0.513688 | 0.749336388 | 0.554 | worse | better |
| base_rgb_generator_geometry_b1024 | 1024 | 7334 | 76304 | 6111 | 0.548514 | 0.743600771 | 0.523 | better | better |
| base_rgb_generator_geometry_b2048 | 2048 | 8177 | 77202 | 5605 | 0.503097 | 0.752241365 | 0.577 | worse | better |
| base_rgb_generator_geometry_b4096 | 4096 | 9169 | 78193 | 4721 | 0.423750 | 0.766952033 | 0.694 | worse | better |
| base_rgb_generator_geometry_b8192 | 8192 | 10625 | 79652 | 4687 | 0.420698 | 0.768463937 | 0.712 | worse | worse |
| base_rgb_generator_geometry_b16384 | 16384 | 12780 | 81826 | 5900 | 0.529575 | 0.750631398 | 0.581 | worse | better |
| base_rgb_generator_geometry_proxy_b512 | 512 | 7307 | 76347 | 5983 | 0.537025 | 0.745663908 | 0.535 | worse | better |
| base_rgb_generator_geometry_proxy_b1024 | 1024 | 7984 | 77022 | 5672 | 0.509110 | 0.751056574 | 0.569 | worse | better |
| base_rgb_generator_geometry_proxy_b2048 | 2048 | 8750 | 77861 | 5541 | 0.497352 | 0.753697419 | 0.589 | worse | better |
| base_rgb_generator_geometry_proxy_b4096 | 4096 | 9796 | 78903 | 6013 | 0.539718 | 0.746889006 | 0.550 | worse | better |
| base_rgb_generator_geometry_proxy_b8192 | 8192 | 11283 | 80411 | 5410 | 0.485594 | 0.757477548 | 0.623 | worse | better |
| base_rgb_generator_geometry_proxy_b16384 | 16384 | 13401 | 82522 | 5336 | 0.478952 | 0.760059375 | 0.648 | worse | better |

## Decode-Time Compute Path

The decoder-side path represented by the legal feature table is: inflate qo1 base frames; resize the last frame of each pair to the 384x512 scorer lattice with generic bilinear interpolation; compute RGB/YUV bins, local luma gradients, local contrast, and chroma-delta bins; optionally compute generator/hybrid coverage bits from shipped PE3 hybrid75 coordinates; optionally add generic row/column/horizon bins; optionally read the counted block-prior table; hash the feature tuple to a shipped bucket table; and keep candidate corrections whose bucket probability crosses the shipped threshold. No scorer weights, scorer forwards, GT frames, GT margin, Fisher table, or cached scorer-native distances are read by this decoder path.

## RECALL EVIDENCE

| source | recalled fact | plan impact |
|---|---|---|
| `.omx/tmp/codex_runs/_common_contract.md`, `.omx/tmp/codex_runs/od6_prompt.md`, `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `.omx/state/main_hot_state.md` | OD6 owns no scorer slot, must avoid protected files/staged index, must use serializer hashes, and live own line is `S=0.7539807296911207 @ 357,836 B`. | Built a scorer-free repricing receipt only and queued the scorer gate. |
| `.omx/research/ddm_od5_20260805/OD5_GENERATOR_PACKET_RECEIPT.md` | OD5's best projection was `S=0.743783052`, ratio `0.530`, but its context row mixed scorer-native cached fields. | Used OD5 as the illegal-feature baseline to quantify the decoder-legality tax. |
| `.omx/research/ddm_od4_20260805/OD4_WEAK_PACKET_RECEIPT.md` | OD4 sparse packet ratio was about `0.711` with `S=0.761509399`. | Printed OD4 as the weak-packet where-tax baseline. |
| `.omx/research/ddm_st2_20260805/ST2_RECEIPT_20260805.md` | The strong ST2 selected table used scorer-native margin/Fisher/head-distance context and is not receiver-legal as a decoder feature. | Excluded ST2 features and rebuilt hashes from qo1 RGB/generator/geometry/counted-proxy columns. |
| `.omx/research/operator_directive_per_edge_optimality_criteria_20260805.md`, `.omx/research/ddm_qo1_repair_stream_optimal_form_20260804.md`, `/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/C_qo1_pairbit_n600_eval_receipt.json` | Seg-first surgical packet work must price legal bytes; qo1 is the current measured own-vehicle base under SSD custody. | Used qo1 decoded frames as the base-frame legal context and kept pose credit inherited from OD2 only. |
| bounded `rg` and canonical-equation recall over `.omx/research`, `.omx/state`, `docs`, `src/tac`, `experiments`, and `tools` | Existing receiver-closed/worldsheet/address-law evidence emphasizes counted bytes and candidate-universe legality; no existing OD6 decoder-legal targeter table was found. | Added the candidate-universe and false-positive collateral caveat instead of claiming receiver closure. |

## SHA Table

| artifact | bytes | sha256 |
|---|---:|---|
| `/Users/adpena/Projects/pact/experiments/ddm_od6_decoder_legal_context.py` | 56311 | `cb41f8180e61dc4817a1d94c6289f1c6912aecfe855da50662857b324abba1d2` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od2_20260805/od2_js1_n32_cprime_k4.json` | 103690 | `fd1016751e4668ff786692f52f91d924be97081a70a20d11e470150aaf85c6af` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od2_20260805/PAIR_SELECTION.json` | 2388 | `0a8ac26a1cd39c7dc425dbb4922d0dda6f71227b205241d3d771ea9791c2d4f9` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_pe1_20260805/ddm_pe1_repr_race_receipt.json` | 50622 | `d6db13bb1ba14b140fb86b0d6c09dcff836c30a0a155a5d2cbd8638ea788bb88` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_pe3_20260805/ddm_pe3_hybrid_receipt.json` | 69875 | `7839d5c3e18c1b89959538fa518621bd3e4c0a8cd166eccf5d9c240dd79c1716` |
| `/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip` | 357836 | `d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a` |
| `/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive/0.bin` | 357728 | `bd6da15ec41030762e4510555f12d3d141d3c71f495af701666a769b12f0b036` |
| `/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/inflated/0.raw` | 3662409600 | `3ce7d269a7080a4024a576694cd0ddc697099c64cd02fdd2bb879339e4b03f31` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od6_20260805/ddm_od6_decoder_legal_receipt.json` | 550769 | `4775cb3fc8925de33a70945ba1469b962f0934b5527a032e39cb60e0589566b1` |
| `/Users/adpena/Projects/pact/.omx/research/ddm_od6_20260805/NEXT_IF_RESUMED.md` | 888 | `ffd63023edea132bb377f9186e4531ebbe374278d952668730ae484d2f5a5594` |

## NEXT_IF_RESUMED

See `/Users/adpena/Projects/pact/.omx/research/ddm_od6_20260805/NEXT_IF_RESUMED.md`. The first continuation gate is receiver-shape closure: the targeter must operate over a decoder-computable candidate universe with measured false-positive collateral, not only retained OD2 positives.

## Boundaries

- No `upstream/evaluate.py`, SegNet, PoseNet, full n600 scorer job, contest-CPU, or contest-CUDA run.
- OD2 pose credit is inherited for the same-row projection only and was not remeasured by OD6.
- n600 bytes are projected component sums: PE3 measured hybrid75 bytes plus exact counted table/proxy bytes. They are not exact archive bytes.
- Feature columns are decoder-legal, but the table is fit on n32 labels; this is not a receiver-closed candidate-universe proof.
- The counted proxy variant prices a global 24x32 block-prior table once; a per-pair or n600-trained proxy would need its own exact byte price.
- This does not move the frontier.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
