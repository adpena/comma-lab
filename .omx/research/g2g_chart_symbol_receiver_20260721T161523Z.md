# G2g chart-symbol receiver — measured hard-oracle result

**UTC:** 2026-07-21T16:15:23Z  
**Lane:** `lane_g2g_chart_symbol_receiver_578_20260721`  
**Authority:** MEASURED `[macOS-CPU advisory]`; no score claim; pointer `0.19108 [contest-CPU]` unmoved; `MAIN_REVIEW_REQUIRED=true`.

## Verdict

`MEASURED_G2G_CHART_SYMBOL_HARD_ORACLE_NO_ADMISSION_N64_TARGETED_FAMILY_OPEN`.

The direct chart-symbol receiver is real and byte-closed, but none of the six g2f-selected candidates is the first admitted realization correction. All six selected packets passed strict parse-back, canonical re-encode, receiver-derived RGB, factor-2 uint8 exactness, and double decode. None reached the full #549 semantic cell field or its declared pose tube. Five of six selected candidates recovered distortion score at a marginal above `lambda*=25/37,545,489 = 6.658589531221714e-7` score units/byte, but the hard semantic and pose predicates dominate; no D3 correction-price coefficient is authorized.

The negative is formulation-scoped: one support-maximizing LaneLine centerline-intercept coefficient, either sign, at the g2f-selected rung, on six n64 pairs. It does not close joint coefficients, other lane lines, nonlinear chart-symbol QPs, xi-factorized pose, or the broader chart family. The g2f pixel QP result remains an over-constrained pixel formulation, not evidence that no chart correction exists.

## D1 — counted receiver packet

`predictor_upgrade_xi_chart.py` now owns `G2CS1`: a strict fixed-binary packet containing `(pair:u16, line:u8, coefficient:u8, delta:fp32)` rows, a versioned header, and CRC32. Empty correction sets cost zero. A one-symbol candidate is 20 actual emitted bytes: 12-byte header plus one 8-byte row. Generic LBND parsing, coefficient application, rasterization, homography, and exact-R decode are rule-118-free receiver code; only the video-derived `G2CS1` delta is newly counted.

The receiver rejects truncation, trailing bytes, bad CRC, noncanonical ordering, duplicate addresses, invalid pair/line/coefficient addresses, zero/nonfinite deltas, and replay drift. Twelve one-symbol candidate packets were emitted (both signs for six pairs; 240 bytes total diagnostic candidate traffic). The admitted combined packet is empty because no row passed D2.

## D2 — hard CPU-Torch oracle, seed 1234

The mandated execution order was honored: chart-only candidates `[0,34,37,46]`, then pixel/chart overlap controls `[22,30]`. The n16 result is a targeted one-pair subset (`pair 0`), not a full-n16 claim. The n64 result is the six selected candidates, not every n64 pair.

| pair | source level | sign | changed pixels | saturation | delta d_seg | delta d_pose | bytes | recovered S/byte | semantic exact | pose tube | counted | receiver RGB |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 0 | chart-only | + | 102 | 0 | -0.0002390544 | -0.3103098 | 20 | 0.003028408 | false | false | true | true |
| 34 | chart-only | + | 98 | 0 | 0 | -0.2310225 | 20 | 0.001365450 | false | false | true | true |
| 37 | chart-only | + | 114 | 8 | -0.0011850993 | -0.1350719 | 20 | 0.006711964 | false | false | true | true |
| 46 | chart-only | - | 301 | 47 | -0.0002492269 | +0.4757559 | 20 | -0.001513915 | false | false | true | true |
| 22 | overlap | + | 104 | 0 | -0.0004781087 | -0.2611291 | 20 | 0.003960651 | false | false | true | true |
| 30 | overlap | + | 204 | 55 | 0 | -0.9574409 | 20 | 0.005684596 | false | false | true | true |

All six also had `factor2_uint8_exact=true` and `double_decode_identical=true`. Saturation is not the common blocker: pairs 0, 34, and 22 have zero saturated RGB values yet fail both hard predicates. The selected symbols modify only 98–301 pixels while the baseline whole-field `d_seg` is 0.281–0.506 and pose-tube debt is 167.25–180.23 after correction. The single-coefficient move is therefore directionally useful on five rows but far too low-dimensional to close the whole semantic field and pose tube.

The overlap A/B has no byte-per-**admitted** winner: g2f's pixel QP produced no admitted correction packet, and G2g's 20-byte chart packets also admit zero rows. Chart pricing is nevertheless resolved at 20 actual receiver-closed bytes per one-symbol candidate; it is not a pixel proxy.

## D3 — route and next crux

D3 is `NOT_RUN_NO_ADMISSION`. Do not route U1 or register P0 G6/#603 from this result. The next executable crux is a joint multi-coefficient chart-symbol solve with the semantic-cell and pose-tube constraints expressed on the low-dimensional coefficient response, then strict receiver replay. Candidate ranking should retain Fisher/margin reverse-waterfill and stop below the registered rate threshold. n600 is `REFUSED_ABSENT_N64_TARGETED_ADMISSION`.

## Custody and resumability

- Full receipt: `/Volumes/VertigoDataTier/pact/evidence/g2g_chart_receiver_20260721/run_20260721T1622Z/receipt.json`, 105,193 bytes, file SHA-256 `fa49a2ca71cb2960b1e497d425f05c4a496cc7634c45b2e193e3977dfa0667da`, canonical receipt SHA-256 `8b2e86edf53699f5cd819b0f3743bad1978edadfbf3736399580b1f1d0790e6d`. An immediate resume reproduced the file hash byte-for-byte.
- G2f source receipt file SHA-256: `47d3ca538f1b876f7639223a1a9a7714b7db2083eaa0971936b9a43a1e6d0d04`.
- LBND2 source SHA-256: `d2b2a62eeb6ebe45cbf908dafa7e081eabddaca0f424faac970b41eea650d810`; openpilot base remains 121,128 counted bytes.
- Every pair has an immutable stage receipt; checkpoints exist after candidates 2, 4, and 6. Only small packets/JSON persist. No camera, logits, or coverage tensors are retained.
- The sibling directory `/Volumes/VertigoDataTier/pact/evidence/g2g_chart_receiver_20260721/chart_symbol_receiver_openpilot/` is a preserved, superseded pre-seal run whose scorer stages completed but whose final self-hash check exposed the receipt verifier defect. `run_20260721T1612Z/` is a second preserved, superseded run that exposed volatile free-space metadata on resume. Both remain on SSD, were not consumed as authority, and were not deleted.

## Stores consulted

Authority prompt and live inbox; `CLAUDE.md`; `AGENTS.md`; v7.5/v8 specifications; `reports/latest.md`; canonical lane and subagent registries; g2e/g2f receipts and memos; #549 seed/cache/scorer surfaces; openpilot LBND2 chart; #402 receiver helpers; canonical equations/LawRefs named by the measured configuration.

## Pointer delta honesty

No archive was produced, no contest-CPU or contest-CUDA evaluation ran, no score was claimed, and the frontier pointer did not move. This landing is build plus local advisory measurement only and requires MAIN review before merge.
