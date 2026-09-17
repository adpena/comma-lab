# FIFTY-FIFTH POINTER MOVE — S 0.13603403441098336 @ 179,255 B [contest-CUDA T4 n600]: price-first pass 3 on move 54's re-rendered field (pd8): set-priced to a true fixed point with the absorb discipline, chosen on a closed-archive ladder — 26 pairs, −11 B, pose 4.08e-6; the dead-carrier-bytes lever closed by measurement (2026-09-17)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M2PP3T886HAK0SCG63B8JMA9`. Lane `ddm_pd8_price_first_pass3_on_move54_20260916`. Modal wall 1659.9 s. Archive sha `ddadf998ddacab9b356b9b6a01a78c845ab3d6f643dd1e3cf8f37840ae550b8a`, 179,255 B. Runtime tree `a6a60b977f581d25343bc45f0861eb182cac4152662f05671dacaaf761493c68`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·179,255/37,545,489 | 0.11935854664191482 |
| seg 100·0.00010288 | 0.010288 |
| pose √(10·4.08e-06) | 0.006387487769068525 |
| **S** | **0.13603403441098336** |

| | ddm_pd7_price_first_pass2_on_move53_20260916 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13605599532783202 | 0.13603403441098336 | **-2.1960916848667855e-05** |
| d_seg | 0.00010287 | 0.00010288 | 1.0000000000008375e-08 |
| d_pose | 4.1e-06 | 4.08e-06 | -1.999999999999981e-08 |
| bytes | 179,266 | 179,255 | -11 |

## Projection fidelity

Projected 0.13603091365293757; realized − projected = 3.1207580457881523e-06. pd8 three-leg projection on the shipped bytes' own cold decode; T4 drift +3.1e-6

## The mechanism

Price-first token pre-distortion, pass 3, on move 54's re-rendered field (pd8): the prior's rows re-captured, the cheap half enumerated at ≤ 8 bits/token on 588 pairs (4,291 proposals; rank-0 sheet 3.279 bits/token — the dearness moves down the list pass by pass), each proposal realized (re-render from the SHIPPED field; per-pair carrier re-solve with frame-0 repair; seg on the frozen argmax), admitted on the RESOLVED pose per real bit (248 pairs positive; 44 pay at their own price), SET-priced by absorb iterations to a true fixed point (the ledger over-credit had flipped sign, −149 % → +0.15 %; without the absorb the pass would have closed at 0.91 bars, below the admit bar) and chosen on a closed-archive size ladder (set 02: 26 pairs / 29 tokens). Shipped: 179,255 B (−11 B), 2 argmax cells cost (12,129 flips; the census predicted +2), pose 4.08e-6; measured on T4 n600: S 0.13603403441098336 (Δ −2.196e-5 vs move 54; projected 0.13603091365293757, drift +3.1e-6). Receiver, renderer weights, basis, prior and carrier codec unchanged (behaviour 9f6e7168…). The first T4 run inflated in 1,602.7 s (a T4 host outlier for a byte-identical receiver that decoded moves 53/54 in 1,185.9 / 1,112.2 s): the decode leg is minted from a re-measurement of these exact bytes (T4_TIMING_NOTE_move55.json), as at move 50. The "16 dead carrier bytes" lever is CLOSED at instance scope: the receiver reads them (Huffman-length sentinel); the compressed extent is 8 B, not 16.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.01603403441098336. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.016675487769068527: archive ≤ 155,174.8 B → **-24,080.2 B**.
- **DISTORTION corner** at held bytes 179,255: distortion ≤ 0.00064145 → **26.0× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **963.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_pd8_fire/run1/MODAL_REMOTE_RESULT.json` (sha `ae1247f56781fd51b266eaeb1a2a22cfd3824e3efe7d9b0bb0842a71fb48c603`).
- Archive: `/Volumes/APDataStore/pact/ddm_pd8/candidate/candidate_runtime/archive.zip` (sha `ddadf998ddacab9b356b9b6a01a78c845ab3d6f643dd1e3cf8f37840ae550b8a`, 179,255 B).
- Seal: `/Volumes/APDataStore/pact/ddm_pd8/SEAL_ddm_pd8_price_first_pass3_contest_cuda.json`.
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_pd8_price_first_pass3_on_move54_20260916/custody_pointer55/archive.zip` (sha verified: True).
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_pd8_price_first_pass3_on_move54_20260916/custody_pointer55/SEAL_ddm_pd8_price_first_pass3_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: (1) a CPU-axis score (sibling refused by design; CPU_AXIS_ADJUDICATION_move55.json); (2) a decode-time claim from the first fire — 1,602.7 s is above our 1,260 s risk ceiling and is re-measured before any leg is inherited; (3) that pass 4 of price-first generation would clear the admit bar — realized nets −1.04e-4 → −4.59e-5 → −2.20e-5 and paying pairs 128 → 72 → 44; pass 4 projects below the bar (verdict_scope: formulation — price-first generation under the seg SCREEN on this object; the joint seg+pose admission (pd9) is the live successor); (4) any rate saving from the 16 zero carrier bytes — a sentinel the receiver reads.

## Next from here

Next: (a) re-measure move 55's bytes on T4 for the decode leg (fired by MAIN at packet time); (b) pd9 — joint seg+pose Lagrange admission on the price-first pool (live; re-bases on move 55); (c) the reviewable packet (submissions/mrs7): the operator decides — submit as measured on move 53 or restage on the current pointer; (d) psa2's search at nice 10, followthrough after its receipt.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489 with d_seg 0.00010288, d_pose 4.08e-6, B 179,255 → 0.13603403441098336; net vs move 54 = 25·(179,255−179,266)/37,545,489 + 100·(0.00010288−0.00010287) + (sqrt(4.08e-5)−sqrt(4.1e-5)) = −2.196e-5

Own-vehicle frontier: **S 0.13603403441098336 @ 179,255 B [contest-CUDA T4 n600]**, archive sha `ddadf998…50b8a`.
