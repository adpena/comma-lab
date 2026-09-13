# FIFTY-SECOND POINTER MOVE — S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600]: pointer move 52: S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600] — pd3 pose-directed pass 3 (26 pairs; d_pose 4.29e-6 → 4.21e-6; +47 B; seg −1 cell) on move 51 (2026-09-13)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M2D1J97K1QPBQ4KASFH1SG96`. Lane `ddm_pd3_pose_directed_pass3_on_move51_20260913`. Modal wall 1122.1 s. Archive sha `ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`, 179,332 B. Runtime tree `7cb2c609c8e8f55cb8c67e061b2fcb49e471410769cf5530484335f3ee385d64`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·179,332/37,545,489 | 0.11940981778130523 |
| seg 100·0.00010304 | 0.010304 |
| pose √(10·4.21e-06) | 0.0064884512790033344 |
| **S** | **0.13620226906030858** |

| | ddm_pd2_pose_directed_pass2_first_measurement_20260913 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.1362333315680336 | 0.13620226906030858 | **-3.1062507725027055e-05** |
| d_seg | 0.00010305 | 0.00010304 | -9.999999999994822e-09 |
| d_pose | 4.29e-06 | 4.21e-06 | -7.999999999999923e-08 |
| bytes | 179,285 | 179,332 | +47 |

## Projection fidelity

Projected 0.13620052157076992; realized − projected = 1.7474895386559997e-06. three-leg Lagrange admission on the resolved pose; realized minus projected +1.7e-6 inside the seal's spread; the family is draining on depth (admitted/walked 0.09) so pass 4 targets the token price

## The mechanism

Pose-directed token pre-distortion, pass 3 (pd3) on the move-51 field: K_refine 12 and a 283-pair budget declared from an 8-shard timing smoke run under the real fleet condition; 271 pairs carried a resolved-pose credit (median −27.3 % of the pair's own d_pose; 143 neutral and 14 repairing on seg), the three-leg Lagrange admission on the RESOLVED pose kept 26 pairs (median credit −61.6 %; 21 of 26 at ≤ 0 seg cells). Exact legs: token stream 119,055 → 119,097 B (+42 B, 12.9 bits/token, twins) + 5 B carrier splice; seg 12,148 → 12,147 cells on the candidate's own cold parse-back (zero disagreeing); resolved pose 4.2820e-6 → 4.2075e-6 composed at 0.99989 of the per-pair sum; K=12 strictly better than K=8 on 24 of 125 shared pairs and worse on none. Receiver byte-identical to move 51; normal seal inheriting move 51's own minted leg (1,164.9 s). Exact T4 row: d_seg 0.00010304, d_pose 4.21e-6, 179,332 B; realized minus projected +1.7e-6 (T4 print class). This row's decode (1,067.8 s) mints its own leg. Family decay measured: admitted/walked 0.49 → 0.24 → 0.09; tier 2 poor (4 of 100).

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.016202269060308583. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.016792451279003334: archive ≤ 154,999.1 B → **-24,332.9 B**.
- **DISTORTION corner** at held bytes 179,332: distortion ≤ 0.00059018 → **28.5× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **886.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_pd3_fire/run1/MODAL_REMOTE_RESULT.json` (sha `27627b966aa91282dc4cf8b670f5e18bbcb9a7fb456e1d9b93422c534be65c0c`).
- Archive: `/Volumes/APDataStore/pact/ddm_pd3/candidate/candidate_runtime/archive.zip` (sha `ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`, 179,332 B).
- Seal: `/Volumes/APDataStore/pact/ddm_pd3/SEAL_ddm_pd3_pose_directed_pass3_contest_cuda.json`.
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_pd3_pose_directed_pass3_on_move51_20260913/custody_pointer52/archive.zip` (sha verified: True).
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_pd3_pose_directed_pass3_on_move51_20260913/custody_pointer52/SEAL_ddm_pd3_pose_directed_pass3_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: any CPU-axis score (declaration + typed refusal receipt after this harvest). Not claimed: that deeper search pays further — pd3 measured the single-token pose family draining on depth; the unpriced lever is the token PRICE (12.9–17 bits per changed token vs 8.9 for clustered edits). Not claimed: any renderer, frame_embed, prior, basis, or carrier-lattice change.

## Next from here

Next: (1) CPU sibling. (2) Mint this move's leg at harvest (1,067.8 s — done). (3) pd4: the PRICE lever — rank proposals by resolved-pose credit per REAL bit (measure each pair's edit cost by real encode or a calibrated ledger re-verified), prefer edits adjacent to already-changed tokens (clustered price ~8.9 bits/token), keep K_refine 12; normal seal inheriting move 52's leg. (4) PR #140: restage on the final pointer once the operator answers A/B. Rate corner: cap 154,806 B; demand −24,526 B at held distortion.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489: 100·0.00010304 + sqrt(4.21e-5) + 25·179,332/37,545,489 = 0.13620226906030858; Δ vs move 51 = −3.106e-5 (pose −6.14e-5 print-rounded, seg −1e-6, rate +3.13e-5 for +47 B); projected −3.28e-5, realized −3.11e-5, gap +1.7e-6 = the T4 pose-print class

Own-vehicle frontier: **S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600]**, archive sha `ae59c510…3b20e`.
