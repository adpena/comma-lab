# FIFTY-FIRST POINTER MOVE — S 0.1362333315680336 @ 179,285 B [contest-CUDA T4 n600]: pointer move 51: S 0.1362333315680336 @ 179,285 B [contest-CUDA T4 n600] — pd2 pose-directed pass 2 (40 pairs; d_pose 4.45e-6 → 4.29e-6; +90 B; +13 seg cells) on move 50 (2026-09-13)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M2CDRQXPWPYVJT0QAPHMJF1P`. Lane `ddm_pd2_pose_directed_pass2_first_measurement_20260913`. Modal wall 1223.7 s. Archive sha `42e47d0bae1b0647d08db8a5061fe3eec6368b2169d89eb62b14f832fdb0978f`, 179,285 B. Runtime tree `d47ada7de3d8c034d49686ddb5cd352afdf80fd442c70050e8ebea2a6751e418`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·179,285/37,545,489 | 0.11937852241050849 |
| seg 100·0.00010305 | 0.010305 |
| pose √(10·4.29e-06) | 0.006549809157525126 |
| **S** | **0.1362333315680336** |

| | ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13628342713679067 | 0.1362333315680336 | **-5.009556875706922e-05** |
| d_seg | 0.00010294 | 0.00010305 | 1.0999999999999725e-07 |
| d_pose | 4.45e-06 | 4.29e-06 | -1.6000000000000016e-07 |
| bytes | 179,195 | 179,285 | +90 |

## Projection fidelity

Projected 0.13622721373953445; realized − projected = 6.1178284991580956e-06. three-leg Lagrange admission on the resolved pose; realized minus projected +6.1e-6 (the T4 print rounds d_pose to 4.29e-6 against a resolved 4.282e-6); pd2 walked 166 of 588 non-floor pairs so pass 3 is owed

## The mechanism

Pose-directed token pre-distortion, pass 2 (pd2) on the move-50 field: K_refine 8 (from a timing smoke: 33.3 s per refine + 27.2 s per pair), 166 pairs walked, 1,079 refined rows, 165 carried a resolved-pose credit (median −18.5 % of the pair's own d_pose; 82 of 165 at ≤ 0 seg cells), the three-leg Lagrange admission on the RESOLVED pose kept 40 pairs (median credit −52.7 %; 26 of 40 at ≤ 0 cells; 8 of the 40 are pairs move 50 itself edited — "a repair consumes local slack" is refuted for pose credits). Exact legs: token stream 118,978 → 119,055 B (+77 B, 15.4 bits/token, twins) + 13 B carrier splice; seg 12,148 cells on the candidate's own cold parse-back (zero disagreeing); resolved pose 4.449e-6 → 4.282e-6 composed at 1.000129 of the per-pair sum. Receiver byte-identical to move 50. Move 50 had no mintable leg (its first T4 decode 1,375.8 s > 1,260) so MAIN re-measured move 50's exact bytes on T4 (identical score; 1,123.3 s), minted the leg, and pd2 sealed on the normal path inheriting it. Exact T4 row: d_seg 0.00010305, d_pose 4.29e-6, 179,285 B; realized minus projected +6.1e-6 (the T4 pose-print class: the print rounds d_pose to 4.29e-6 against a resolved 4.282e-6). This row's own T4 decode (1,164.9 s) mints its leg directly.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.01623333156803361. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.016854809157525127: archive ≤ 154,905.5 B → **-24,379.5 B**.
- **DISTORTION corner** at held bytes 179,285: distortion ≤ 0.00062148 → **27.1× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **933.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_pd2_fire/run1/MODAL_REMOTE_RESULT.json` (sha `adf41c7ec2650d768971071e4daeab070d694225f0c20b7e4a97dd20992b4d26`).
- Archive: `/Volumes/APDataStore/pact/ddm_pd2/candidate/candidate_runtime/archive.zip` (sha `42e47d0bae1b0647d08db8a5061fe3eec6368b2169d89eb62b14f832fdb0978f`, 179,285 B).
- Seal: `/Volumes/APDataStore/pact/ddm_pd2/SEAL_ddm_pd2_pose_directed_pass2_contest_cuda.json`.
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_pd2_pose_directed_pass2_first_measurement_20260913/custody_pointer51/archive.zip` (sha verified: True).
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_pd2_pose_directed_pass2_first_measurement_20260913/custody_pointer51/SEAL_ddm_pd2_pose_directed_pass2_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: any CPU-axis score (declaration + typed refusal receipt, fired after this harvest). Not claimed: closure of the pose-directed family — pd2 walked 166 of the 588 non-floor pairs at K_refine 8; tier 2 and deeper K are unpriced. Not claimed: any renderer, frame_embed, prior, basis, or carrier-lattice change. Not claimed: that the first-measurement chain can carry a distortion move (measured: it is rate-only; a distortion row seals only by inheriting the pointer's measured leg).

## Next from here

Next: (1) CPU sibling (declaration + refusal receipt). (2) Mint this move's leg at harvest (1,164.9 s ≤ 1,260 — done at harvest per the law). (3) pd3: pose-directed pass 3 on the move-51 field — the tier-2 pairs pd2 never walked plus a re-walk of the credited pairs at K_refine ≥ 12; same three-leg admission; normal seal inheriting move 51's leg. (4) PR #140: restage on the final pointer once the operator answers A/B. Rate corner at this move: cap 154,758 B; demand −24,527 B at held distortion.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489: 100·0.00010305 + sqrt(4.29e-5) + 25·179,285/37,545,489 = 0.1362333315680336; Δ vs move 50 = −5.0096e-5 (pose −1.20e-4 print-rounded, seg +1.1e-5, rate +5.99e-5 for +90 B); projected −5.62e-5, realized −5.01e-5, gap +6.1e-6 = the T4 pose-print class

Own-vehicle frontier: **S 0.1362333315680336 @ 179,285 B [contest-CUDA T4 n600]**, archive sha `42e47d0b…0978f`.
