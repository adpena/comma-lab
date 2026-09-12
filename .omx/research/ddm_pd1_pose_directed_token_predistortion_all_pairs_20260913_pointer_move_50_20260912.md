# FIFTIETH POINTER MOVE — S 0.13628342713679067 @ 179,195 B [contest-CUDA T4 n600]: pointer move 50: S 0.13628342713679067 @ 179,195 B [contest-CUDA T4 n600] — pd1 pose-directed token pre-distortion (20 pairs; d_pose 4.55e-6 → 4.45e-6; +42 B; +8 seg cells) on move 49, normal seal inheriting move 49's minted leg (2026-09-12)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M2BYVCXGYTFC2BN8WZSF5PAS`. Lane `ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913`. Modal wall 1440.6 s. Archive sha `1ea274f612a26183f31f6d439505d0289bd3751b4da9467343fc156203503cd7`, 179,195 B. Runtime tree `87a343e7b1272b193a340ddb5a1a5ff5daec4e86c69b6d92111015ed413569a9`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·179,195/37,545,489 | 0.11931859510472749 |
| seg 100·0.00010294 | 0.010294 |
| pose √(10·4.45e-06) | 0.006670832032063167 |
| **S** | **0.13628342713679067** |

| | ddm_sj1_t4_token_predistortion_pass7_20260912 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13632299781031237 | 0.13628342713679067 | **-3.9570673521699407e-05** |
| d_seg | 0.00010287 | 0.00010294 | 7.000000000000441e-08 |
| d_pose | 4.55e-06 | 4.45e-06 | -9.999999999999989e-08 |
| bytes | 179,153 | 179,195 | +42 |

## Projection fidelity

Projected 0.13628242427525275; realized − projected = 1.0028615379242822e-06. three-leg Lagrange admission on the resolved pose; realized minus projected +1.0e-6 inside the seal's declared spread; pd1 carried 41 credited pairs and refined 3 proposals per pair, so pass 2 is owed

## The mechanism

Pose-directed token pre-distortion over all 600 pairs (pd1): the carrier re-solve credit that moved the pointer at moves 42 and 49 as a side effect, run as the objective. For each pair, up to 32 single-token proposals ranked by pose saliency on the token grid (masked to argmax-interior cells) were screened by render + frozen argmax; the ≤2-cell survivors were refined (re-render, per-pair carrier re-solve with frame-0 repair inside) and the best resolved-pose gain kept: 41 pairs carried (median credit −21.6 % of the pair's own d_pose; 26 of 41 at ≤ 0 seg cells), then the three-leg Lagrange admission on the RESOLVED pose kept 20 pairs: pose 4.5436e-6 → 4.4489e-6, seg +8 cells, token stream 118,938 → 118,978 B (+40 B, real encode, twins; 16.98 bits/token), carrier splice +2 B. Receiver code byte-identical to move 49; normal seal inheriting move 49's own minted t4_direct leg (1,106 s ≤ 1,260). Exact T4 row: d_seg 0.00010294, d_pose 4.45e-6, 179,195 B; realized minus projected +1.0e-6 (the T4 pose-print class). pp1's floor law (0.25–0.45 % on the 12 hard pairs) does not generalize off them.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.01628342713679068. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.016964832032063167: archive ≤ 154,740.2 B → **-24,454.8 B**.
- **DISTORTION corner** at held bytes 179,195: distortion ≤ 0.0006814 → **24.9× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **1,023.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_pd1_fire/run1/MODAL_REMOTE_RESULT.json` (sha `d4ae2a987546cc300381d2d6dc18339e97c9fa394952b8c0e925ac3346d059e2`).
- Archive: `/Volumes/APDataStore/pact/ddm_pd1/candidate/candidate_runtime/archive.zip` (sha `1ea274f612a26183f31f6d439505d0289bd3751b4da9467343fc156203503cd7`, 179,195 B).
- Seal: `/Volumes/APDataStore/pact/ddm_pd1/SEAL_ddm_pd1_pose_directed_token_predistortion_contest_cuda.json`.
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913/custody_pointer50/archive.zip` (sha verified: True).
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913/custody_pointer50/SEAL_ddm_pd1_pose_directed_token_predistortion_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: any CPU-axis score (the receiver declares linux-nvidia-t4; the CPU sibling is a declaration + typed refusal receipt, fired after this harvest). Not claimed: closure of the pose-directed family — pd1 refined only 3 proposals per pair under its time budget and 41 pairs carried credits; a second pass on the move-50 field is owed. Not claimed: any change to the renderer, its per-pair frame_embed (pp1 closed), the prior, the basis, or the carrier lattice. Not claimed: the modelled rate ledger as a charge — the subset was priced by its own real encode (the ledger was 1.9× optimistic).

## Next from here

Next: (1) CPU-axis sibling of this exact archive (declaration + refusal receipt, after this harvest). (2) MINT this move's own t4_direct leg from this harvest (law: an inherited leg blocks its successor) — measured T4 wall 1,440.6 s (vs 1,106 at move 49; still ≤ 1,800). (3) pd2: pose-directed pass 2 on the move-50 field with more refines per pair (pd1 refined 3 of ≤32 screened) and the 12 floor pairs excluded; same three-leg admission; normal seal inheriting this move's minted leg. (4) PR #140: restage on the final pointer once the operator answers A/B. Rate corner at this move: cap 154,701 B; demand −24,494 B at held distortion.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489: 100·0.00010294 + sqrt(4.45e-5) + 25·179,195/37,545,489 = 0.13628342713679067; Δ vs move 49 = −3.957e-5 (pose −6.75e-5 from the resolved carrier on 20 re-rendered pairs, seg +7e-6, rate +2.80e-5 for +42 B); projected −4.06e-5, realized −3.96e-5, gap +1.0e-6 = the T4 pose-print class

Own-vehicle frontier: **S 0.13628342713679067 @ 179,195 B [contest-CUDA T4 n600]**, archive sha `1ea274f6…03cd7`.
