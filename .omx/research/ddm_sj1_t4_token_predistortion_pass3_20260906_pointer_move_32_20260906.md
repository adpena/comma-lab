# THIRTY-SECOND POINTER MOVE — S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]: sj1 pass 3: a second round of token pre-distortion on the residual field (370-pair admitted subset) with the carrier re-solved — 1,447 more flips gone, pose below base, for +741 B (2026-09-06)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M1TFD35EPY2YZHNV3VKJP6MG`. Lane `ddm_sj1_t4_token_predistortion_pass3_20260906`. Modal wall 535.9 s. Archive sha `06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3`, 181,645 B. Runtime tree `6508d184e60ab3e3c63b05ed5f0f3bc3ff551408f38af1620f90c62c0b21783e`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·181,645/37,545,489 | 0.12094994953987681 |
| seg 100·0.00010913 | 0.010912999999999999 |
| pose √(10·5.1e-06) | 0.0071414284285428505 |
| **S** | **0.13900437796841966** |

| | ddm_sj1_t4_token_predistortion_joint_20260906 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.1398140172839628 | 0.13900437796841966 | **-0.000809639315543148** |
| d_seg | 0.00012009 | 0.00010913 | -1.0960000000000003e-05 |
| d_pose | 5.4e-06 | 5.1e-06 | -2.9999999999999967e-07 |
| bytes | 180,904 | 181,645 | +741 |

## Projection fidelity

Projected 0.13900021608143795; realized − projected = 4.161886981712826e-06. seg leg matched the shipped bytes to the cell (12,866 = 12,866; T4 print 0.00010913 vs 1.0914e-4); pose 5.093e-6 vs the 2-sig-fig print 5.1e-6; realized − projected = +4.2e-6, entirely the pose print

## The mechanism

Second pass of token PRE-DISTORTION on the residual field of the 31st move (sj1 pass 3): the same single-cell 3×3 × 4-class proposal family under REALIZED
acceptance over all 600 pairs repaired 1,447 more flips (10.22 % of the remaining 14,157) with 1,339 changed tokens (1.081 cells/token); the marginal coding cost
FELL to 5.31 bits per changed token (from 6.23 in pass 2a — the context model had already absorbed the earlier edits' neighbourhood structure). The jg5 Lagrange
admission over pose damage kept 370 of 445 edited pairs (the 75 dropped are those whose carrier re-solve went badly); the subset was priced by a REAL encode pair
(twin byte-identical, 120,225 B; the ledger-sum estimate under-charged by 19.6 B), and non-admitted pairs CARRY the live row's planes (a silent-revert class the arm
caught and made unrepresentable). The pose carrier was re-solved on the candidate's own renders from the live row's coefficients (guard-enforced): d_pose 5.398e-6 →
5.093e-6. Model sections byte-identical to the 31st-move archive; +728 B token stream, +13 B carrier. Full CPU inflate decoded exactly the admitted field; the final
seg leg on the shipped bytes matched the admission to the cell (12,866 = 12,866). Seal `SEAL_ddm_sj1_token_predistortion_pass3_contest_cuda.json`; memo
`ddm_sj1_multipass_token_predistortion_20260905.md`.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.019004377968419667. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.01805442842854285: archive ≤ 153,103.9 B → **-28,541.1 B**.
- **DISTORTION corner** at held bytes 181,645: distortion ≤ -0.00094995 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-1,426.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass3_20260906/MODAL_REMOTE_RESULT.json` (sha `56838b87033259d8ce2134734a067aebb35089df84867930bf71921afda1d5fa`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate_pass3/candidate_runtime/archive.zip` (sha `06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3`, 181,645 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/SEAL_ddm_sj1_token_predistortion_pass3_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_token_predistortion_pass3_32nd_move/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_token_predistortion_pass3_32nd_move/SEAL_ddm_sj1_token_predistortion_pass3_contest_cuda.json` (sha verified: True).

## What this does NOT claim

A distortion move (+741 B). The T4 row is authority for d_seg and d_pose. The convergence rule (< 1 % of remaining flips repaired per pass) is NOT reached
(10.22 % in pass 3, with efficiency falling only 1.14× while sites fall 3.95×): a pass 4 is a successor decision on this row, not part of it. 45.8 % of the flipped
cells the arm inherited are gone; the remaining 12,866 include the render→re-segment sites no single-cell token move repairs (the renderer's share — measured on
the born vehicle at 62–65 %, on this vehicle still to be partitioned). `[contest-CPU]` stays RECORD-WITH-REASON. PR #140 is nine moves behind.

## Next from here

sj1 re-checks the convergence rule on the 12,866-cell residual and proposes pass 4 if the marginal economics hold (they did through pass 3: 5.31 bits/token vs an
11.0-bit break-even). Then the unrepairable remainder is the renderer's: the composition of pre-distortion with a renderer fold-back at optimal form (pose in the loop
+ re-solve) is the next door on the seg axis; on the rate axis the carrier's per-atom quantizer step and the packed Rice-k field width remain. The token stream
(now 120.2 KB) is at its floor under the exact-field paradigm for the coder; each pre-distortion pass spends bytes there deliberately.

Equations leg (`tac.canonical_equations`): token_predistortion_multipass_yield_v1: second T4 anchor — pass 3 repairs 10.22 % of the remaining flips at 1.081 cells/token for 5.31 bits/token (marginal cost FELL from 6.23: the context model absorbs earlier edits); the subset admission priced by real encode (ledger under-charged 19.6 B)

Own-vehicle frontier: **S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]**, archive sha `06c44dc4…63cd3`.
