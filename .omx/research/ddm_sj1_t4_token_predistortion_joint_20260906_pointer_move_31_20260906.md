# THIRTY-FIRST POINTER MOVE — S 0.1398140172839628 @ 180,904 B [contest-CUDA T4 n600]: sj1 multi-pass token pre-distortion with the carrier re-solved on the candidate's own renders: 9,593 flips repaired, d_seg 0.000201 → 0.000120, d_pose below base — a distortion move paid for by +6,118 B (2026-09-06)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M1T6TCW2JS1JEW5CSZH3FVBY`. Lane `ddm_sj1_t4_token_predistortion_joint_20260906`. Modal wall 545.7 s. Archive sha `42aa84b59f71d83b8f11a26c635a7af8f32dcfdf183e3fea4bb2007e74a5f2f8`, 180,904 B. Runtime tree `9d51671dedd0b2b10b39903fba6883ba4438b587ab9e54097aa5ff1f7c8e404b`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,904/37,545,489 | 0.12045654805561329 |
| seg 100·0.00012009 | 0.012008999999999999 |
| pose √(10·5.4e-06) | 0.007348469228349534 |
| **S** | **0.1398140172839628** |

| | ddm_pc1_t4_lattice_x16_on_rc1_20260905 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.14411787458634504 | 0.1398140172839628 | **-0.004303857302382225** |
| d_seg | 0.00020139 | 0.00012009 | -8.13e-05 |
| d_pose | 5.77e-06 | 5.4e-06 | -3.6999999999999985e-07 |
| bytes | 174,786 | 180,904 | +6,118 |

## Projection fidelity

Projected 0.1398087424644421; realized − projected = 5.2748195207008575e-06. seg leg via the jg1 instrument (14,157 flips on the parse-back raw vs T4's 14,167: +10 cells, +0.07 %), pose via cpu_torch n600 (5.398e-6 vs T4 print 5.40e-6); realized − projected = +5.3e-6

## The mechanism

Multi-pass PRE-DISTORTION of the stored semantic label field (sj1): for every flipped cell of the frontier body (23,749 at step 0, the jg1 instrument reproducing the
T4 seg leg to −0.033 %), single-cell token moves from a 3×3 × 4-class family were proposed and accepted only under REALIZED acceptance — re-render frame 2p+1
through the receiver's own SemanticTokenRenderer, re-segment with the frozen SegNet (cpu_torch argmax, DALI GT lineage), keep the move only if flips fall.
Pass 2a over all 600 pairs repaired 9,593 flips (40.39 %) with 7,804 changed tokens (1.229 cells per token); the vertical neighbours carry half the repairs (dashcam
boundaries run horizontally). The edited field was re-encoded through the shipped coder (encoder control byte-identical, 113,419 B; edited stream +6,078 B =
6.23 bits per changed token against a 12.52-bit break-even). The pose carrier — stranded 430× by the render change, as jg1/jg4 measured — was RE-SOLVED for every
pair on the candidate's own renders on the live ×16 lattice (full n600 damped Gauss–Newton; guard `assert_carrier_is_pointer` enforced the live tree), landing
BELOW base: d_pose 5.7675e-6 → 5.398e-6. Model sections byte-identical to the 30th-move archive. Twin encode byte-identical; parse-back identity PASS; seg leg
re-measured on the parse-back raw. Seal `SEAL_ddm_sj1_token_predistortion_joint_contest_cuda.json`, admit bar derived from the 30th-move pointer; memo
`ddm_sj1_multipass_token_predistortion_20260905.md`.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.019814017283962815. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.019357469228349532: archive ≤ 151,146.9 B → **-29,757.1 B**.
- **DISTORTION corner** at held bytes 180,904: distortion ≤ -0.00045655 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-685.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_joint_20260906/MODAL_REMOTE_RESULT.json` (sha `c33efa10169d66afef1b8530c384889223bca0ce61e7dbc0e52acdb3487dcc93`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate/candidate_runtime/archive.zip` (sha `42aa84b59f71d83b8f11a26c635a7af8f32dcfdf183e3fea4bb2007e74a5f2f8`, 180,904 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/SEAL_ddm_sj1_token_predistortion_joint_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_token_predistortion_joint_31st_move/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_token_predistortion_joint_31st_move/SEAL_ddm_sj1_token_predistortion_joint_contest_cuda.json` (sha verified: True).

## What this does NOT claim

This move INCREASES the archive (174,786 → 180,904 B, +6,118 B) and pays through distortion: seg −0.008132 S, pose −0.000247 S, rate +0.004074 S. It is
therefore a move toward the score, not toward the rate corner — the sub-0.12 rate demand at held distortion GROWS by 6,118 B while the distortion corner shrinks.
The T4 row is authority for both d_seg and d_pose (the local instruments predicted them; the row decides). The 566-pair admission optimum (−5.2e-5 better than
the full 600) is NOT claimed: its rate leg is a per-pair sum along the full-edit coder trajectory, an estimate until its own encode pair runs. Pass 3 (a second
round on the residual) is running and is a successor candidate, not part of this row. `[contest-CPU]` stays RECORD-WITH-REASON. PR #140 is eight moves behind.

## Next from here

sj1's pass 3 on the residual field (convergence rule < 1 % not yet reached; the marginal rate is well under break-even) → a successor candidate on THIS base with
its own re-solve and encode pair; the 566-pair subset with its own encode. cl3's ladder is closed (capacity both directions; seeds ≤ 73 B). The carrier lattice
lever is exhausted at ×16. md4 confirmed the born vehicle's persistent set is data-anchored. Remaining named doors: the per-atom quantizer step of the carrier
(3,731 B at fixed alphabet), the packed Rice-k field width, the renderer at optimal form (pose in the loop + re-solve), and now the composition of pre-distortion
with the renderer (the flips that pre-distortion cannot repair are the renderer's).

Equations leg (`tac.canonical_equations`): token_predistortion_multipass_yield_v1 (sj1): T4 anchor — pass 2a repairs 40.39 % of flips at 1.229 cells/changed token for 6.23 bits/token against a 12.52-bit break-even; the carrier re-solve on the candidate renders lands 0.936× base d_pose (the composition law, numerical, on the seg actuator)

Own-vehicle frontier: **S 0.1398140172839628 @ 180,904 B [contest-CUDA T4 n600]**, archive sha `42aa84b5…5f2f8`.
