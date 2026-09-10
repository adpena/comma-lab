# THIRTY-NINTH POINTER MOVE — S 0.13766931482209038 @ 180,186 B [contest-CUDA T4 n600]: rp1 rate-directed token pre-distortion (473 argmax-neutral token changes over 253 pairs, priced by real re-encode through tc1's mixer, carrier re-solved; seg exactly 0, pose below base) — S 0.1376693148220904 @ 180,186 B, Δ −2.21e-4 vs move 38 (11× bar) (2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M24F980TS3S60D0QTAQX2TR0`. Lane `ddm_rp1_t4_rate_directed_predistortion_20260910`. Modal wall 1065.5 s. Archive sha `8877f75d87bf25b410264e08682959c7710cf677307bd5452039ce53835f6bf4`, 180,186 B. Runtime tree `5e1e6dac4b0391d77596883e4d9b4b1988b22ccc8ad0b07e544e7ad76ad38927`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,186/37,545,489 | 0.11997846132727157 |
| seg 100·0.00010698 | 0.010698 |
| pose √(10·4.89e-06) | 0.0069928534948188355 |
| **S** | **0.13766931482209038** |

| | ddm_sj1_t4_token_predistortion_pass5_20260909 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.1378902937630636 | 0.13766931482209038 | **-0.00022097894097322657** |
| d_seg | 0.00010632 | 0.00010698 | 6.599999999999971e-07 |
| d_pose | 5.06e-06 | 4.89e-06 | -1.7000000000000007e-07 |
| bytes | 180,436 | 180,186 | -250 |

## Projection fidelity

Projected 0.1376665464876166; realized − projected = 2.768334473796097e-06. optimistic by +2.77e-6: the 8-dp pose print (4.89e-6 vs 4.886129e-6 resolved) — the pose-print class (pc2 +1.84e-6, sj1 pass 5 +7.73e-6)

## The mechanism

rp1: RATE-DIRECTED token pre-distortion — sj1's realized instrument with the objective inverted. Candidate single-token changes ranked by the shipped coder's marginal bit saving, accepted ONLY where the pair's realized argmax is identical on all 196,608 cells (strict neutrality, realized through the receiver's own renderer + frozen cpu_torch SegNet: neutrality is 4.36 % of 19,200 proposals at n600 and FLAT in rank — only the prize varies, 40×; SegNet's influence radius is 343 cells so isolation cannot predict it), then per-pair Lagrange selection with rate as the objective and the resolved pose as the cost, the selected subset priced by REAL twin re-encode through tc1's live mixer, the carrier re-solved per pair from the live coefficients. Admitted: 473 argmax-neutral token changes over 253 pairs; three legs each from its own measurement: seg EXACTLY 0 (0 of 117,964,800 cells differ on the candidate's own parse-back; 12,614 = 12,614 on the move-37 base), pose −5.16e-5 (5.049766e-6 → 4.886129e-6, BELOW base), rate −202 B net (−211 B tail, +9 B carrier). The finding: an adaptive coder pays out 0.15–0.27 of a first-order −log2 p saving when a surprise is taken OUT of the field (838 neutral changes promised −827.1 B, returned −144.65 B; Lagrange selection raised the payout 0.1445 → 0.2663), while sj1 measured 1.279× for changes that ADD surprise — the asymmetry law, both tails on one object. Exact T4: seg 0.00010698, pose 4.89e-6, 180,186 B → S 0.1376693148220904 (Δ −2.21e-4 vs move 38, 11.05× bar; −2.48e-4 vs move 37; projection +2.77e-6 optimistic, pose-print class). Equation `rate_directed_predistortion_yield_v1`.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.017669314822090387. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.017690853494818835: archive ≤ 153,649.9 B → **-26,536.1 B**.
- **DISTORTION corner** at held bytes 180,186: distortion ≤ 2.1539e-05 → **821.4× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **32.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_rp1_t4_rate_directed_predistortion_20260910/MODAL_REMOTE_RESULT.json` (sha `41d24304787b7c17505550bb17e516c981e985d8822c0b74e329e1e6c67c7b5f`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_rp1_rate_directed_predistortion/candidate/candidate_runtime/archive.zip` (sha `8877f75d87bf25b410264e08682959c7710cf677307bd5452039ce53835f6bf4`, 180,186 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_rp1_rate_directed_predistortion/SEAL_ddm_rp1_rate_directed_predistortion_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_rp1_rate_directed_predistortion/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_rp1_rate_directed_predistortion/SEAL_ddm_rp1_rate_directed_predistortion_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: composition with sj1's pass-5 subset (move 38; 13-pair overlap, zero cell collisions; the tails will NOT add — the registered yield law says per-position prices do not sum; the naive sum 0.1376395 @ 180,234 B is a number to be missed, not quoted); the 44 pose-bound dropped pairs (−27.71 B stranded; frame-0 selector target); that the first-order ranking is a price (it is a ranking: 0.14–0.27× when removing surprise); the corner (−26,909 B → now ≈ −26,7xx B, re-derive at this move): this row covers 0.75 % of it; any CPU-axis row (single-axis waiver; lineage moves 24–39 all contest-CUDA T4); seal re-validation against any pointer other than move 37 (the seal binds 670d38d0 with zero tolerance — it is a receipt of the move-37-base measurement).

## Next from here

sj1's 42-pair pass-5 subset re-bases onto THIS field by re-verification (13 overlapping pairs; re-solve over the 282-pair union from move 39's coefficients; twin re-price against this tail; re-seal) — the composition receipt is the measurement of whether the two tails add. Then the FRAME-0 selector round on the move-39 field (sizing: the shipped mode is the default on 81/84 of sj1's dropped pairs; 74 have a strictly better mode; rp1's 44 pose-bound pairs join; selector bytes charged; carrier re-solve; three-column pose leg). tc2 prices the lane-boundary context map the rp1 census named (Lane→Road = 37 % of the payable attribution; oracle bound first). bnd3 decides the level of bnd2's closure. PR #140 swap packet gated on the operator's one-line confirm (16 moves behind).

Equations leg (`tac.canonical_equations`): rate_directed_predistortion_yield_v1

Own-vehicle frontier: **S 0.13766931482209038 @ 180,186 B [contest-CUDA T4 n600]**, archive sha `8877f75d…f6bf4`.
