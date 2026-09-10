# THIRTY-EIGHTH POINTER MOVE — S 0.1378902937630636 @ 180,436 B [contest-CUDA T4 n600]: sj1 pass 5 (42-pair Lagrange subset of the fifth single-token pre-distortion pass, priced by real re-encode through tc1's mixer, carrier re-solved) — S 0.1378902937630636 @ 180,436 B, Δ −2.70e-5 (1.35× bar) (2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M24EG4R59D365KWRT7538XA8`. Lane `ddm_sj1_t4_token_predistortion_pass5_20260909`. Modal wall 1185.4 s. Archive sha `eae99e0083129a911103bf691bdbf3763b9b7c0e3e365a6e3dc1cd2d2d4eec07`, 180,436 B. Runtime tree `376de794c9bbb4c28a9020c3dc3854ee26803eceae9e3c2deb4517c667aed997`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,436/37,545,489 | 0.1201449260655521 |
| seg 100·0.00010632 | 0.010631999999999999 |
| pose √(10·5.06e-06) | 0.007113367697511495 |
| **S** | **0.1378902937630636** |

| | ddm_cmp2_t4_sm1_semantic_coder_20260909 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13791730003757818 | 0.1378902937630636 | **-2.7006274514573825e-05** |
| d_seg | 0.00010698 | 0.00010632 | -6.599999999999971e-07 |
| d_pose | 5.05e-06 | 5.06e-06 | 9.999999999999904e-09 |
| bytes | 180,388 | 180,436 | +48 |

## Projection fidelity

Projected 0.13788255886142803; realized − projected = 7.734901635580993e-06. optimistic by +7.73e-6: the 8-dp component print (pose 5.06e-6 vs 5.0516e-6 resolved; seg 0.00010632 vs 0.000106302) — the pose-print class seen at pc2 (+1.84e-6)

## The mechanism

sj1 pass 5: a fifth realized single-token pre-distortion pass on the move-37 field (237 cells repaired by 235 token changes, seg −2.01e-4 on the instrument), priced by REAL re-encode through tc1's live shared mixer (control re-encode of the shipped pass-4 field reproduced the live tail byte-identically, 119,779 B; twins byte-identical; rebuilt control archive = the live sha). The FULL 126-pair field FAILS after the per-pair carrier re-solve (resolved pose +1.328e-4 vs stale +2.483e-2 — the re-solve removed 1,235× the bar and the field still loses: net +8.42e-5). The Lagrange sweep keeps 42 of 126 pairs (80 of 237 cells, 47 of 229 tail bytes; 4.82 bits/token vs a 10.45 break-even) whose seg yield covers rate with a near-neutral resolved pose (+1.31e-6), then prices THAT subset by its own twin re-encode (+47 B measured vs +48.96 B ledger-sum: the ledger residual is small and either-signed, −1.96/+5.70/+19.6 B across passes 5/4/3). Seg gate on the shipped bytes: 12,534 cells predicted = 12,534 measured. Only the carrier section moved among frame-1 sections (42 pairs, 205 coordinates); hpac/semantic/tail identical. Exact T4: seg 0.00010632, pose 5.06e-6, 180,436 B → S 0.1378902937630636 (Δ −2.70e-5, 1.35× bar; projection 0.13788256 was optimistic by 7.7e-6, the 8-dp print class).

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.017890293763063614. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.017745367697511494: archive ≤ 153,568.0 B → **-26,868.0 B**.
- **DISTORTION corner** at held bytes 180,436: distortion ≤ -0.00014493 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-217.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass5_20260909/MODAL_REMOTE_RESULT.json` (sha `fbc62d6f75db684cb0c2d1496671f058706b919e08b92d6b0b57b47613118339`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass5_price/candidate_pass5/candidate_runtime/archive.zip` (sha `eae99e0083129a911103bf691bdbf3763b9b7c0e3e365a6e3dc1cd2d2d4eec07`, 180,436 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass5_price/SEAL_ddm_sj1_token_predistortion_pass5_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_pass5_price/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_pass5_price/SEAL_ddm_sj1_token_predistortion_pass5_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: any frame-0 selector re-selection (not in this chain; measured separately as a sizing on the full field's 84 dropped pairs: 74 have a strictly better mode, the shipped mode is the default on 81/84 — bytes uncharged, batch-1 instrument); recovery of the 84 dropped pairs; a pose credit (the subset's resolved pose is near-neutral, unlike pass 4's −3.5e-5 credit); pass 6 (the singles family CONVERGED: 80/12,614 = 0.634 % < the pre-registered 1 %); any CPU-axis row (single-axis waiver, lineage moves 24–38 all contest-CUDA T4); any composition with rp1's concurrent candidate (13-pair overlap, zero position collisions — re-base owed by the smaller win); that the ledger sum is a price (it is a ranking; the subset was re-encoded).

## Next from here

rp1's exact row (call fc-01M24F980TS3S60D0QTAQX2TR0; 180,186 B; projected 0.13766655) lands next and, if lower, becomes the pointer; the smaller win (this subset) then re-bases onto rp1's field by re-verification. Then the FRAME-0 selector round on the promoted field: all 600 pairs sized on the current renders, the set chosen by Lagrange with the selector-blob bytes charged, carrier re-solve against the final pointer, three-column pose leg (stale / carrier-resolved / frame-0-reselected+resolved). tc2 (codex) prices the lane-boundary context map the rp1 census named (oracle bound first). PR #140 swap packet stays gated on the operator's one-line confirm (now 15 moves behind).

Equations leg (`tac.canonical_equations`): token_predistortion_multipass_yield_v1

Own-vehicle frontier: **S 0.1378902937630636 @ 180,436 B [contest-CUDA T4 n600]**, archive sha `eae99e00…eec07`.
