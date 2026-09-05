# TWENTY-SIXTH POINTER MOVE — S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]: cl2 λ=1.0 HPAC-prior control repack: rate-only −41 B on the shipped fs2 mixer, distortion held (2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M1S4PBEPBKQJVEPWVRDJHGNT`. Lane `ddm_cl2_t4_lambda1_control_repack_20260905`. Modal wall 545.9 s. Archive sha `08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e`, 179,982 B. Runtime tree `0b21f8c3c38043c0977a17dd6730f7e94817e5b53f6bdc1539543f155f44e5e7`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·179,982/37,545,489 | 0.11984262610083464 |
| seg 100·0.00020139 | 0.020139 |
| pose √(10·6.14e-06) | 0.007835815209663893 |
| **S** | **0.14781744131049854** |

| | ddm_ps2_t4_snapshot_fireproof_20260905 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.14784474152757654 | 0.14781744131049854 | **-2.7300217078002342e-05** |
| d_seg | 0.00020139 | 0.00020139 | 0.0 |
| d_pose | 6.14e-06 | 6.14e-06 | 0.0 |
| bytes | 180,023 | 179,982 | -41 |

## Projection fidelity

Projected 0.14781744131049854; realized − projected = 0.0. rate-only projection: d_seg/d_pose held by decoded-field identity and a byte-identical parse-back render; projection error 0.0

## The mechanism

The HPAC prior (the context-mixing arithmetic coder's learned model, 13,515 B IHS1-packed on the shipped fs2 mixer) was re-trained under
the reproducing instrument cl2 identified: the JF1 warm-start law (60-epoch cosine from the shipped epoch-634 EMA init `ff2d3e45…`, seed
20260716, batch 8, QAT 0.5, λ = 1.0) on the CURRENT token field, held bit-identical. Only two numbers move: the packed model (13,466 B, −49 B)
and the RC64 token stream (113,419 B, +8 B) → joint 126,885 B vs the shipped 126,926 B = −41 B; archive 179,982 B. Every other section is the
shipped byte. The receiver-copy decode reproduces the field byte-for-byte and the parse-back render (3,662,409,600 B, sha `f86bfaf3…`) is
identical to the shipped render, so d_seg and d_pose are held by construction and the T4 row is a custody confirmation of a rate-only move.
A fresh-root twin of the same law reproduced the stream (`e07274ca…`) and the archive (`08ec8533…`) byte-exactly. Seal
`SEAL_ddm_cl2_lambda1_control_repack_contest_cuda.json` (sha `3cf630a6…`); memo `ddm_cl2_hpac_prior_capacity_ladder_on_shipped_object_20260905.md`.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.027817441310498542. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.027974815209663894: archive ≤ 138,205.2 B → **-41,776.8 B**.
- **DISTORTION corner** at held bytes 179,982: distortion ≤ 0.00015737 → **177.8× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **236.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_cl2_t4_lambda1_control_repack_20260905/MODAL_REMOTE_RESULT.json` (sha `982336582d12af43c49cc53ed89fd4453792143efafd99a8c4381e0255d9f720`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/candidate_archive.zip` (sha `08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e`, 179,982 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/SEAL_ddm_cl2_lambda1_control_repack_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_cl2_lambda1_control_repack_26th_move/candidate_archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_cl2_lambda1_control_repack_26th_move/SEAL_ddm_cl2_lambda1_control_repack_contest_cuda.json` (sha verified: True).

## What this does NOT claim

This is NOT a capacity win and NOT evidence for the prior-capacity ladder: the ladder's bigger direction is FALSIFIED (λ 1.0→0.5: +350 B model
bought +156 B of stream; secant +0.446 against the −1 the law `hpac_prior_capacity_slope_v1` requires; net +465 B). The −41 B is the
retrain/pack-size residual of one control at one seed — training noise of that order, banked because the archive is real and its output is
byte-identical. It does not open the rate corner (demand −41,818 B; this row is 0.098 % of it). `[contest-CPU]` stays RECORD-WITH-REASON
(the 1,800 s inflate timeout); no CPU claim. PR #140 still carries afr1's bytes; a public update is the operator's decision.

## Next from here

cl3 (live, Opus) prices the untested half of the same axis on the same object: the SMALLER prior (λ = 2.0, then 4.0 iff 2.0 pays) and seed
selection at λ = 1.0 (min-of-3), each through cl2's exact price/verify path with a twin for the winner — prior-law prediction −350…−50 B for
λ = 2.0, −40…−90 B beyond this control for the seed minimum, whole-axis falsifier if λ = 2.0 nets ≥ 0. md3's different-initialisation burn
cell (Metal, terminal ~18:50Z) decides whether the born vehicle's 62 % persistent error partition is data- or init-anchored (J ≥ 0.70 vs
≤ 0.45). The rate corner's named doors remain: architectural receptive-field change, a different generator form, joint field+model escape.

Equations leg (`tac.canonical_equations`): hpac_prior_capacity_slope_v1: control anchor confirmed on T4 (J 126,885 = −41 B; ΔS −2.7300216e-5 = exactly 41×25/37,545,489); the ladder's bigger direction stays FALSIFIED (secant +0.446 vs −1)

Own-vehicle frontier: **S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]**, archive sha `08ec8533…8ad4e`.
