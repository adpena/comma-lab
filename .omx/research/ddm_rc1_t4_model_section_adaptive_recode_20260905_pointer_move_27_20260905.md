# TWENTY-SEVENTH POINTER MOVE — S 0.14666350774473783 @ 178,249 B [contest-CUDA T4 n600]: rc1 adaptive recode of the two model sections: −1,733 B at zero distortion (the model sections had never been entropy-coded) (2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M1SG7CY107YXHSFS26NWV69T`. Lane `ddm_rc1_t4_model_section_adaptive_recode_20260905`. Modal wall 557.1 s. Archive sha `1438049e3655fbcfa8eb289fa51ac58f834d72d8a09586353663cea68e57c122`, 178,249 B. Runtime tree `064bb2081967cb4db3b0524c22a05a27d69c30bc98f19679560b046e68647835`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·178,249/37,545,489 | 0.11868869253507391 |
| seg 100·0.00020139 | 0.020139 |
| pose √(10·6.14e-06) | 0.007835815209663893 |
| **S** | **0.14666350774473783** |

| | ddm_cl2_t4_lambda1_control_repack_20260905 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.14781744131049854 | 0.14666350774473783 | **-0.001153933565760712** |
| d_seg | 0.00020139 | 0.00020139 | 0.0 |
| d_pose | 6.14e-06 | 6.14e-06 | 0.0 |
| bytes | 179,982 | 178,249 | -1,733 |

## Projection fidelity

Projected 0.14666350774473783; realized − projected = 0.0. rate-only lossless projection; decoded field, carrier and tail byte-identical; projection error 0.0

## The mechanism

The two MODEL sections of the RX1 container — the SM3R renderer body (66,339 params packed as 3/4-bit signed codes, prune masks, fp16 scales) and the
IHS1 HPAC probability model (20,416 integer rows) — had only ever been coded by generic compressors (Brotli q11, XZ, ck2's parameter-free 2-plane byte
de-interleave). rc1 codes their PACKED INTEGER CODES with an adaptive per-group tree coder (contexts are pure geometry: group, position; no transmitted
model, rule-118 clean) and restores each body byte-for-byte in the receiver before any parser runs: semantic 30,856 → 30,246 B (−610), hpac 13,466 →
12,343 B (−1,123). Brotli sits within 0.8 % / 0.3 % of the code streams' order-0 entropy only where the packing width is CONSTANT; where the width changes
every few hundred symbols (IHS1) it cannot, which is why the smaller body gave the larger credit. xz −9e and zstd −22 both LOSE to Brotli on these bodies.
Decoded field (token sha cc10a7b0…), carrier and tail are byte-identical to cl2; twin encode byte-identical; +0.11 s inflate. Attempt 1 on T4 failed in
3.6 s because `runtime/f26_inflate.py` read the semantic magic before the codec dispatch — the arm's identity check had run through the library path, not
`inflate.sh`; the fix restores the rider before the magic guard, and the public path was exercised locally to the CUDA gate on both trees before the
re-seal. Seal `SEAL_ddm_rc1_model_section_adaptive_recode_contest_cuda.json` (file sha 9ef5b420…, canonical digest 32968e32…); memo
`ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md`; commits 8979e18aa, feb9e5804.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.02666350774473783. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.027974815209663894: archive ≤ 138,205.2 B → **-40,043.8 B**.
- **DISTORTION corner** at held bytes 178,249: distortion ≤ 0.0013113 → **21.3× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **1,969.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_rc1_t4_model_section_adaptive_recode_20260905_r2/MODAL_REMOTE_RESULT.json` (sha `95c1b7eb8f801d509afad7a8fa460fd0a54b71ac237100b82d4abb5cbe8c04ce`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_rc1_model_section_adaptive_recode/staged_runtime/archive.zip` (sha `1438049e3655fbcfa8eb289fa51ac58f834d72d8a09586353663cea68e57c122`, 178,249 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_rc1_model_section_adaptive_recode/SEAL_ddm_rc1_model_section_adaptive_recode_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_rc1_model_section_adaptive_recode_27th_move/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_rc1_model_section_adaptive_recode_27th_move/SEAL_ddm_rc1_model_section_adaptive_recode_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Rate-only: d_seg 0.00020139 and d_pose 6.14e-06 are IDENTICAL to cl2 by construction (lossless recode) and were confirmed on T4. This is not a model change
and not evidence about any model's quality. It is 4.1 % of the sub-0.12 rate demand. `[contest-CPU]` stays RECORD-WITH-REASON (the 1,800 s inflate timeout;
+0.11 s does not change that). PR #140 still carries afr1's bytes (now four moves behind); a public update is the operator's decision. Order-1 contexts on
these bodies are CLOSED at formulation scope (context dilution converted 10 B of ~1,139 B of first-order structure); a semi-static or depth-mixing prior
is untested (rc1 ITEM 1). The seal contract validated custody on an archive the contest path could not decode — three apparatus ITEMs are registered
(`ddm_seal_contract_gap_inflate_sh_smoke_directive_20260905.md`).

## Next from here

Live on the same object: sj1 (multi-pass token pre-distortion; pass 2a at 46 % flip repair on the first 93 pairs, ~4 h to finish, then jg5 admission +
carrier re-solve + exact re-encode — the seg-debt pool, 30 KB-eq) and pc1 (pose-carrier efficiency: basis/coefficient precision, generated basis, learned
rank-8 — the 22 KB carrier pool). bd1 (bidirectional temporal context) CLOSED at family scope: the label field is piecewise-constant over tens of pairs
(P(32)/P(1) = 1.07–1.13), so the token stream is at its floor under any predictor that reads other frames' labels. Every later candidate re-prices on
THIS object (178,249 B): the carrier and tail are byte-identical, so sj1's and pc1's byte deltas compose additively with rc1's coder. Named doors
remaining: rc1 ITEM 1 (semi-static/depth-mixing prior on IHS1), the renderer width distillation at optimal form (pose in the loop + re-solve; wd2's
w64 toy was INSTANCE scope), and the joint field+model escape.

Equations leg (`tac.canonical_equations`): model_section_adaptive_recode_ceiling_v1: T4 anchor confirms −1,733 B (semantic −610, hpac −1,123); credit tracks packing WIDTH STABILITY, not the raw coder win; ΔS −1.15393e-3 = exactly 1,733 × 25/37,545,489

Own-vehicle frontier: **S 0.14666350774473783 @ 178,249 B [contest-CUDA T4 n600]**, archive sha `1438049e…7c122`.
