# THIRTY-FIFTH POINTER MOVE — S 0.13867171823146562 @ 181,521 B [contest-CUDA T4 n600]: sj1 pass 4 token pre-distortion, 112-pair Lagrange subset + per-pair carrier re-solve: −252 cells at +148 B, pose a credit by selection (35th pointer move) (2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M23NAMJRKJWPRWRJ77WF3R55`. Lane `ddm_sj1_t4_token_predistortion_pass4_20260909`. Modal wall 596.0 s. Archive sha `b0ca809ce2c657dfce97e73148a83b9b20c128461ced4c1f6ce1b386ddfd1d20`, 181,521 B. Runtime tree `5797e46a509261350ca7dfe2d1771f50b205d913e48cd266c7042dcd108ba1c7`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·181,521/37,545,489 | 0.12086738302968966 |
| seg 100·0.00010698 | 0.010698 |
| pose √(10·5.05e-06) | 0.007106335201775948 |
| **S** | **0.13867171823146562** |

| | ddm_pc2_t4_carrier_scales_resolve_20260909 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13882326433317044 | 0.13867171823146562 | **-0.00015154610170481364** |
| d_seg | 0.00010913 | 0.00010698 | -2.149999999999998e-06 |
| d_pose | 5.1e-06 | 5.05e-06 | -5.000000000000037e-08 |
| bytes | 181,373 | 181,521 | +148 |

## Projection fidelity

Projected 0.13867280587520867; realized − projected = -1.0876437430418218e-06. projection +1.09e-6 PESSIMISTIC (first of the wave); seg 12,614 = 12,614; pose print 5.05e-6

## The mechanism

sj1 pass 4 (`.omx/research/ddm_sj1_multipass_token_predistortion_20260905.md` §15–§22; seal `SEAL_ddm_sj1_token_predistortion_pass4_contest_cuda.json`, sha c887162d…): a fourth pass of single-cell token pre-distortion under realized acceptance (render → frozen SegNet → keep iff flips fall) over all 600 pairs on the move-34 body, repairing 423 of the 12,866 residual cells (3.288 % vs 2.586 % projected) with 415 changed tokens at a MEASURED 6.4000 bits/token (twin encode byte-identical; the pre-registered 4.5176 was falsified upward — the marginal price is RISING, 6.23 → 5.31 → 6.40, cheapest repairs first). The carrier was re-solved per pair from pc2's coefficients (`refine_pair`; 1,062 of 7,200 coordinates moved; stale d_pose 1.432e-4 = 28.1× base vs pass 3's 486×). The Lagrange subset sweep kept 112 pairs (252 of the 423 cells) where the seg gain and the re-solve both help: that turns the full field's pose COST (+1.044e-4, net only −3.31e-5) into a pose CREDIT (−3.53e-5), a 1.40e-4 swing from selection alone. Non-admitted pairs carry the live row's planes (the cured subset writer). Priced by real encode (+142 B stream vs a +136.3 ledger sum; +6 B carrier). Seg on the shipped bytes: 12,614 cells predicted = 12,614 measured (second pass in a row with zero disagreement). T4: d_seg 0.00010698, d_pose 5.05e-6, 181,521 B.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.018671718231465628. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.017804335201775948: archive ≤ 153,479.4 B → **-28,041.6 B**.
- **DISTORTION corner** at held bytes 181,521: distortion ≤ -0.00086738 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-1,302.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass4_20260909/MODAL_REMOTE_RESULT.json` (sha `b149b4303b67451aedf4689be9d94f40d2006acc0149737ae5ec31fe0229a92c`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate_pass4/candidate_runtime/archive.zip` (sha `b0ca809ce2c657dfce97e73148a83b9b20c128461ced4c1f6ce1b386ddfd1d20`, 181,521 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/SEAL_ddm_sj1_token_predistortion_pass4_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass4_20260909/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass4_20260909/SEAL_ddm_sj1_token_predistortion_pass4_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: the full 600-pair field (nets −3.31e-5 only; pose is a cost there) — the subset is the row. Not claimed: the pre-registered projection's reasoning — two of three inputs were wrong (bits/token 1.417× high; pose leg sign), the CONTINUE decision held on its floor margin but the floor was mislabelled (pose can go positive). Not claimed: any transfer of the pass-3 "falling price" model. Not claimed: convergence — pass 5 projects −4.1e-5 subset-only (full field positive) with a model-error band whose pessimistic end misses the admit bar; pass 6 would fall under the 1 %/pass rule. Not claimed: a CPU-axis number. Projection residual: T4 −1.09e-6 BELOW the projection (the first pessimistic projection; prior two were +5.27e-6 / +4.16e-6 optimistic).

## Next from here

1. cmp1 (codex, queued rank 0): compose rc3's model section (−201 B) + tc1's tail-coder re-encode of THIS tail (−549 B on the previous field) onto this tree → one seal → one T4 call.
2. fe1 re-bases its FiLM candidate (15 pairs re-verified on this tree's parse-back argmax; three builds: FiLM / FiLM+331 / 331-alone; smallest measured archive) after cmp1.
3. sj1: the two-cell SLIDE family sizing (12 pairs; stop rule ≥ 5 % of refused cells at ≥ 1.5 cells/slide) decides whether pass 5 is singles or singles+slides; pass 5 is subset-only and its admission may fire the stop. Pass-4 census: 89.46 % of the residual explicitly refused ~4× — the renderer door (rw1's fp32 per-row scales, receiver change) inherits it.
4. Binding arithmetic re-derived at this move by the packet (pointer line).

Equations leg (`tac.canonical_equations`): tac.canonical_equations: token_predistortion_multipass_yield_v1 (anchors: pass 4 3.288 %/1.0193 cells/token/6.4000 bits/token — price rising; subset sweep flips the pose leg sign) + exchange_ratio_noise_floor_v1

Own-vehicle frontier: **S 0.13867171823146562 @ 181,521 B [contest-CUDA T4 n600]**, archive sha `b0ca809c…d1d20`.
