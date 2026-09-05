# TWENTY-EIGHTH POINTER MOVE — S 0.1451981569076111 @ 176,448 B [contest-CUDA T4 n600]: pc1 lattice ×4 with the full re-solve: −1,801 B AND d_pose −6.6 % — the carrier's atoms untouched, its coefficients re-solved on a coarser lattice (custody twin of fc-01M1SHRJ45…) (2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M1SJPXG6KXN0WTWQ9TZ4GZY6`. Lane `ddm_pc1_t4_lattice_x4_on_rc1_20260905`. Modal wall 538.9 s. Archive sha `891add546f5cf0943929b566f29dd4318f1d8b2ab76ae05183d8189098880f40`, 176,448 B. Runtime tree `01b088005996b97dfbff831348c8dc5896b06f809a31a2aee9cd701182e47eb8`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·176,448/37,545,489 | 0.11748948056050089 |
| seg 100·0.00020139 | 0.020139 |
| pose √(10·5.73e-06) | 0.007569676347110225 |
| **S** | **0.1451981569076111** |

| | ddm_rc1_t4_model_section_adaptive_recode_20260905 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.14666350774473783 | 0.1451981569076111 | **-0.0014653508371267332** |
| d_seg | 0.00020139 | 0.00020139 | 0.0 |
| d_pose | 6.14e-06 | 5.73e-06 | -4.0999999999999946e-07 |
| bytes | 178,249 | 176,448 | -1,801 |

## Projection fidelity

Projected 0.1452005596984867; realized − projected = -2.402790875599692e-06. pose-changing projection from cpu_torch n600 (5.727914e-6); realized − projected = −2.4e-6 = the evaluator's 3-sig-fig pose print; two T4 rows identical

## The mechanism

The pose carrier's 600 × 12 coefficients (int12, AR1 + Rice, 9,830 B) were re-quantized onto a lattice coarsened ×4 (12 → 10 bits per coefficient) and
then RE-SOLVED for every pair with the full n600 damped Gauss–Newton (`ddm_jg5.refine_pair`, the fs2 solver verbatim) against the frozen PoseNet on the
shipped renders — the twelve basis atoms are bit-identical to the shipped carrier, which is why this is the one variant of five that survived. The
re-solve did not merely recover the coarsening cost (4.79× without it), it landed BELOW the base: d_pose 6.14e-6 → 5.73e-6 (T4; 5.727914e-6 cpu_torch
n600) while the coefficient payload fell by 1,801 B. Token tail, model sections (rc1's adaptive coding) and basis are byte-identical to the 27th-move
archive; only the carrier coefficient block moved. Determinism twin: 24/24 pairs reproduce the codes bit-identically. Seal
`SEAL_ddm_pc1_v3_lattice_x4_resolved_on_rc1.json` (canonical digest cd454d01…); memo `ddm_pc1_pose_carrier_efficiency_20260905.md`; law
`pose_carrier_basis_rate_fidelity_exchange_v1`; 12 commits (edeef17b0…fdbd0ced1).
Two independent T4 rows exist on these bytes: `fc-01M1SHRJ45TJRMA9G2YXCE4MTW` (lane `ddm_pc1_t4_v3_lattice_x4_on_rc1_20260905`, refused at promotion by the
checkpoint-maturity gate because the lane id's variant token "v3" reads as an untagged vehicle version) and this custody twin
`fc-01M1SJPXG6KXN0WTWQ9TZ4GZY6` — identical components (d_seg 0.00020139, d_pose 5.73e-06, 176,448 B, S 0.1451981569076111): the archive scores
deterministically on T4. The fire tool now refuses non-promotable lane ids before spend (commit 9b4a8d898).

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.025198156907611097. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.027708676347110224: archive ≤ 138,604.9 B → **-37,843.1 B**.
- **DISTORTION corner** at held bytes 176,448: distortion ≤ 0.0025105 → **11.0× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **3,770.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_pc1_t4_lattice_x4_on_rc1_20260905_custody/MODAL_REMOTE_RESULT.json` (sha `c8f643b826770d8ad95b2b198f8687645ee736f5cd0f4962f867dca613eedfc1`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency/retained/v3_on_rc1_candidate_runtime/archive.zip` (sha `891add546f5cf0943929b566f29dd4318f1d8b2ab76ae05183d8189098880f40`, 176,448 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency/SEAL_ddm_pc1_v3_lattice_x4_resolved_on_rc1.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_pc1_lattice_x4_28th_move/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_pc1_lattice_x4_28th_move/SEAL_ddm_pc1_v3_lattice_x4_resolved_on_rc1.json` (sha verified: True).

## What this does NOT claim

This is a POSE-CHANGING move: the T4 row is authority for d_pose (5.73e-6 printed at 3 sig figs; the projection used 5.727914e-6, so the realized S differs
from the projection by −2.4e-6 — print rounding, not instrument error). It is not evidence that the basis can be touched: V1/V2 (basis 5→4/3 bits) and
V4 (a zero-byte generated DCT basis, d_pose ≥ 0.9986, 39,748× past break-even) and V5 (rank-8 SVD refit) are all REFUSED with measured numbers — the
carrier's 22 KB buy a specific positioned 12-dim subspace (zero carrier: d_pose 52.1; the carrier pays for itself 1,556×). 4.4 % of the sub-0.12 rate
demand. `[contest-CPU]` stays RECORD-WITH-REASON. PR #140 is now five moves behind; a public update is the operator's decision.

## Next from here

Same arm, same machinery, already running: lattice ×8 (priced −2,693 B, break-even d_pose 9.27e-6) and ×16 (−3,484 B, 1.03e-5) with the full re-solve on
THIS base; the same-object pose ceiling 1.694e-5 binds. sj1 (multi-pass token pre-distortion) continues on the seg-debt pool and must re-solve its edited
pairs' carriers FROM THIS lattice and these coefficients (the carrier block is no longer cl2's). Open on the carrier per pc1's ITEMs: the per-atom
quantizer step (moves 3,731 B at fixed alphabet), the packed Rice-k field width, the basis_scales blast radius. Closed today at family scope: temporal
context on the token coder (bd1), basis precision/rank/generated bases on the carrier (pc1).

Equations leg (`tac.canonical_equations`): pose_carrier_basis_rate_fidelity_exchange_v1: T4 anchor (two rows) — lattice ×4 + re-solve = −1,801 B and d_pose 6.14e-6→5.73e-6; coarsening alone 4.79× worse, re-solve recovers 5.13×

Own-vehicle frontier: **S 0.1451981569076111 @ 176,448 B [contest-CUDA T4 n600]**, archive sha `891add54…80f40`.
