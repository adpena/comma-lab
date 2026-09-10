# FORTY-THIRD POINTER MOVE — S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600]: sj1 pass 6: token pre-distortion seg-Lagrange subset (161 pairs / 344 cells / 335 tokens) re-based onto move 42's field with the carrier re-solved and the pose read on the RESOLVED pose — seg 12,540→12,196 cells exactly as admitted, +228 B, pose 4.59e-06 below base (2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M2691VTJYKNCXX01CKKTN44N`. Lane `ddm_sj1_t4_token_predistortion_pass6_20260910`. Modal wall 1148.5 s. Archive sha `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`, 180,466 B. Runtime tree `a726739a52452f8824c4eb9f98322f927b8b5d2d4c6ac1717911f61b841a9803`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,466/37,545,489 | 0.12016490183414577 |
| seg 100·0.00010345 | 0.010345 |
| pose √(10·4.59e-06) | 0.00677495387438173 |
| **S** | **0.1372848557085275** |

| | ddm_rp1_round2_rate_directed_predistortion_k192_20260910 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.1374765052591843 | 0.1372848557085275 | **-0.00019164955065681388** |
| d_seg | 0.00010637 | 0.00010345 | -2.920000000000006e-06 |
| d_pose | 4.66e-06 | 4.59e-06 | -7.000000000000018e-08 |
| bytes | 180,238 | 180,466 | +228 |

## Projection fidelity

Projected 0.13728267021291818; realized − projected = 2.1854956093192435e-06. sealed projection on the resolved pose; exact −1.9165e-4 vs projected −1.938e-4, the 2.2e-6 gap is the T4 pose-print class the last three packets measured

## The mechanism

Token pre-distortion pass 6 (seg Lagrange subset) re-based onto move 42's field, with the carrier re-solved on the edited field and the pose read on the RESOLVED pose. The subset: 161 pairs / 344 cells / 335 tokens, admitted under the same pre-registered rule as passes 3–5 (per-pair Lagrange price against the real coder's tail). Seg leg measured on the shipped decode: 12,540 → 12,196 flipped cells (−2.9180e-4 S), prediction = measurement exactly (12,196 = 12,196). Rate: +228 B (+220 B token stream by twin encode against move 42's tail, +8 B carrier), +1.4649e-4 S, 5.2537 bits/token against a 10.4585 break-even (margin 1.991×). Pose: resolved 4.586763e-06 vs base 4.649477e-06 on move 42's own configuration (a −4.6142e-5 S credit; gate ratio 0.99774). Projected net −1.93835e-4 vs move 42 (9.7× the −2e-5 bar). Why the family re-opened after pass 5 declared convergence on move 37's field: a token-repair family exhausts per OBJECT — of 489 accepted positions, 375 are new on the 365 pairs whose token planes were re-rendered by moves 38–42, 114 are pass-5 carryover, and exactly zero are new on the 235 unchanged pairs (stale-pair control re-found 67 of 67). Decode wall clock: a t4_direct leg built for move 42 from its own Modal harvest, 978.12 s on Tesla T4 against the 1,260 s ceiling.

<!-- # FORMALIZATION_PENDING: packet input; the equations leg is written by tools/pointer_move_packet.py --equations-leg at harvest, on the exact row -->

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.017284855708527502. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.01711995387438173: archive ≤ 154,507.3 B → **-25,958.7 B**.
- **DISTORTION corner** at held bytes 180,466: distortion ≤ -0.0001649 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-247.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/fire/MODAL_REMOTE_RESULT.json` (sha `d2aadf9809ed6ef9ef435807082fb4849631f224118aba23dd60c2c61443a12e`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/candidate/candidate_runtime/archive.zip` (sha `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`, 180,466 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/SEAL_ddm_sj1_token_predistortion_pass6_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass6_20260910/custody_pointer43/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass6_20260910/custody_pointer43/SEAL_ddm_sj1_token_predistortion_pass6_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: any CPU-axis score (the receiver declares linux-nvidia-t4; the CPU path refuses by design, receipt fc-01M25W34DGHHJES03531X3VDXV). Not claimed: frame-0 repair — measured, priced, fixed-point-verified (6 adopted, +5 B, would add −2.11549e-4 × 0.075 of one bar) but NOT shipped: this arm's close has no selector splice (a build blocker, not an economic one), recorded in FRAME0_VERDICT.json. Not claimed: the first control encode's number (jg2's route emits the HPAC stream, 120,200 B, not the shipped tc1-mixed 119,613 B stream; the instrument flagged delta_trustworthy=false and the number was discarded). Not claimed: any transfer of the 5.25 bits/token price to another field; the price is per object. The 7.7e-6 spread between the local pose convention and the T4 print is the pose-print class the last three packets measured; the exact row decides.

## Next from here

Next: (1) the counted-rider cure (rlc4: 180,178 B candidate on move 42's field, raw-identical, −60 B) re-based onto move 43's bytes once pr14 amends the normalized receiver digest to exclude/normalize MANIFEST.sha256 — the two objects are independent and compose by object change; (2) frame-0 selector splice in this arm's close (build item; +0.075 bar measured); (3) pass 7 only on a successor field that re-renders pairs (the family exhausts per object; on an unchanged field the yield is the pass-5 floor); (4) PR #140 swap re-staged on the shipping bytes on the operator's one-line confirm.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489: 100·0.00010345 + sqrt(4.59e-5) + 25·180,466/37,545,489 = 0.1372848557085275; Δ vs move 42 = −1.9165e-4 (seg −2.92e-4, rate +1.518e-4, pose −7.7e-6 on the T4 print); the seg-debt pool re-opens per OBJECT (375/489 admitted positions new on re-rendered pairs, 0 on unchanged)

Own-vehicle frontier: **S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600]**, archive sha `7beb6a5f…71c1e`.
