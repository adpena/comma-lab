# FORTIETH POINTER MOVE — S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600]: composition of moves 38 and 39 by re-verification (sj1's 42-pair seg subset re-based onto rp1's rate-directed field; 282-pair union re-solved; seg 12,540 = 12,540) — S 0.13763861019288715 @ 180,233 B, Δ −3.07e-5 (1.54× bar) (2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M24MA90ER7YTGRTCX9SNQNM4`. Lane `ddm_sj1_t4_compose39_rp1_union_20260910`. Modal wall 1052.2 s. Archive sha `986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`, 180,233 B. Runtime tree `27fc92e8ee156d01ffe65ceb2e141ac0c98a0ccd7b9096d12aa18919fcb28a39`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,233/37,545,489 | 0.1200097566980683 |
| seg 100·0.00010636 | 0.010636 |
| pose √(10·4.89e-06) | 0.0069928534948188355 |
| **S** | **0.13763861019288715** |

| | ddm_rp1_t4_rate_directed_predistortion_20260910 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13766931482209038 | 0.13763861019288715 | **-3.070462920323758e-05** |
| d_seg | 0.00010698 | 0.00010636 | -6.199999999999907e-07 |
| d_pose | 4.89e-06 | 4.89e-06 | 0.0 |
| bytes | 180,186 | 180,233 | +47 |

## Projection fidelity

Projected 0.13763577081553857; realized − projected = 2.839377348573535e-06. optimistic by +2.84e-6: the 8-dp pose print (4.89e-6 vs 4.887e-6 resolved) — the pose-print class (pc2 +1.84e-6, sj1 pass 5 +7.73e-6, rp1 +2.77e-6)

## The mechanism

Composition of the two sibling children of move 37 by re-verification, never by carrying: sj1's move-38 pass-5 subset (42 pairs, 78 tokens, seg repairs) re-based onto rp1's move-39 field (253 pairs, 473 argmax-neutral token changes). 13 overlapping pairs, ZERO position collisions (measured); an assertion that no untouched cell moved. Cross-arm control identity: rp1's shipped field re-encoded through sj1's loop = rp1's own rider body byte-for-byte (119,568 B, 26e9eba5…) and the rebuilt control archive = move 39's own sha — two arms, two encoders, same bytes. Composed stream 119,618 B (twins identical); carrier re-solved per pair over the 282-pair union from move 39's coefficients (resolved pose 4.887e-6 vs stale 2.22e-5; pose base 4.886129e-6 passed pm2's magnitude gate at −0.08 % — after that gate REFUSED a stale-table pairing of move 37's carrier with move 39's renders at 21.3×, its first live catch); seg re-verified on the 13 shared pairs by recompute: 80 repairs became 74 (all four losing pairs inside the overlap); seg-final on the shipped bytes 12,540 predicted = 12,540 measured. Frame-0 deliberately excluded (§30: a repair lever for edit-broken pairs inside future edit rounds). Exact T4: seg 0.00010636, pose 4.89e-6, 180,233 B → S 0.13763861019288715 (Δ −3.07e-5 vs move 39, 1.54× bar; projection 0.13763577 optimistic by +2.84e-6, pose-print class).

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.01763861019288715. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.017628853494818835: archive ≤ 153,743.0 B → **-26,490.0 B**.
- **DISTORTION corner** at held bytes 180,233: distortion ≤ -9.7567e-06 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-14.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_sj1_t4_compose39_rp1_union_20260910/MODAL_REMOTE_RESULT.json` (sha `cd6d5ef5243e26fa1868100a2bbb34efe0d6446b4a7fb9bd1eab7f309f5d3d00`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/candidate_runtime/archive.zip` (sha `986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`, 180,233 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/SEAL_ddm_sj1_compose39_rp1_union_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_compose39_price/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_compose39_price/SEAL_ddm_sj1_compose39_rp1_union_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: additivity — the naive sum of moves 38 and 39 (0.1376341479 @ 180,234 B) was MISSED by +1.62e-6 in projection and by +4.46e-6 in the exact row; four fifths of the projected miss is six seg repairs lost on the 13 shared pairs (token-additivity ≠ seg-additivity: a repair is a property of the render); rate anti-synergy +6.4 %/token (the yield law's mirror); the container returned 3 B (lottery, two-sided). Not claimed: any frame-0 selector effect (excluded by design); any CPU-axis row (single-axis waiver, lineage moves 24–40 all contest-CUDA T4); the corner (re-derived by the packet) — this row is 0.17 % of it.

## Next from here

The field under the three-leg admission remains the only class moving the pointer (moves 31–40). Next on this object: a second RATE-directed pass on the move-40 field (rp1 round 2: neutrality is 4.36 % and flat in rank, so the prize is in more realized proposals per pair; frame-0 repair inside its admission on the pairs it breaks; the composition law says predict seg sub-additivity by pair overlap). tc3 (codex) is at seal on the 78 B lane-predictor tail with the better predictor priced. The generator remains the only representation-level door (Addendum 20). PR #140 swap packet gated on the operator's one-line confirm (17 moves behind).

Equations leg (`tac.canonical_equations`): token_predistortion_multipass_yield_v1

Own-vehicle frontier: **S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600]**, archive sha `986d536b…e9857`.
