# FIFTY-FOURTH POINTER MOVE — S 0.13605599532783202 @ 179,266 B [contest-CUDA T4 n600]: price-first pass 2 on move 53's re-rendered field (pd7): the prior's cheap half re-priced after the move, credit checked after, set-priced (4 iterations) and chosen on a closed-archive size ladder — 53 pairs, −20 B, 1 cell repaired, pose 4.10e-6 (2026-09-16)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M2P468WE5DNZYCMDGPDG6Y1J`. Lane `ddm_pd7_price_first_pass2_on_move53_20260916`. Modal wall 1169.2 s. Archive sha `5c6bf403b4cb4554fe24a46bdf5b46d62876854764a10a22d8c90d76a7292ee6`, 179,266 B. Runtime tree `76b6860df018cbb1ed5ea0d22b3d7a04f11f68c5bf5e1edc6de1d62edc09f0a6`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·179,266/37,545,489 | 0.11936587109039917 |
| seg 100·0.00010287 | 0.010287 |
| pose √(10·4.1e-06) | 0.006403124237432848 |
| **S** | **0.13605599532783202** |

| | ddm_pd6_price_first_generator_on_move52_20260916 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.1361014714463198 | 0.13605599532783202 | **-4.5476118487763895e-05** |
| d_seg | 0.00010288 | 0.00010287 | -1.0000000000008375e-08 |
| d_pose | 4.14e-06 | 4.1e-06 | -4.0000000000000464e-08 |
| bytes | 179,286 | 179,266 | -20 |

## Projection fidelity

Projected 0.13605554401927; realized − projected = 4.5130856202169056e-07. pd7 three-leg projection on the shipped bytes' own cold decode; T4 drift +4.5e-7

## The mechanism

Price-first token pre-distortion, pass 2, on move 53's re-rendered field (pd7): the prior's own probability rows were re-captured on move 53's field (they reproduce the coder's per-frame bits), the cheap half of the plane was enumerated on 588 pairs at ≤ 8 bits/token (4,291 proposals; 341 with a NEGATIVE real charge; the rank-0 sheet now costs 3.075 bits/token vs pd6's 2.163 — move 53 spent the 118 cheapest tokens), each proposal was realized (re-render; per-pair carrier re-solve with frame-0 repair; seg on the frozen argmax), admitted on the RESOLVED pose per real bit (264 pairs positive; 72 pay at their own price), SET-priced by four re-encode iterations (residual −59.6 % → −1.35 %) and chosen on a size ladder of closed archives (set 03: 53 pairs / 66 tokens). Shipped: 179,266 B (−20 B), 1 argmax cell repaired (12,127 flips), pose 4.099e-6 on the shipped bytes' own cold decode; measured on T4 n600: S 0.13605599532783202 (Δ −4.548e-5 vs move 53; projected 0.13605554401927, drift +4.5e-7). Receiver, renderer weights, basis, prior and carrier codec unchanged (behaviour digest 9f6e7168…); decode leg = this candidate's own t4_direct measurement.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.01605599532783203. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.01669012423743285: archive ≤ 155,152.8 B → **-24,113.2 B**.
- **DISTORTION corner** at held bytes 179,266: distortion ≤ 0.00063413 → **26.3× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **952.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_pd7_fire/run1/MODAL_REMOTE_RESULT.json` (sha `1b16aac02348493a3e22d9740a4cc0207b491ad1274b1b964c0d36b27433cf77`).
- Archive: `/Volumes/APDataStore/pact/ddm_pd7/candidate/candidate_runtime/archive.zip` (sha `5c6bf403b4cb4554fe24a46bdf5b46d62876854764a10a22d8c90d76a7292ee6`, 179,266 B).
- Seal: `/Volumes/APDataStore/pact/ddm_pd7/SEAL_ddm_pd7_price_first_pass2_contest_cuda.json`.
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_pd7_price_first_pass2_on_move53_20260916/custody_pointer54/archive.zip` (sha verified: True).
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_pd7_price_first_pass2_on_move53_20260916/custody_pointer54/SEAL_ddm_pd7_price_first_pass2_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: (1) a CPU-axis score — the sibling was refused behind the live single-flight CUDA claim (fired after DISPATCHED) and the receiver family refuses by design on the CPU path (CPU_AXIS_ADJUDICATION_move54.json); (2) that the converged set-price ledger picks the best set — the size ladder on closed archives did (set 00 and set 02 were within 0.001 bars of each other; set 03 won by rate); (3) that the renderer is unfaithful — pd6's nine "unexplained" pairs are explained as a stale overlay composition (overlay from set 00, closed set 01; the nine pairs are exactly the token-plane difference between those fields); pd6's formulation-scope renderer law is superseded by a free per-pair plane comparison; (4) a family-level claim: the cheap half re-prices after every move (rank-0 sheet +42 % vs pd6); price-first pass 3 remains open, with diminishing cheapest tokens.

## Next from here

Next: (a) pd8 — price-first pass 3 on move 54's re-rendered field with the 16 unread carrier bytes (offsets 123–138, found by the second fresh reader of the minimal receiver) and the unused header flag bit removed from the archive in the same byte close (−16 B, no decode change; the sealed receiver must be checked for the same dead bytes); (b) the reviewable packet (submissions/mrs7, measured on move 53's bytes) — the operator decides whether to submit it as measured or restage it on move 54 (archive pin + identity + one T4 run); (c) resume psa2's followthrough after its search receipt; (d) storage: keep every arm ≤ 2 GiB retained; APDataStore ~24 GiB.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489 with d_seg 0.00010287, d_pose 4.1e-6, B 179,266 → 0.13605599532783202; net vs move 53 = 25·(179,266−179,286)/37,545,489 + 100·(0.00010287−0.00010288) + (sqrt(4.1e-5)−sqrt(4.14e-5)) = −4.548e-5

Own-vehicle frontier: **S 0.13605599532783202 @ 179,266 B [contest-CUDA T4 n600]**, archive sha `5c6bf403…92ee6`.
