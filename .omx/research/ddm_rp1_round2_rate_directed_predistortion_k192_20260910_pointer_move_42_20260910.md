# FORTY-SECOND POINTER MOVE — S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600]: rp1 round 2 — rate-directed pre-distortion re-based onto move 40's field + carrier re-solve; the pose landed below base (seg exactly 0, rate +5 B) (2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M25TM1G2SNMGXZ39PFWQ4T0V`. Lane `ddm_rp1_round2_rate_directed_predistortion_k192_20260910`. Modal wall 1034.5 s. Archive sha `f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f`, 180,238 B. Runtime tree `a0f7cede2324c0e4f3e21cfc0e59ba115a1e1cfaf479e23a4ad246cbfb76d8f9`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,238/37,545,489 | 0.12001308599283392 |
| seg 100·0.00010637 | 0.010637 |
| pose √(10·4.66e-06) | 0.006826419266350405 |
| **S** | **0.1374765052591843** |

| | ddm_sj1_t4_compose39_rp1_union_20260910 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13763861019288715 | 0.1374765052591843 | **-0.00016210493370283352** |
| d_seg | 0.00010636 | 0.00010637 | 9.999999999994822e-09 |
| d_pose | 4.89e-06 | 4.66e-06 | -2.299999999999995e-07 |
| bytes | 180,233 | 180,238 | +5 |

## Projection fidelity

Projected 0.13746779326523825; realized − projected = 8.711993946058927e-06. seal projection composed the T4's own d_seg and exact bytes with a locally measured RESOLVED pose (pose-print class +1.8e-6..+7.7e-6 optimistic, stated before the row); exact row +8.7e-6 above projection, pose 4.66e-06 vs 4.649e-06 projected

## The mechanism

Round 2 of rp1's rate-directed token pre-distortion, re-based onto move 40's field and closed by a
carrier re-solve. The n600 K=192 sizing shards (4,503 argmax-neutral tokens, 15,552 first-order bits)
had been verified against sj1's pass-4 body — move 37's field — because the sizing script took its
ranking from a flag and its base field from a source constant. rp1 caught this before any heavy step
and re-verified the accepted set on move 40's field with the same cumulative bisection
(`experiments/ddm_rp1_rebase.py`, daadbf7a2): 4,142 of 4,503 tokens survived; the 307 control pairs
whose base plane was unchanged transferred 2,401 of 2,401 tokens (exactly 1.0), so the instrument did
not move and the 361 lost tokens belong to the base alone. Twin encodes of the full rebased field
showed the first-order promise (−1,756.6 B) inverting under the real coder (+620 B); the pass survived
only as a per-pair selection (146 saving pairs), priced by real encode, not by the ledger sum. The
carrier was re-solved on the resulting field and the pose read on the RESOLVED pose: the admission's
net is pose −1.74e-4 (4.649e-6 vs base 4.887e-6), seg exactly 0 over 117,964,800 cells (12,540 local
flips = 12,540 predicted), rate +5 B (+3.3e-6). Frame 0 was measured inside the admission (17
adoptions, all outside the kept set, −1.95e-6 = 0.1× bar) and NOT adopted. The receiver is move 40's,
byte-identical except archive.zip and the two pins; the decode-wall-clock leg is inherited from move
40's `t4_direct` measurement (990.054 s on Tesla T4).

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.017476505259184316. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.017463419266350404: archive ≤ 153,991.4 B → **-26,246.6 B**.
- **DISTORTION corner** at held bytes 180,238: distortion ≤ -1.3086e-05 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-19.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_rp1_round2_t4_20260910/MODAL_REMOTE_RESULT.json` (sha `e312b67e494b58cb1981ceab2bfcc105bfa5729feab78b09c08a9b53148df41e`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_rp1_round2/candidate/candidate_runtime/archive.zip` (sha `f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f`, 180,238 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_rp1_round2/SEAL_ddm_rp1_round2_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_rp1_round2_rate_directed_predistortion_k192_20260910/custody_pointer42/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_rp1_round2_rate_directed_predistortion_k192_20260910/custody_pointer42/SEAL_ddm_rp1_round2_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: that the rate-directed pass paid on rate — it did not (+5 B net after the twin encode;
the first-order K-curve promised −1,756.6 B and the coder charged +620 B on the full rebased field).
The row's gain is the carrier re-solve landing pose below base; the token edits are the field the
carrier was re-solved on, not the byte saver. Not claimed: frame-0 adoption (measured, priced at 0.1×
the bar, not adopted). Not claimed: any transfer of the 361 tokens the rebase lost, or any yield from
proposals the pass-4 acceptance rejected (the rebase is conservative; its yield is a floor). Not
claimed: a CPU-axis score — this row is contest-CUDA T4 only; the contest-CPU adjudication of the
shipping receiver has never been bought and remains a packet blocker. Not claimed: the projection's
pose leg beyond the pose-print class (+1.8e-6 … +7.7e-6 optimistic, stated before the row). Not
claimed: rlc1's rule-118 cure (a separate receiver on move 40's tree, to be re-based onto this
pointer) or any receiver-code door; the receiver here is move 40's, unchanged.

## Next from here

Next from here. (1) rlc1's rule-118 cure (−60 B, counted 19 B geometry, receiver digest b06e59a6…)
sits on move 40's tree; re-base it onto this pointer's archive and take its timing authority from its
own cold T4 fire (`t4_direct`, pr11's preferred route) — five local calibration runs of the move 40
receiver agree at 783–797 s but every one was refused under the frozen admission rule by host daemons
or by our own tool activity, so local calibration is suspended for receivers that have not run on T4.
(2) The seg-debt pool is unchanged by this move (seg exactly 0): sj1's pass-6 Lagrange subset on the
rebased field is the next seg lever, composed by re-verification (pair overlap sub-additivity law).
(3) The rate corner: the rate-directed pass does not pay under the real coder above K≈32 per pair;
the remaining rate lever is representation (container/tail), not token flips. (4) Packet readiness for
PR #140: contest-CPU evaluation of the shipping receiver, the two `except ImportError` research
fallbacks in `runtime/rc3_shared_mixer.py` and `sm1_semantic_mixer.py` (a receiver change → fresh
T4 + CPU rows), the runtime-tree digest vs the auth-eval manifest, and the manifests; the swap itself
waits for the operator's one-line confirm. (5) Custody: vr8 audits every certified MOVE for ExFAT
`._`-stub destinations (rp1 found five 1.83 GB sj1 overlays recorded MOVED with only stubs; re-render
reproduced sha 37ea3842…). Sub-0.12 arithmetic must be re-derived at this move by the packet.

Equations leg (`tac.canonical_equations`): _(equations leg owed by the arm)_

Own-vehicle frontier: **S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600]**, archive sha `f111ab42…8756f`.
