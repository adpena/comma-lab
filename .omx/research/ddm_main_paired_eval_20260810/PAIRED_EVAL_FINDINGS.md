# Paired local exact eval — PR130 base vs cp2 rank-1 composed

**Axis:** `[macOS-CPU advisory; upstream AV GT; immutable evaluate.py; n600]` · `score_claim=false`
**Not** the contest-CUDA/DALI axis that carries the 0.172141297 bar. The DELTA is the finding.

## The headline: the composed −8,688 B was never a win

| | base (PR130 CPR1) | cp2 rank-1 composed |
|---|---:|---:|
| archive bytes | 191,052 | 182,364 (−8,688) |
| d_seg | 0.00042735 | 0.01473595 (**34.5×** worse) |
| d_pose | 0.00015911 | 3.47795892 (**21,859×** worse) |
| rate term | 0.12721375 | 0.12142875 |
| **S** | **0.2098374** | **7.4924** |

The −8,688 B buys 0.005785 S of rate and costs ~7.28 S of distortion. **Not bankable.**

## Harness validated by prediction, not by assertion

Before running, the base row was PREDICTED from ai1's independently-measured
AV-GT row: ai1's realized raw was byte-identical to the base raw, so its
distortion IS the base distortion — 0.208229 − rate(188,636 B) = 0.082635 —
giving base = 0.082635 + rate(191,052 B) = **0.209838**.
Measured: **0.2098374**. Match to 6 significant figures.

That cross-validates three things at once: hp3's retained raw is the base
render, ai1's axis label was correct, and this harness is wired correctly.
The catastrophic candidate number is therefore the CANDIDATE, not the instrument.

## Attribution: exact, and $0 (no decode, no scorer)

Direct comparison of the two retained raws, by frame parity:

| frame parity | mean abs delta | max abs delta | pixels changed |
|---|---:|---:|---:|
| EVEN (pose carrier) | 0.0000 | 0 | 0.00% |
| ODD (semantic) | 67.71 | 240 | 99.26% |

Even frames are BYTE-IDENTICAL. So `ai1`'s ANS + temporal-reversion token leg
is confirmed losslessAGAIN, independently. 100% of the damage is the semantic
render path = `sm3`'s pointwise low-rank r32.

## Characterization: defect, not graceful degradation

Odd-frame statistics vs base, sampled at frames 1/3/101/601:

- base mean ~130.4, sd ~68.5
- cand mean ~77.6, sd ~49.5, 239-245 unique values, correlation **+0.30 to +0.32**

Structure survives (not a collapse), but a CONSISTENT DC shift (-52) and
variance compression (0.73x) on every frame, with correlation ~0.31, is the
signature of a factorization/dequant defect -- e.g. low-rank applied without
centering, so rank-32 spends itself representing the mean instead of the
structure. A genuinely over-aggressive rank-32 approximation degrades
gracefully; it does not shift the DC of all 600 frames by the same amount.

**VERDICT SCOPE: INSTANCE** (this pointwise-low-rank-r32 implementation).
NOT a family verdict on low-rank semantic coding. The named next measurement is
whether the SM3R packer/receiver centers before factorizing.

## Corrected rate ledger on the PR130 base

| candidate | section | delta bytes | distortion status |
|---|---|---:|---|
| ai1 ANS + temporal_reversion | tokens | -2,416 | LOSSLESS (raw byte-identical, twice confirmed) |
| hp3 requant frame_embed step2 | hpac | -8 | LOSSLESS (raw byte-identical) |
| sm3 pointwise low-rank r32 | semantic | -6,272 | **REFUTED** (S 0.2098 -> 7.4924) |
| sm3 vector/scale VQ32 | semantic | -4,648 | UNMEASURED (same receiver, same suspicion) |
| SD1 mixed q3/q4 | semantic | -848 | UNMEASURED |

**Bankable today: -2,424 B** (ai1 + hp3), = 7.3% of the 33,252 B sub-0.15 rate
target -- NOT the 26.1% the composed figure implied. Every downstream budget
that counted sm3's -6,272 B must be re-derived.

## What this vindicates

`cp2` did NOT overclaim: its receipt recorded `d_seg_status: UNMEASURED` and
`d_pose_status: UNMEASURED` explicitly and honestly. The exposure was MAIN
carrying "-8,688 B = 26.1% of the rate target" forward as if byte-closure implied
score-safety. Byte-closure and parse-back prove the BYTES round-trip. They say
nothing about whether the reconstructed model still works. Those are different
claims and this is what the difference costs.

## Custody

- harness + arms: `/Volumes/VertigoDataTier/pact/ddm_main_paired_eval_20260810/`
- base raw sha256 `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353` (= established base raw hash)
- cand raw sha256 `46ca24e7004c5a3ea42a118981a4fdf6a523e9d5b56cf6baff4444a062176f32` (matches cp2's own receipt)
- both raws retained; both reports retained; no payload discarded.
