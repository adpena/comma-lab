# A1 R(D) derivation (entry packet — D5 expansion #3)

Per N council Decision 5 (Ballé position): "the R(D) derivation should be in
the packet. The packet expansion is publishable-grade research."

This memo derives the Shannon rate-distortion accounting for A1's score
contribution, specific to A1's archive bytes and the contest scorer functional.

## Contest score functional

Per the handoff `pact_score_lowering_handoff_2026-05-11.md` and verified
against `upstream/evaluate.py`:

```
S(B, d_seg, d_pose) = 25 * (B / N) + 100 * d_seg + sqrt(10 * d_pose)
```

where:
- `B` = archive byte count
- `N` = ground-truth raw output byte count (`37,545,489` bytes for the public test set 600 pairs × 2 frames × 874 × 1164 × 3 bytes)
- `d_seg` = average SegNet argmax-disagreement rate
- `d_pose` = average MSE on first 6 pose dimensions

## A1's R(D) point (both axes, EXACT same archive bytes)

### CPU axis (ranking score)

| Quantity | Value |
|---|---:|
| `B` (archive bytes) | 178,262 |
| `B / N` (rate, unscaled) | `4.74789e-3` |
| `d_seg` (SegNet distortion) | `5.6023e-4` |
| `d_pose` (PoseNet distortion) | `3.286e-5` |
| `25 * B / N` (rate contribution) | `0.11869725` |
| `100 * d_seg` (seg contribution) | `0.05602300` |
| `sqrt(10 * d_pose)` (pose contribution) | `0.01812744` |
| **Total `S`** | **`0.19284769`** (matches canonical 0.19284758 within rounding noise) |

### CUDA axis (paired-axis diagnostic)

| Quantity | Value |
|---|---:|
| `B` | 178,262 (identical bytes) |
| `B / N` | `4.74789e-3` |
| `d_seg` | `6.6299e-4` |
| `d_pose` | `1.7103e-4` |
| `25 * B / N` | `0.11869725` |
| `100 * d_seg` | `0.06629900` |
| `sqrt(10 * d_pose)` | `0.04135577` |
| **Total `S`** | **`0.22635202`** |

## Marginal derivatives at A1's operating point (CPU)

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":

```
dS/dB     = 25 / N         = 25 / 37,545,489      = 6.658e-7 per byte
dS/dd_seg = 100
dS/dd_pose= 5 / sqrt(10 * d_pose) = 5 / sqrt(3.286e-4) = 275.87
```

At A1's CPU operating point (`d_pose = 3.286e-5`), the **pose marginal is
275.87 / 100 = 2.76× more valuable per unit than seg**. This matches the
prior council's PR106-r2-frontier finding (2.79× at `pose_avg = 3.4e-5`) —
A1 sits at essentially the same operating point.

The crossover threshold (where pose and seg marginals are equal) is at
`d_pose = 2.5e-4`. A1 CPU's `3.29e-5` is **7.6× below** the crossover; A1
CUDA's `1.71e-4` is **1.46× below** the crossover (still pose-dominant
marginal but closer).

## Per-byte EV at A1's operating point

To break even on a 1-byte addition to A1's archive on the CPU axis, the
addition must reduce either:
- `d_seg` by `dS/dB / dS/dd_seg = 6.658e-9` per byte, OR
- `d_pose` by `dS/dB / dS/dd_pose = 6.658e-7 / 275.87 = 2.414e-9` per byte

Both thresholds are tiny — A1 is byte-stretched per the score functional.
The bias-correction sidecar adds zero archive bytes (it's an inflate-time
constant `1.0` subtraction); its EV is therefore unbounded-per-byte. The
rest of A1's bytes are dominated by the encoded decoder blob (PR101
split-Brotli) and the latent_blob (PR101 ORIGINAL, 15,387 bytes).

## Shannon R(D) frontier position (CPU axis)

A1's CPU score `0.19285` sits at the silver-medal-band cluster on the
empirical contest leaderboard:

| Anchor | CPU score | Rank-tier proxy |
|---|---:|---|
| PR101 (gold) | 0.19284 | gold display tier (rounds to 0.19) |
| **A1** | **0.19285** | **gold display tier (rounds to 0.19)** |
| PR103 (silver) | 0.19487 | silver display tier (rounds to 0.19) |
| PR102 (third) | 0.19538 | bronze display tier (rounds to 0.20) |
| PR #107 (our apogee) | 0.19664 | (rounds to 0.20) |

A1 is on the empirical Pareto frontier on the CPU axis (no candidate
displaces A1 without paying byte-cost). The R(D) interpretation is that
A1 has saturated PR101's substrate-engineering envelope on the contest's
CPU evaluation pipeline; further descent requires substrate-class
boundary movement (different architecture family) OR pose-axis-specific
bolt-ons that don't add bytes (per the bias-correction sidecar pattern).

## Cross-references

- `~/Downloads/pact_score_lowering_handoff_2026-05-11.md` (score functional source)
- `upstream/evaluate.py` (canonical scorer; sha `7da71a84...`)
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE"
- `feedback_grand_council_5_design_decisions_review_20260511.md` Decision 5 R(D) expansion
- Shannon (1948), *A Mathematical Theory of Communication*
- Berger (1971), *Rate Distortion Theory*
