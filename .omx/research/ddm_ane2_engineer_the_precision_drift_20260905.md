# ddm_ane2 — the drift IS engineerable: 18 profile-chosen ops put SegNet under its bar at 90.8% ANE occupancy, an fp32 TAIL hurts where an fp32 HEAD helps, and 92.83% of PoseNet's error is born in its first 18 ops

Arm: `ddm_ane2_engineer_the_precision_drift` (2026-09-05). Tokens: `[no-triality] [p0-ledger-ok]`.
Lane: `lane_ddm_ane2_engineer_precision_drift_20260905`. Craft contract: `docs/operating_manual_craft_handoff.md`.
Parent: ane1 (`.omx/research/ddm_ane1_ane_screening_lane_20260905.md`). Operator directive: *"Drift can be engineered to fix."*
Axis of every row below: **`[macOS-CPU/ANE advisory]`**, frozen scorers, real frames.
`score_claim=false`, `promotable=false`. **Pointer: UNMOVED. This arm bought no exact row.**

## ANSWER FIRST

1. **The charter's ladder had the direction backwards, and the data says so cleanly.** An fp32
   TAIL (the last k ops) does nothing to either scorer through k=32 and makes PoseNet
   monotonically WORSE beyond it — self-MSE **1.663e-03 at k=0 rising to 2.218e-03 at k=192**
   while dim-0's error stays pinned at **0.150–0.154 for every k**. An fp32 HEAD (the first k ops)
   is the cure: **hd8 4.397e-04, hd16 3.739e-04, hd64 1.333e-04, hd128 7.023e-05** — a **23.7×**
   reduction, with **54.3% of ops still on the Neural Engine**.

2. **The per-op profile says why, and it is one number.** With CoreML fp32 as the reference and
   ONE contiguous op group flipped to fp16 at a time, **PoseNet group 0 (ops 0:18) carries 92.83%
   of the summed drift** — 3.195e-03 against 4e-06 to 6e-06 for twelve of the other fifteen
   groups, which sit AT the all-fp32 floor. Group 0 alone is **1.92× the drift of the entire
   fp16 model.** A model that predicts the drift is distributed puts group 0 at 1/16 = 6.25%;
   measured 92.83%, **residual 13.85**.

3. **SegNet's drift has the opposite topology, and that is the finding pair.** No SegNet group
   dominates: the largest is **g13 (ops 243:261) at 33.8%** of the summed excess over the fp32
   floor, with nine other groups between 3% and 10%. That is exactly why SegNet's tail ladder has
   a single knee at **k=64** (9.710e-04 → 5.726e-04, **1.70×**) and nothing before it: k=64 is the
   first rung whose fp32 tail contains ops 243:261, and k=32 is the last one that misses them.
   Two independent measurements, one mechanism.

4. **A mixed-precision graph DOES reach the ANE — ane1's "fp32 never reaches the ANE" is true only
   of a fully fp32 graph.** `MLComputePlan` places **96.5% of PoseNet's ops on the Neural Engine
   at k=8** and 93.1% at hd16. CoreML partitions; it does not fall back wholesale. This reopens
   the whole selective-precision family that a wholesale reading would have closed.

5. **The input changes the drift by more than any split does.** Through the SAME toolchain, SegNet
   fp16 flips **9.710e-04 of pixels on GT frames** and **4.404e-05 on a real generated decode** —
   **22.0×** from the input alone. That closes ane1's open 513× gap to the 2026-07-13 lane as
   **22.0× inputs × 25.5× toolchain = 561×**, against 513× observed (9% agreement). The
   operationally relevant input is the codec's output, not GT: the scorer reads generated frames,
   and any drift bar quoted on GT is quoting the wrong distribution by a factor of twenty.

6. **SegNet PASSES its bar, at n600, on the operational input — with 18 ops.** The profile-guided
   set `g13` (ops 243:261, **6.1% of the network**) holds **90.8% of ops on the Neural Engine**,
   costs **~4.9 ms** against the all-fp16 model's 3.26 ms, and measures **3.2188e-05 ≤ 3.3e-05**
   at n600. The blind tail split reaches the same place with **3.6× more fp32 ops and 4× the
   latency**. `g13stem` (56 ops, 75.5% ANE) clears the bar by 12.6% and is the route I would ship.
   The charter's falsifier therefore **FIRES on the pose axis and does NOT fire on the SegNet
   axis**.

7. **The pose axis stays closed, by the fp16 drift itself.** The best split measured is
   **7.023e-05 = 9.04× the exact d_pose**, and ~850× above the 1%-of-d_pose a backend needs to be
   readable. Correctly converted, CoreML fp32 pose is **2.448e-12 = 3.1e-07 of d_pose**, so the
   obstruction is the half-precision arithmetic and not a floor.

8. **⚠ I found a defect in my own instrument, and attributed it completely.** Passing
   `FP16ComputePrecision(op_selector=...)` — which every mixed rung needs — makes coremltools
   declare the model's OUTPUT fp16 even when the selector transforms zero ops. One cast. It cost my
   all-fp32 endpoints a factor of **1.57 million** (pose) and turned a bit-exact SegNet into 3.4e-06
   of flips. Every SHAPE finding survives, because every rung carries the same term; the SegNet
   pass rates are **upper bounds**, so the pass is conservative. Full attribution in section 7.

9. **What is worth taking:** the fp32-HEAD rule; **profile before ladder** (18 chosen ops beat 64
   ordinal ones); `coreml_cpu_fp32` as a pose-sweep screen (**38/39 argmin, Kendall tau 1.00**,
   reproducing pr1's gain to the digit); and the three controls ane1 owed or never ran — the input
   control (**22.0×**), the compute-unit control (no effect), and the conversion-path control (which
   caught the defect above).

## PRIOR-LAW PREDICTION vs OUTCOME (the owed line, counted plainly)

The charter carried one prediction chain and one falsifier; I pre-registered a second, sharper
model before the ladder ran (`/Volumes/VertigoDataTier/pact/ddm_ane2_precision/prereg_derived_prediction.md`,
written after the op enumeration and before any rung). Both are counted.

### The charter's chain

| charter claim | outcome |
|---|---|
| a head-only fp32 split cannot cure PoseNet — feature error 1e-3 x \|W·f\| ≈ 0.03 ≫ 2.8e-03 | **CORRECT.** k=8 (the output head) measures **1.7764e-03**, against **1.6630e-03** at k=0. Not a cure; marginally worse. |
| the fp32 split must move EARLY enough | **CORRECT, and now localized.** The per-op profile puts **92.83%** of the drift in ops **0:18**. |
| the ladder reaches the d_pose bar at a split leaving ≥50% of FLOPs in fp16, for ≥3x end-to-end — OR it does not, and the profile says where the error is born | **the OR branch fired.** No tail rung improves on k=0 at all; the profile named ops 0:18. |
| curable by per-op fp32 for THAT op alone | **NO on pose, YES on SegNet.** On pose the best fp32-HEAD set (hd128) reaches **7.0226e-05 = 9.04x d_pose** — 23.7x better, still ~850x above readable. On SegNet the profile-guided **`g13`, 18 ops**, reaches **3.2188e-05 <= 3.3e-05** at n600 on the operational input. The charter's instinct was right; it named the wrong 18 ops. |
| SegNet: per-op fp32 on the final decoder + logits (a few % of FLOPs) reaches ≤3.3e-05 | **FALSIFIED, and from both directions.** k=12 (final decoder + logits, 4.0% of ops) measures **9.7364e-04** against 9.7097e-04 at k=0 — no change. The profile agrees: group 15 (ops 279:297, the final decoder) carries **0.1%** of the summed excess. Making the decoder fp16 costs almost nothing, so making it fp32 gains almost nothing. |
| ...i.e. bit-exact-enough argmax at ≥10x trunk speed | **FALSIFIED** on both halves for that split. |
| **FALSIFIER**: no split with ≥30% fp16 FLOPs reaches either bar → drift is not engineerable by precision placement | **SPLIT VERDICT, and the split is the result.** On the **pose** axis it **FIRES** — nothing reaches the bar, at any k, on either ladder, on either direction. On the **SegNet** axis, on the **operationally relevant input**, it **does NOT fire**: `g13` leaves **93.9% of ops fp16** (90.8% proved on the Neural Engine) and measures **3.2188e-05 ≤ 3.3e-05 at n600**. |

### My own pre-registration

| P | claim | outcome |
|---|---|---|
| P1 | drift falls as `sqrt(N_fp16)`; PoseNet MSE at k=192 should be `1.663e-03 x 94/286 = 5.466e-04` | **FALSIFIED by 4.06x.** Measured **2.2180e-03** — it went UP. The 1.0% fit of the model to the single all-fp16 point was a coincidence of one point fitting one parameter, which is what one point always does. |
| P2 | no PoseNet split with any fp16 op reaches the pose bar | **CONFIRMED.** The best split (hd128) measures **7.0226e-05 = 9.04x d_pose**, ~850x above readable. My reasoning was right and my arithmetic was loose: I derived "zero fp16 ops" from a random-walk error model that section 12 then falsified. Right answer, wrong route. |
| P3 | SegNet reaches its bar at k≈158 (47% fp16) | **FALSIFIED on GT frames** (k=128 measures 13.46x the bar). Reached on the **generated** decode at **k=64**, i.e. at 78.5% fp16 — the opposite side of my predicted split. |
| P4 | that split costs ~47.8 ms (6.6x) | **wrong in the safe direction.** k=128 on GT measures **31.59 ms**; k=64 on the generated decode measures **19.81 ms** = **15.8x** against ane1's 313.74 ms cpu-torch dense pass. |

## 1. A MIXED graph reaches the ANE — the structural precondition, MEASURED

ane1 proved that an all-fp32 graph places **0.0%** of its ops on the Neural Engine, and recorded
that as "fp32 can NEVER reach the ANE". That is true of a *wholly* fp32 graph and false of a
mixed one. `MLComputePlan` on the mixed rungs:

| PoseNet rung | k=0 | k=1 | k=8 | k=16 | k=32 | k=64 | k=128 | k=192 | k=286 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ANE op fraction | 100.0% | 99.0% | 96.5% | 93.4% | 87.9% | 76.9% | 54.6% | 32.8% | 0.0% |

CoreML PARTITIONS. It does not fall back wholesale. Every rung below is therefore a real
fp16-on-ANE / fp32-elsewhere hybrid, not a CPU model wearing an ANE label — which is exactly the
claim ane1's per-op census was built to make checkable, now used the other way round.

The conversion is `coremltools.transform.FP16ComputePrecision(op_selector=...)` — the documented
selector API, read from the installed package, never guessed. Each conversion re-derives the
compute-op sequence from the selector's own observations and refuses if it drifted from the
enumerated one (`assert_op_sequence_stable`), so an ordinal split can never be silently
relabelled.

## 2. THE TAIL LADDER — flat, then actively harmful on the pose axis

| k | fp16 ops | ANE % | NE ms | CPU ms | self-MSE | x d_pose | dim0 mean abs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 286 (100.0%) | 100.0% | 0.85 | 5.89 | 1.6630e-03 | 214.0 | 0.1511 |
| 1 | 285 (99.7%) | 99.0% | 0.83 | 5.79 | 1.7380e-03 | 223.7 | 0.1519 |
| 2 | 284 (99.3%) | 98.6% | 0.84 | 5.96 | 1.7380e-03 | 223.7 | 0.1519 |
| 4 | 282 (98.6%) | 97.6% | 1.04 | 6.20 | 1.8503e-03 | 238.1 | 0.1514 |
| 8 | 278 (97.2%) | 96.5% | 1.11 | 8.12 | 1.7764e-03 | 228.6 | 0.1511 |
| 16 | 270 (94.4%) | 93.4% | 1.47 | 6.50 | 1.7235e-03 | 221.8 | 0.1500 |
| 32 | 254 (88.8%) | 87.9% | 1.49 | 5.90 | 1.7502e-03 | 225.2 | 0.1500 |
| 64 | 222 (77.6%) | 76.9% | 2.39 | 5.47 | 1.9117e-03 | 246.0 | 0.1502 |
| 128 | 158 (55.2%) | 54.6% | 3.35 | 6.45 | 2.1721e-03 | 279.5 | 0.1531 |
| 192 | 94 (32.9%) | 32.8% | 5.53 | 6.53 | 2.2180e-03 | 285.5 | 0.1537 |
| 286 | 0 (0.0%) | 0.0% | 7.27 | 8.40 | 3.8352e-06 | 0.5 | 0.0053 |

Two things this table says that a headline cannot.

**dim-0's error is pinned.** The mean absolute error of output dimension 0 — the dimension ane1
measured as carrying 99.96% of the pose damage — is **0.1500 to 0.1537 for every k from 0 to 192**,
then collapses to 0.0053 at k=286. Making 67% of the network fp32 does not move it. That is not a
gradient with a small slope; it is a quantity that the tail simply does not control.

**The tail split makes it worse, monotonically past k=32.** self-MSE rises 1.6630e-03 → 2.2180e-03,
a 33% increase, while ANE occupancy falls from 100% to 32.8% and latency rises 0.85 → 5.53 ms. The
rung is worse on every axis at once.

I had a mechanism for that and **the cast census killed it.** A topological prefix cut through a
residual network ought to sever many skip connections and insert many fp16↔fp32 casts. Measured,
in the saved MIL programs: **1 cast at k=0, 3 at k=1, 4 at k=16/32/64/192, 5 at k=128** (SegNet
peaks at 8). Three to eight casts cannot produce a 33% MSE change. The boundary-cast explanation is
FALSIFIED by the instrument I built to confirm it, and I am recording it as an open mechanism
rather than reaching for the next story. The remaining named candidate — that a mixed graph
shatters the ANE portion into more segments, each crossing re-materializing activations — is
measured by the `segments` subcommand in stage 7.

| k | fp16 ops | ANE % | NE ms | CPU ms | flips | flip rate | x bar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 297 (100.0%) | 100.0% | 3.64 | 29.75 | 22908 | 9.7097e-04 | 29.42 |
| 1 | 296 (99.7%) | 99.0% | 6.31 | 31.82 | 22862 | 9.6902e-04 | 29.36 |
| 2 | 295 (99.3%) | 98.7% | 6.94 | 27.40 | 22862 | 9.6902e-04 | 29.36 |
| 4 | 293 (98.7%) | 98.0% | 6.85 | 30.06 | 22955 | 9.7296e-04 | 29.48 |
| 8 | 289 (97.3%) | 96.7% | 11.22 | 29.27 | 22996 | 9.7470e-04 | 29.54 |
| 12 | 285 (96.0%) | 95.0% | 15.56 | 36.24 | 22971 | 9.7364e-04 | 29.50 |
| 16 | 281 (94.6%) | 93.7% | 23.56 | 32.51 | 23015 | 9.7550e-04 | 29.56 |
| 32 | 265 (89.2%) | 87.2% | 30.87 | 44.00 | 22996 | 9.7470e-04 | 29.54 |
| 64 | 233 (78.5%) | 76.7% | 31.07 | 42.25 | 13510 | 5.7263e-04 | 17.35 |
| 128 | 169 (56.9%) | 55.7% | 31.59 | 43.50 | 10477 | 4.4407e-04 | 13.46 |
| 297 | 0 (0.0%) | 0.0% | 52.33 | 55.92 | 548 | 2.3227e-05 | 0.70 |

SegNet's tail ladder is flat through k=32 and then steps at **k=64** (9.7097e-04 → 5.7263e-04,
**1.70x**). Section 3 says why that knee is exactly there.

## 3. THE PER-OP PROFILE — the measurement that decides, and the two architectures disagree

Flip ONE contiguous op group to fp16 at a time against the all-fp32 model. This is the only
measurement here that tests *where the error is born* rather than *whether moving a boundary
helps*.

| group | ordinals | ops | ANE % | self-MSE | x d_pose | dim0 mean abs |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 0:18 | 18 | 6.5% | 3.1954e-03 | 411.2 | 0.15120 |
| 1 | 18:36 | 18 | 5.8% | 6.2060e-06 | 0.8 | 0.00759 |
| 2 | 36:54 | 18 | 6.2% | 9.5755e-05 | 12.3 | 0.03379 |
| 3 | 54:72 | 18 | 0.0% | 1.1938e-05 | 1.5 | 0.00918 |
| 4 | 72:90 | 18 | 0.0% | 4.8847e-06 | 0.6 | 0.00628 |
| 5 | 90:108 | 18 | 0.0% | 5.6889e-06 | 0.7 | 0.00646 |
| 6 | 108:126 | 18 | 0.0% | 4.5005e-06 | 0.6 | 0.00604 |
| 7 | 126:144 | 18 | 0.0% | 3.8369e-06 | 0.5 | 0.00550 |
| 8 | 144:162 | 18 | 0.0% | 4.1055e-06 | 0.5 | 0.00576 |
| 9 | 162:180 | 18 | 0.0% | 5.7119e-06 | 0.7 | 0.00657 |
| 10 | 180:198 | 18 | 0.0% | 5.8107e-06 | 0.7 | 0.00696 |
| 11 | 198:216 | 18 | 0.0% | 1.2734e-05 | 1.6 | 0.01016 |
| 12 | 216:234 | 18 | 0.0% | 4.8450e-05 | 6.2 | 0.01969 |
| 13 | 234:252 | 18 | 0.0% | 5.5451e-06 | 0.7 | 0.00626 |
| 14 | 252:269 | 17 | 0.0% | 4.0677e-06 | 0.5 | 0.00607 |
| 15 | 269:286 | 17 | 0.0% | 2.7637e-05 | 3.6 | 0.01507 |

**PoseNet: group 0 is the whole story.** Ops 0:18 alone measure self-MSE **3.1954e-03** — **1.92x
the drift of the entire fp16 model** — and their dim-0 mean absolute error is **0.15120**, against
**0.1511** for the full fp16 model. The first 18 ops reproduce the whole error of the network to
three digits. Twelve of the other fifteen groups sit at the all-fp32 floor (3.8e-06 to 6.2e-06)
and are indistinguishable from no fp16 at all. Group 0's share of the summed drift is **92.83%**;
a model that says the drift is distributed predicts 6.25%; **residual 13.85**.

What is in ops 0:18: `sub` and `mul` (the `net._mean` / `net._std` normalisation), then the first
conv/add/gelu block. The model's input is UNNORMALISED YUV6 in 0..255 and the normalisation
happens INSIDE the graph, so in fp16 the stem subtracts a large constant from a large value to get
a small one — textbook cancellation, at the one place in the network where it is unavoidable.
I state the location as MEASURED and the cancellation reading as INFERRED: the profile localizes,
it does not prove the arithmetic mechanism.

| group | ordinals | ops | ANE % | flips | flip rate | x bar |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 0:19 | 19 | 6.6% | 4861 | 2.0604e-04 | 6.244 |
| 1 | 19:38 | 19 | 5.9% | 4823 | 2.0443e-04 | 6.195 |
| 2 | 38:57 | 19 | 4.6% | 2306 | 9.7741e-05 | 2.962 |
| 3 | 57:76 | 19 | 6.3% | 3986 | 1.6895e-04 | 5.120 |
| 4 | 76:95 | 19 | 5.6% | 1939 | 8.2186e-05 | 2.490 |
| 5 | 95:114 | 19 | 0.0% | 776 | 3.2891e-05 | 0.997 |
| 6 | 114:133 | 19 | 6.3% | 2119 | 8.9815e-05 | 2.722 |
| 7 | 133:152 | 19 | 5.0% | 4194 | 1.7776e-04 | 5.387 |
| 8 | 152:171 | 19 | 0.0% | 635 | 2.6915e-05 | 0.816 |
| 9 | 171:189 | 18 | 5.6% | 4082 | 1.7302e-04 | 5.243 |
| 10 | 189:207 | 18 | 0.0% | 786 | 3.3315e-05 | 1.010 |
| 11 | 207:225 | 18 | 5.9% | 4577 | 1.9400e-04 | 5.879 |
| 12 | 225:243 | 18 | 0.0% | 662 | 2.8059e-05 | 0.850 |
| 13 | 243:261 | 18 | 5.9% | 15248 | 6.4629e-04 | 19.585 |
| 14 | 261:279 | 18 | 3.6% | 673 | 2.8525e-05 | 0.864 |
| 15 | 279:297 | 18 | 5.7% | 580 | 2.4584e-05 | 0.745 |

**SegNet is the opposite: distributed, with one hot spot.** The largest single group is **g13
(ops 243:261) at 33.8%** of the summed excess over the fp32 floor; nine further groups sit between
3% and 10%. No small set carries it.

**And that explains the tail ladder's only knee.** k=64 covers ops 233:296 — the first rung whose
fp32 tail CONTAINS ops 243:261. k=32 covers 265:296 and misses them entirely. The two measurements
were made independently and corroborate each other exactly: the ladder's step is g13.

## 4. THE MIRROR — an fp32 HEAD is the cure the profile predicts, and it works

`selective` takes arbitrary ordinal sets, so the mirror of the tail ladder needed no new code path.

| label | fp32 ops | fp16 % | ANE % | NE ms | self-MSE | x d_pose | dim0 mean abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| hd1 | 1 | 99.7% | 99.3% | 1.63 | 1.8717e-03 | 240.9 | 0.15085 |
| hd2 | 2 | 99.3% | 99.0% | 1.11 | 1.9119e-03 | 246.1 | 0.14850 |
| hd4 | 4 | 98.6% | 97.9% | 1.31 | 1.8305e-03 | 235.6 | 0.14977 |
| hd8 | 8 | 97.2% | 96.5% | 2.03 | 4.3967e-04 | 56.6 | 0.06124 |
| hd16 | 16 | 94.4% | 93.1% | 1.99 | 3.7392e-04 | 48.1 | 0.05943 |
| hd32 | 32 | 88.8% | 87.5% | 3.62 | 3.4081e-04 | 43.9 | 0.05586 |
| hd64 | 64 | 77.6% | 76.7% | 7.42 | 1.3328e-04 | 17.2 | 0.03793 |
| hd128 | 128 | 55.2% | 54.3% | 6.80 | 7.0226e-05 | 9.0 | 0.02184 |

Monotone, and the step is where the profile said it would be: nothing at hd1/hd2/hd4, then
**hd8 = 4.3967e-04 (3.78x better than all-fp16)**, hd16 4.4x, hd64 12.5x, **hd128 = 7.0226e-05,
23.7x better, with 54.3% of ops still proved on the Neural Engine**.

So the direction is settled: **on this hardware an fp32 HEAD buys drift and an fp32 TAIL spends
it.** That is the transferable rule, and it is the opposite of the ladder the charter specified.

| label | fp32 ops | fp16 % | ANE % | NE ms | flips | flip rate | x bar |
|---|---:|---:|---:|---:|---:|---:|---:|
| hd1 | 1 | 99.7% | 99.3% | 4.77 | 22870 | 9.6936e-04 | 29.374 |
| hd2 | 2 | 99.3% | 99.0% | 5.24 | 22892 | 9.7029e-04 | 29.403 |
| hd4 | 4 | 98.7% | 98.3% | 6.84 | 22887 | 9.7008e-04 | 29.396 |
| hd8 | 8 | 97.3% | 96.3% | 7.38 | 21722 | 9.2070e-04 | 27.900 |
| hd16 | 16 | 94.6% | 92.3% | 11.95 | 21772 | 9.2282e-04 | 27.964 |
| hd32 | 32 | 89.2% | 88.6% | 15.40 | 21022 | 8.9103e-04 | 27.001 |
| hd64 | 64 | 78.5% | 77.1% | 29.73 | 20274 | 8.5932e-04 | 26.040 |
| hd128 | 128 | 56.9% | 53.0% | 22.16 | 19959 | 8.4597e-04 | 25.636 |

SegNet's head mirror moves the flip rate only **1.15x** (9.7097e-04 → 8.4597e-04 at hd128) against
its tail ladder's **2.19x** — again exactly what a distributed profile with a hot spot in the tail
predicts.

## 5. THE DRIFT IS NOT "fp16" — IT IS "fp16 ON THE ANE" (6.32x of it)

The same fp16 `.mlpackage`, the same 120 pairs, the same reference; only the requested compute
unit differs:

| ComputeUnit requested | ANE ops | drift |
|---|---:|---:|
| `CPU_AND_NE` | 287/287 | self-MSE 1.6630e-03 (214.02x d_pose) |
| `CPU_ONLY` | 0/287 | self-MSE 2.6334e-04 (33.89x d_pose) |

**CPU_ONLY 2.6334e-04 against CPU_AND_NE 1.6630e-03 — the Neural Engine's own arithmetic is 6.32x
worse than CPU fp16 on the identical graph.** SegNet says the same thing independently: the same
fp16 package measures **1.5428e-04 on the CPU** against **9.7097e-04 on the ANE** — **6.29x**. Two
architectures, two reading axes, the same factor to three digits. ane1 attributed the whole 1,448x
to "fp16"; a factor of ~6.3 of it belongs to the ANE specifically, not to half-precision. That matters for anyone
reasoning about fp16 elsewhere in this codebase: an fp16 result measured on a CPU does not predict
the ANE's.

The obvious confound is the compute-unit REQUEST itself. It is ruled out:

| ComputeUnit requested | ANE ops | drift |
|---|---:|---:|
| `ALL` | 0/287 | self-MSE 3.8352e-06 (0.49x d_pose) |
| `CPU_AND_GPU` | 0/287 | self-MSE 3.8352e-06 (0.49x d_pose) |
| `CPU_AND_NE` | 0/287 | self-MSE 3.8352e-06 (0.49x d_pose) |
| `CPU_ONLY` | 0/287 | self-MSE 3.8352e-06 (0.49x d_pose) |

| ComputeUnit requested | ANE ops | drift |
|---|---:|---:|
| `ALL` | 0/298 | flip rate 2.3227e-05 (0.70x bar) |
| `CPU_AND_GPU` | 0/298 | flip rate 2.3227e-05 (0.70x bar) |
| `CPU_AND_NE` | 0/298 | flip rate 2.3227e-05 (0.70x bar) |
| `CPU_ONLY` | 0/298 | flip rate 2.3227e-05 (0.70x bar) |

All four requests — `CPU_ONLY`, `CPU_AND_NE`, `CPU_AND_GPU`, `ALL` — return numbers identical to
seven digits on an all-fp32 graph, on both scorers, with 0 ANE ops in every plan. **Requesting the
ANE does not perturb a graph that cannot use it.** `ComputeUnit.ALL` is numerically safe; it only
changes placement and speed. (ane1 measured that it also costs latency on an fp32 graph — 96.81 ms
against 85.77 ms — and that stands.)

## 6. THE INPUT MOVES THE DRIFT 22x — MORE THAN ANY SPLIT DOES

ane1 owed a control it could not run: its own decode of the pointer body no longer exists on disk,
so it could not separate the INPUT from the TOOLCHAIN in its 513x gap to the 2026-07-13 lane. Run
here on a real generated decode
(`ddm_afr1_tile48_receiver_identity/identity_v1/out/0.raw`, 1200 frames), through the SAME
toolchain as the GT rows above:

| k | fp16 ops | ANE % | NE ms | CPU ms | flips | flip rate | x bar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 297 (100.0%) | 100.0% | 3.20 | 22.05 | 1039 | 4.4039e-05 | 1.33 |
| 32 | 265 (89.2%) | 87.2% | 17.71 | 30.59 | 1032 | 4.3742e-05 | 1.33 |
| 64 | 233 (78.5%) | 76.7% | 19.81 | 31.10 | 739 | 3.1323e-05 | 0.95 |
| 128 | 169 (56.9%) | 55.7% | 25.31 | 38.96 | 671 | 2.8441e-05 | 0.86 |
| 297 | 0 (0.0%) | 0.0% | 45.11 | 46.29 | 81 | 3.4332e-06 | 0.10 |

| SegNet fp16 argmax flip rate | value | vs GT |
|---|---:|---:|
| GT frames, this arm, coremltools 9.0, n120 | 9.7097e-04 | 1.00x |
| **generated decode, this arm, same toolchain, n120** | **4.4039e-05** | **0.045x (22.0x lower)** |
| pointer-body decode, ane1, same toolchain, n600 | 4.818e-05 | — (1.09x of this arm's generated number) |
| GT frames, 2026-07-13 lane, older toolchain, n24 | 2.4746e-02 | 25.5x higher than this arm's GT |

**ane1's 513x gap decomposes as 22.0x from the input times 25.5x from the toolchain = 561x**,
against the 513x observed — agreement within 9%. The question ane1 left open is closed, and the
answer is that both suspects were real and roughly equal in log terms.

The operational reading matters more than the forensic one: **the scorer reads generated frames,
not GT.** Smoother codec output has wider top-2 margins, so it is 22x more tolerant of the same
numeric noise. Any drift bar quoted on GT frames is quoting the wrong distribution by a factor of
twenty.

On the operational input, **all-fp16 already sits at 1.33x the bar**, and **k=64 measures
3.1323e-05 = 0.95x the bar at 19.81 ms** — passing, at 76.7% ANE occupancy and **15.8x** against
ane1's 313.74 ms cpu-torch dense pass. That is thin: 5% of margin at n120 is a coin toss with a
good attitude, which is why section 13 re-measures it at n600.

## 7. THE fp32 FLOOR WAS MINE — one output cast, fully attributed

Every all-fp32 endpoint here carried a residual against 1-thread CPU-torch: pose self-MSE
**3.8352e-06**, SegNet **2.3227e-05** on GT and **3.4332e-06** on the generated decode. ane1
measured its own `coreml_cpu_fp32` at **0 flips** and **2.4348e-12**. Section 5 already ruled out
the ComputeUnit request. The remaining suspects were the INPUT and this arm's CONVERSION PATH, and
`refconvert` builds both endpoints both ways inside one instrument, on one reference:

| model / input | fp32 via `FP16ComputePrecision(selector -> none)` | fp32 via `compute_precision=FLOAT32` | ratio |
|---|---:|---:|---:|
| PoseNet, GT (self-MSE) | 3.835206e-06 | **2.448134e-12** | 1,566,600x |
| SegNet, GT (flip rate) | 2.322727e-05 | **4.238553e-08** | 548x |
| SegNet, generated decode (flip rate) | 3.433228e-06 | **0.000000e+00** | bit-exact |
| PoseNet, GT, **fp16** | 2.633368e-04 | **2.633368e-04** | **1.000000** |
| SegNet, GT, **fp16** | 1.542833e-04 | **1.542833e-04** | **1.000000** |

**The fp16 endpoints are identical to seven digits; the fp32 endpoints are not.** So the input is
exonerated and the conversion CALL is the cause — and the MIL diff says exactly what it is:

| | total ops | const | cast | compute | output dtype |
|---|---:|---:|---:|---:|---|
| fp32, selector | 1058 | 771 | **1** | 286 | **65552 = FLOAT16** |
| fp32, plain | 1056 | 770 | **0** | 286 | 65568 = FLOAT32 |

**One cast, on the model OUTPUT.** Passing an `FP16ComputePrecision` instance makes coremltools
declare the mlprogram's output fp16 *even when the selector transforms zero ops*, so the result is
rounded to half precision on the way out. The arithmetic checks: fp16 spacing at |31.16| is
2^-6 = 0.015625, so a uniform rounding error has RMS 0.0045, and over 6 dims that is a self-MSE of
**3.4e-06** against the measured **3.835e-06** — agreement within 13%. The mechanism is confirmed
by magnitude, not just by structure.

**What this does and does not touch.** The artifact is a CONSTANT additive rounding of the OUTPUT,
identical on every rung of every ladder.

* **Every shape finding stands**, because every rung carries the same term: group 0's 92.83%; the
  head-helps / tail-hurts asymmetry; the 6.3x ANE-vs-CPU factor (same package both sides); the
  22.0x input effect (same conversion both sides); the hybrid's failures (far above the floor).
* **The SegNet finalist pass is CONSERVATIVE, and that is now a mechanism rather than a hope.** A
  correctly built model has no output cast, so it can only have FEWER flips than measured: on the
  generated decode a plain-converted all-fp32 SegNet flips **zero** pixels where the selector-built
  one flips 3.4332e-06. The measured `g13` rate is therefore an UPPER BOUND on the true one, and a
  measured pass is a real pass.
* **One earlier reading of mine is WITHDRAWN.** I wrote that the pose axis is "closed by the FLOOR,
  not by the split" because the all-fp32 endpoint sat at 0.49x d_pose. That floor was my own output
  cast. Correctly built, CoreML fp32 pose is **2.448e-12 = 3.1e-07 of d_pose** — ane1's number. The
  pose axis is closed by the fp16 DRIFT alone: the best split, hd128 at 7.0226e-05, is ~6.6e-05
  after removing the artifact, still **8.5x d_pose** and ~850x above readable. Same verdict,
  correct reason.

## 8. THE PROFILE-GUIDED SET BEATS THE LADDER — one third the fp32 ops for the same drift

The sensitivity profile is not just an explanation; it is a cheaper cure. Holding SegNet's single
hot group `g13` (ops 243:261, **18 ops = 6.1% of the network**) at fp32 and leaving everything else
fp16:

| set | ordinals | fp32 ops | fp16 % | ANE % | NE ms | flip rate | x bar |
|---|---|---:|---:|---:|---:|---:|---:|
| g13 | 243..260 (18 ops) | 18 | 93.9% | 90.8% | 4.70 | 5.8344e-04 | 17.680 |
| g13stem | 0..260 (56 ops) | 56 | 81.1% | 75.5% | 20.01 | 4.7963e-04 | 14.534 |
| g13hot | 0..260 (130 ops) | 130 | 56.2% | 30.2% | 29.98 | 1.7620e-04 | 5.339 |

| set | ordinals | fp32 ops | fp16 % | ANE % | NE ms | flip rate | x bar |
|---|---|---:|---:|---:|---:|---:|---:|
| g13 | 243..260 (18 ops) | 18 | 93.9% | 90.8% | 4.92 | 3.1365e-05 | 0.950 |
| g13stem | 0..260 (56 ops) | 56 | 81.1% | 75.5% | 20.99 | 2.8441e-05 | 0.862 |
| g13hot | 0..260 (130 ops) | 130 | 56.2% | 30.2% | 29.26 | 1.1910e-05 | 0.361 |

| SegNet route | fp32 ops | ANE occupancy | flip rate, generated decode | vs 3.3e-05 bar |
|---|---:|---:|---:|---:|
| all fp16 (`k=0`) | 0 | 100.0% | 4.4039e-05 | 1.33x |
| tail split `k=64` | 64 (21.5%) | 76.7% | 3.1323e-05 | **0.95x** |
| **profile-guided `g13`** | **18 (6.1%)** | **90.8%** | **3.1365e-05** | **0.95x** |

**Eighteen ops chosen by the profile do what sixty-four ops chosen by ordinal do**, and give back
14 points of Neural Engine occupancy. That is the practical payoff of measuring where the error is
born instead of sweeping where the boundary sits.

| set | ordinals | fp32 ops | fp16 % | ANE % | NE ms | self-MSE | x d_pose |
|---|---|---:|---:|---:|---:|---:|---:|
| g0 | 0..17 (18 ops) | 18 | 93.7% | 92.4% | 2.07 | 3.5186e-04 | 45.3 |
| g0g2 | 0..53 (36 ops) | 36 | 87.4% | 84.6% | 3.23 | 1.9381e-04 | 24.9 |
| g0g2g15 | 0..285 (53 ops) | 53 | 81.5% | 78.0% | 3.40 | 2.3084e-04 | 29.7 |
| g0g12g15 | 0..285 (53 ops) | 53 | 81.5% | 66.8% | 2.31 | 3.4083e-04 | 43.9 |

On the pose axis the same trick reaches **1.9381e-04** (`g0g2`, ops 0:18 + 36:54, 84.6% ANE) — an
**8.6x** improvement on all-fp16 and still **25x d_pose**. Adding more groups makes it worse, not
better, which is the tail-split pathology of section 2 reappearing.

## 9. THE REALIZED HYBRID — measured NO-GO twice over, on correctness AND on speed

ane1 priced the exact-argmax hybrid GO on pixel area (0.357% band, 89x headroom under a 3x bar) and
NO-GO on the 2026-07-13 lane's tile realization. This arm BUILT it: fp16 ANE dense pass, band
selected from the **fp16** margin, fp32 recompute on fixed-size 128x128 crops through a real CoreML
model converted for that shape, both denominators timed in the same run.

| band | dense fp16 flips | hybrid flips | crops/frame | crop area / frame | tile occupancy | crop-vs-fullframe disagreement | hybrid ms | vs cpu-torch | vs dense `coreml_cpu_fp32` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4456 (100% flip coverage) | 366 | **9,195** | 21.5 | **1.792** | 0.448 | **9,195 / 28,616 = 32.1%** | 116.49 | 1.71x | **0.39x** |
| 0.05 (91.5% coverage) | 366 | **894** | 13.0 | **1.083** | 0.271 | **863 / 1,679 = 51.4%** | 71.43 | 2.80x | **0.65x** |

**The hybrid makes the argmax WORSE, by 25x at the full band.** A U-Net crop is not a window on the
full-frame result: EfficientNet-B2's 23 squeeze-excitation blocks average over the whole input, so
the crop's global gates see a different image, and **a third to a half of every recomputed band
pixel disagrees with the fp32 reference it was supposed to restore.** The band SELECTION is fine —
0.4456 covers 100% of the fp16 flips, exactly as ane1's margin census predicted. It is the
recompute that fails.

**And it fails the speed bar independently.** The crops together cover **1.79 frames** at the full
band, because 21.5 crops of 128x128 with halos is 352,256 pixels against a 196,608-pixel frame. The
0.357%-of-pixels arithmetic never priced the halo or the tile granularity. Measured tile occupancy
is **0.448 (21.5 of 48 tiles)** — the 2026-07-13 lane measured a median 22.5 of 48. Two lanes, two
independent implementations, the same occupancy.

**Verdict: the hybrid is closed on this architecture, and now on evidence rather than on price.**
It is not slow because the band is big; it is slow because a scattered band lights up half the
tiles, and it is wrong because a U-Net has global context. Both are properties of the architecture,
not of the implementation, so a better kernel does not rescue it. The honest successor is a scorer
whose receptive field is local, and we do not get to choose the scorer.

## 10. THE POSE SCREEN THAT DOES WORK — `coreml_cpu_fp32`, replayed at last

ane1 recommended `coreml_cpu_fp32` and replayed `ane_fp16_screen`. It never replayed the backend it
recommended, so the recommendation was never tested on the thing the sweep actually does: RANK.
Same 39 pairs, same runtime, all 8 modes confirmed on `cpu_torch`:

| quantity | `coreml_cpu_fp32` (this arm) | `ane_fp16_screen` (ane1) | bar |
|---|---:|---:|---|
| argmin agreement vs `cpu_torch` | **38 / 39 = 97.44%** | 4 / 39 = 10.26% | >= 95% |
| Kendall tau-b, median over pairs | **1.0000** | 0.0714 | — |
| total confirmed gain if adopted | **+1.2080e-04** | -4.728e-02 | pr1's CPU sweep: +1.208e-04 |
| seconds per screened forward | 0.0582 | 0.0779 | — |
| seconds per confirmed forward | 0.1131 | 0.1603 | — |
| end-to-end speedup | **1.94x** | 2.06x | — |

**It reproduces pr1's sweep gain to the digit (+1.2080e-04 against +1.208e-04) with perfect median
rank correlation.** The 1.94x is modest because Amdahl still eats it — `render_frame0` +
`preprocess_input` stay in torch — but it is free and it is lossless. This is the one screening
result in the ane1/ane2 pair that clears its bar, and it is on the CPU.

## 11. THE DEFECT WAS IN MY OWN INSTRUMENT, AND ITS CONTROL FOUND IT

Section 7 is the result; this is the note about how it was caught, because that part is the
transferable bit. The control that found it — run ane1's OWN package against THIS arm's reference —
was not in the charter. I added it because two arms reported the same quantity two orders of
magnitude apart and I could not name which of us was wrong. The rule I would carry forward: **when
your number disagrees with a sister arm's on a quantity you both claim to measure, the cheapest
experiment is almost always to run THEIR artifact through YOUR instrument**, because it holds the
input fixed and varies exactly the thing in dispute.

Two further guards earned their keep the same way. `assert_op_sequence_stable` would have caught a
relabelled rung; `_compiled` rebuilding a stale `.mlmodelc` would have caught a placement proof
describing a different graph. Neither fired, which is what a guard doing its job looks like.

## 12. TWO MECHANISM HYPOTHESES, BOTH FALSIFIED BY MY OWN MEASUREMENTS

Section 2's finding — a longer fp32 tail makes PoseNet *worse* — deserved a mechanism. I had two,
and built the instrument for each. Both are dead:

| hypothesis | test | outcome |
|---|---|---|
| a topological prefix cut severs many residual skip connections, and each fp16/fp32 boundary cast rounds | `castcount` over every rung's saved MIL | **FALSIFIED.** 1 cast at k=0, 3 at k=1, 4 at k=16/32/64/192, 5 at k=128; SegNet peaks at 8. Three to eight casts cannot produce a 33% MSE change. |
| the mixed graph shatters the ANE portion into many segments, each crossing re-materializing activations | `segments` — contiguous device runs in program order | **FALSIFIED.** **Every** mixed rung has exactly **1 ANE segment and 1 crossing**, from k=1 through k=192 and across the whole fp32-HEAD mirror. |

So the tail-split worsening is **MEASURED and UNEXPLAINED**. I am recording it that way rather than
reaching for a third story. The next probe I would run is a per-op activation diff between the k=0
and k=64 graphs at the point where their programs diverge — but it does not change any verdict
here, because no tail rung was ever a candidate.

## 13. THE n600 FINALIST — the row the bar was written for

Five percent of margin at n120 is a coin toss with a good attitude. The three rungs that matter,
re-measured at n600 on the same generated decode:

| k | fp16 ops | ANE % | NE ms | CPU ms | flips | flip rate | x bar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 297 (100.0%) | 100.0% | 3.20 | 22.05 | 1039 | 4.4039e-05 | 1.33 |
| 32 | 265 (89.2%) | 87.2% | 17.71 | 30.59 | 1032 | 4.3742e-05 | 1.33 |
| 64 | 233 (78.5%) | 76.7% | 19.81 | 31.10 | 739 | 3.1323e-05 | 0.95 |
| 128 | 169 (56.9%) | 55.7% | 25.31 | 38.96 | 671 | 2.8441e-05 | 0.86 |
| 297 | 0 (0.0%) | 0.0% | 45.11 | 46.29 | 81 | 3.4332e-06 | 0.10 |_N600

| set | ordinals | fp32 ops | fp16 % | ANE % | NE ms | flip rate | x bar |
|---|---|---:|---:|---:|---:|---:|---:|
| g13 | 243..260 (18 ops) | 18 | 93.9% | 90.8% | 4.93 | 3.2188e-05 | 0.975 |
| g13stem | 0..260 (56 ops) | 56 | 81.1% | 75.5% | 19.37 | 2.8856e-05 | 0.874 |

| route | fp32 ops | ANE occupancy | ms (`CPU_AND_NE`) | flip rate n600 | vs 3.3e-05 bar |
|---|---:|---:|---:|---:|---:|
| all fp16 (`k=0`) | 0 | 100.0% | 3.26 | 4.5742e-05 | 1.386x |
| tail split `k=64` | 64 (21.5%) | 76.7% | 20.12 | 3.1264e-05 | **0.947x PASS** |
| **profile-guided `g13`** | **18 (6.1%)** | **90.8%** | **~4.9** | **3.2188e-05** | **0.975x PASS** |
| `g13stem` | 56 (18.9%) | 75.5% | ~21 | 2.8856e-05 | **0.874x PASS** |
| all fp32 (`k=297`) | 297 | 0.0% | 40.07 | 3.0009e-06 | 0.091x |

**The n120 estimates held.** `k=64` moved 3.1323e-05 → 3.1264e-05 (0.2%); the all-fp16 endpoint
moved 4.4039e-05 → 4.5742e-05 (3.9%). The n120 stratified sample was representative of the n600
population it was drawn from, which is what a stride sample is supposed to be and is not always.

**`g13` is the route.** Eighteen fp32 ops — 6.1% of the network, chosen by the sensitivity profile
— hold 90.8% of the graph on the Neural Engine, cost **~4.9 ms against the all-fp16 model's
3.26 ms**, and bring the argmax flip rate from 1.386x the bar to **0.975x**. The tail split reaches
the same place with 3.6x more fp32 ops and **4x the latency**. Against ane1's 313.74 ms 1-thread
cpu-torch dense pass that is roughly **64x**, and against the bit-exact dense `coreml_cpu_fp32`
route **~8x**.

**Two honest qualifications, both stated rather than buried.**

1. **The margin is 2.5%.** `g13` clears its bar by one part in forty at n600. `g13stem` clears it by
   12.6% for 4x the latency, and is the route to pick if the bar must not be re-litigated. I would
   ship `g13stem` and keep `g13` as the aggressive option, not the reverse.
2. **These rates are UPPER BOUNDS** — see section 7. The measured numbers include an fp16 rounding
   of the output logits that a correctly converted model does not have (a plain-converted all-fp32
   SegNet flips **zero** pixels on this decode where the selector-built one flips 3.4332e-06). The
   true `g13` rate is lower than 3.2188e-05, so the pass is conservative. It should still be
   re-measured once the conversion emits an fp32 output, and that is the first owed item below.

## 14. WHAT IS WORTH TAKING, AND WHAT IS NOT

**Take:**

1. **`coreml_cpu_fp32` as a pose-sweep SCREEN.** 38/39 argmin agreement, Kendall tau-b 1.00, and it
   reproduces pr1's sweep gain to the digit. Wired: `tac.ane_screening.BACKEND_AXIS_VERDICTS`
   records it, and `assert_backend_admissible_for_axis` refuses the backends measured unfit rather
   than leaving a future caller to rediscover them.
2. **The fp32-HEAD rule.** On this hardware, holding the FIRST ops at fp32 buys drift and holding
   the LAST ops at fp32 spends it. That is architecture-independent enough to try first anywhere
   the ANE is used, and it is the opposite of the intuition the charter and I both started with.
3. **Profile before ladder.** For SegNet, 18 profile-chosen ops did what 64 ordinal-chosen ops did,
   at 14 points more ANE occupancy. The `sensitivity` subcommand costs one conversion per group.
4. **The two controls.** The input control (22.0x) and the compute-unit control (no effect) are
   cheap, and each removed a suspect that would otherwise have been argued about indefinitely.

**Do not take:**

1. **The crop-recompute hybrid.** Closed on measurement, not on price, and closed twice: the crops
   cover 1.79 frames, and a third to a half of every recomputed pixel disagrees with the reference
   because a U-Net pools globally.
2. **Any fp16 pose reading.** 214x d_pose on the ANE, 33.9x on the CPU, 9.0x at the best split. The
   axis has no slack, which is ane1's finding and this arm only sharpens it.
3. **This arm's absolute fp32-endpoint numbers**, pending section 11's conversion fix.

**Owed, and named rather than left implicit:**

* Re-measure the SegNet generated-decode verdicts through the plain conversion path (section 11).
* The tail-worsening mechanism (section 12) — two hypotheses falsified, none standing.
* A per-op activation diff at the point where the k=0 and k=64 programs diverge, if anyone wants
  that mechanism.

## Equations leg (`tac.canonical_equations`)

Extends ane1's **`scorer_fp16_drift_by_axis_v1`** with three anchors from this arm, each carrying
the prediction it was measured against so the residual counts something:

| anchor | predicted | measured | residual |
|---|---|---|---|
| `ane2_posenet_fp32_tail_split_does_not_reduce_drift_n120_20260905` | random walk over fp16 ops: MSE ∝ N_fp16, so 5.466e-04 at k=192 | **2.2180e-03** (it rose) | **3.06** |
| `ane2_posenet_drift_is_born_in_the_first_18_ops_n120_20260905` | drift distributed: group 0's share = 1/16 = 6.25% | **92.83%** | **13.85** |
| `ane2_posenet_fp32_head_is_the_cure_n120_20260905` | the tail ladder's flatness read as "no split helps": 1.6630e-03 | **7.0226e-05** at hd128 | **0.958** |

Registered by `tools/ddm_ane2_register_equation_anchors.py`. The equation's own form is unchanged —
`A_axis = ε₁₆·‖y‖∞ / slack_axis` still holds; what these anchors add is that ε₁₆ is not a property
of the op count but of WHICH ops, and that on the ANE it is not ε₁₆ at all but ε_ANE ≈ 6.32·ε₁₆.

## Apparatus

- **`src/tac/ane_precision.py`** — the precision-placement algebra, importable from the main
  `.venv` with no `coremltools`: op-sequence identity (`compute_op_names`,
  `assert_op_sequence_stable`), the split/group/selective constructions, the per-axis verdicts, and
  the realized-hybrid geometry (`margin_band_mask`, `dilate_bool`, `occupied_tiles`, `crop_boxes`,
  `crop_boxes_with_cores`, `fixed_crop_boxes`, `hybrid_speedup`).
- **`experiments/ddm_ane2_engineer_precision_drift.py`** — `enumerate` / `reference` /
  `sensitivity` / `ladder` / `selective` / `units` / `segments` / `castcount` / `refconvert` /
  `hybrid`.
- **`src/tac/ane_screening.py`** — extended with `BACKEND_AXIS_VERDICTS`, `backend_axis_verdict`
  and `assert_backend_admissible_for_axis`, so a measured negative refuses rather than waits to be
  rediscovered.
- **Tests**: `src/tac/tests/test_ane_precision.py` (**47**) + 6 added to
  `src/tac/tests/test_ane_screening.py` (**34** total). All run in the main `.venv` without an ANE.
- **Runbook**: `docs/runbook_ane_precision_20260905.md`.

Three guards are worth naming because each caught something:

* `assert_op_sequence_stable` re-derives the compute-op sequence on EVERY conversion and refuses if
  it drifted, so an ordinal split can never be silently relabelled.
* `_compiled` rebuilds a stale `.mlmodelc` rather than reusing it, so a placement proof can never
  describe a different graph than the one measured.
* The hybrid measures its own denominators, because the reference report's per-pair wall clock
  includes GT decode and dividing by it would have inflated every speedup by ~2x. That field is
  now named `mean_ms_per_pair_including_frame_io`; its old name said `median_forward_ms`, which was
  a field name that lied.

## Receipts

All under `/Volumes/VertigoDataTier/pact/ddm_ane2_precision/`, with launch manifests and `run.log`
under `run_*/`. Payloads, not lengths: per-pair flip-rate arrays and per-rung pose vectors are on
disk for every rung, with sha256 in the reports.

| artifact | what |
|---|---|
| `enumerate.json` | 297 SegNet / 286 PoseNet compute ops, ordinal → name → op_type |
| `prereg_derived_prediction.md` | my pre-registration, written before the first rung |
| `reference_n600.json` + `reference/*.npz` | the CPU-torch fp32 authority, GT frames |
| `reference_generated_n120.json`, `reference_generated_n600.json` | the same on a generated decode |
| `screen/ladder_*.json`, `screen/sensitivity_*.json` | the ladders and the profiles |
| `stage2/mirror_*.json`, `stage2/units_*.json`, `stage2/castcount.json` | mirror, controls, casts |
| `stage3/`, `stage6/`, `stage8/` | input control, targeted sets, n600 finalist |
| `stage5/hybrid_*.json` | the realized hybrid |
| `stage7/units_ane1_*.json`, `stage7/segments_*.json` | conversion-path and segment controls |
| `stage9/refconvert_*.json` | selector vs plain `compute_precision` |
| `run_*.sh`, `render_tables.py`, `splice_memo.py` | the reproducible drivers |

**One process discipline**: stages 5-9 run inside a single governed launch
(`run_tail_serial.sh`, receipt `ane2_tail_serial_5678_v2`) after the coordinator measured that a
chain of per-stage launches is serial in EXECUTION but parallel in declared PEAK — five live
reservations held 42 GiB and refused a sister arm's Metal cell. Peaks are declared from measured
RSS, not rounded.

## Own-vehicle frontier

**fs2 S 0.14784474152757654 @ 180,023 B `[contest-CUDA T4 n600]` — UNMOVED by this arm.**

