# ddm_rz1 — The realization wall: what it is, and what actually attacks it

**Arm:** `ddm_rz1` · 2026-08-03 · **scorer-free** (0 scorer forwards; `ddm_pu2` holds the slot)
**Charge:** `ddm_sx1` (seg) and `ddm_br1` (rate) converged on a wall nobody owned —
*description is cheap, coding is closed, realization is the wall.* Derive what realization is
from `upstream/` alone, measure its capacity, rank the attacks with pre-registered falsifiers.

---

## THE ANSWER, first

> **The realization wall is a CONSTRAINT COLLISION, not a capability gap — and the collision is
> survivable, which is the good news nobody had priced.**
>
> d_seg is **luma-led**: the dominant boundary (Road↔Lane, **50.25%** of all boundary pixels)
> is **76.6% luma-parallel in energy** (MEASURED, n600), and luma is exactly what the **shared**
> `D` hands to PoseNet — `modules.py:73` and `:109` are the identical call. So seg realization
> and pose compete for the *same* degrees of freedom, and `pz1`'s measured **79× pose:seg
> penalty** is what that collision costs when you pay it in luma.
>
> **But the exactly-pose-free subspace is not the dead end it has been read as.** Projecting the
> cross-boundary discriminative direction orthogonally off the luma normal — which is precisely
> the `Q3` constraint — leaves **55.0%** of it (boundary-mass-weighted, n600; 48.4% on
> Road↔Lane). The corpus's own independent n96 margin-gradient split gives **sqrt(0.212) =
> 46.0%**. Two methods, two vehicles, agreeing within 1.2×.
>
> **So the standing corpus claim "Road↔Lane is luma-separable and CANNOT be chroma-repaired" is
> too strong.** At equal perturbation magnitude a pose-free chroma move delivers ~half the margin
> motion of an unconstrained one — at **exactly zero pose cost**, with gamut to spare (§2.3) and
> free addressing (§2.5). Against a 79× pose penalty, a 2× directional loss is a *good trade*.
> That is this arm's main reversal, and §3 re-ranks on it.

Three things this kills, four it ranks, in §3.

---

## NEXT-IF-RESUMED

1. **A2's falsifier is the single highest-value next action, and it is $0 and local.** Two
   existing instruments (`ddm_pz1_dseg_window_solve_n600.py` ~20 min +
   `ddm_pz1_dpose_window_solve_paired.py` ~12 min) answer **three** questions in one probe:
   does an isoluminant nudge lower d_seg · does the uint8 camera lift break exact pose-nullity
   (R4) · does a 0.60-bit amplitude clear the `fd2` staircase. Spec in §3.2.
2. **A1 needs a trainer owner, not more analysis**, and it is *one line*, not a rewire — see
   the §4 correction: the charge's premise ("no `eval_roundtrip` equivalent") is **refuted**;
   `direct_description_joint_descent.py:2419` already computes the realized d_seg and
   **discards** it at `:2359`.
3. **A3 (luma-for-seg / frame_0-for-pose) is the highest-capacity untested attack** and its
   one blocking unknown is a repo fact, not a measurement: **does our vehicle emit frame_0
   independently, or as a warp of frame_1?** If warped, `ddm_pz1` §7.1's lattice law applies and
   the compensator must be an additive frame_0 residual. **I did not resolve this.**
4. **Not done here: the true SegNet Jacobian.** Every sensitivity number in §2.4 is a
   chain-rule spatial proxy (`INFERRED`), deliberately, because this arm is scorer-free — and it
   is **biased optimistic** (§5). One scorer-holding arm replaces it in one pass.
5. No cached per-pixel flip map exists for the live-best `cx1` (corpus §10b, scope stated).
   `experiments/ddm_pz1_dseg_window_solve_n600.py` regenerates one at n600 in ~20 min for $0.
   Every falsifier below is written against that instrument.

---

## §1 — What realization IS (DERIVED from `upstream/` alone, verified numerically)

Derived and verified **before** reading our corpus, per the charge. §4 diffs it against ours.

### R1. `D` is a PARTITION, not a resampling. **The resize is not the wall.**

`modules.py:73` (PoseNet) and `:109` (SegNet) make the *identical*
`F.interpolate(x, size=(384,512), mode='bilinear')` — `antialias=False`, `align_corners=False`.
`frame_utils.py`: `camera_size=(1164,874)`, `segnet_model_input_size=(512,384)`.

- `scale_h = 874/384 = 2.276042`, `scale_w = 1164/512 = 2.273438`. **Both > 2**, so the 2-tap
  supports of adjacent outputs are disjoint. **Measured: 0 row overlaps, 0 column overlaps.**
- Camera rows read **768/874**, columns **1024/1164** → blind fraction
  `1 − (768/874)(1024/1164) =` **0.226969**.
- **Empirical confirmation:** perturb one camera pixel, count affected scorer pixels, 200 random
  sites → `{1: 155, 0: 45}`. **Every camera pixel feeds AT MOST ONE scorer pixel.**

`D` is exactly a disjoint 4→1 pooling: each of the 196,608 scorer pixels reads a **private** 2×2
camera block; 22.6969% of camera pixels feed nothing. Consequences:

- **(a) There is no cross-talk to fight and no aliasing to pre-compensate.** For any target
  scorer-plane value, setting the 4 private camera pixels to it realizes it exactly. **Any claim
  that "we lose because of the resize" is refuted at the pixel level** — this *predicts
  `ddm_pz1`'s n600 loss a priori.*
  **Honest bound:** this says `D` is neutral. It does **not** say the archive can reach an
  arbitrary scorer plane — under a byte budget the renderer's expressiveness is the real limit.
  What R1 establishes is *where the wall is not*.
- **(b) Camera coordinates are a 5.1744× redundant description basis** (1,017,336 / 196,608).
  Null dimension `1,017,336 − 196,608 = 820,728` = **80.6742%** — which is exactly the
  independently-measured "resize nullity" of corpus #580. Cross-check to 4 decimal places. ✅
- **(c) 22.6969% of camera pixels (230,904) are blind to BOTH scorers** and must receive zero
  bits.

### R2. The pose-free actuator, and its exact authority law

`rgb_to_yuv6` (`frame_utils.py:48`) maps each 2×2 scorer block's 12 RGB numbers to 6 outputs
(four luma phases + the 2×2 means `U_sub,V_sub`) ⇒ **6-dim null space per block × 49,152 blocks
= 294,912 exactly pose-free dims per frame.** (Independently reproduces the corpus `Q3` count.)

With `dv=V−128`, `du=U−128`: `R=Y+1.402dv`, `B=Y+1.772du`, `G=Y−0.71414dv−0.34415du`. The
feasible `(du,dv)` hexagon has inradius `min(Y,255−Y)/1.772`; since `1.772 > 1.402 > 0.79274`
the `B` constraint always binds:

> **Guaranteed isotropic RGB reach of the pose-free chroma actuator = `min(Y, 255−Y)`.**

Brute-force verified on the uint8 cube. UV gamut area ≥25% of peak only for `Y∈[34,221]`, ≥10%
only for `Y∈[22,233]`, **exactly zero at `Y=0,255`**. *(Note this is the inradius — guaranteed in
every direction — so §2 reachability figures are conservative; the hexagon is elongated and the
`B`-direction reach is ~2× larger.)*

**Caveat found by attacking my own derivation:** `rgb_to_yuv6` **clamps** `U,V` to `[0,255]`, and
in-gamut RGB reaches `U,V=255.5` at the pure-blue/pure-red corners. Within 0.5/255 of those two
corners the clamp activates and block-mean preservation — hence exact pose-nullity — **breaks**.
Negligible in area; an implementation must exclude those corners.

### R3. `W`, flip counts, and `sx1`'s headline — re-derived independently, matching to the digit

`W = 4·37,545,489 / 117,964,800 =` **1.2731082153320** B/flip. `d_seg(cx1)=0.004311790` →
**508,639 flips**. Gap to the PR130 floor (0.0002966) → **473,651 flips** → budget **603,009 B**.
`sx1`'s exact-`L*` cost **253,341 B = 42.01%** → residual **349,668 B = 0.7382 B = 5.91
bits/gap-flip.** Independent reconstruction of `sx1` from the live constants. ✅

### R4. Scorer-plane precision survives the uint8 camera lift; exact pose-nullity does not

Achievable scorer value = `{w·v : v∈{0..255}^4}`. Worst max-gap over 12 sampled sites
**0.861/255** ⇒ **≥8.21 bits/channel** (an *upper* bound on the gap — the enumeration was
decimated — so true precision is better). uint8 camera pixels give *more* than 8 bits in the
scorer plane, not less.

**Consequence, a live correction to a standing claim:** `Q3`'s "d_pose EXACTLY 0" (corpus:
5.684e-14) is exact **in the scorer plane**. The archive must emit uint8 **camera** pixels, and
exact luma preservation would need an exact linear relation among four irrational tap weights,
which generically does not exist on the integer lattice. After the lift, luma is preserved only
to **≤~0.9 LSB**. Sub-quantization, but **not zero**; its size in d_pose units is
`ASSUMED_AWAITING_VERIFICATION`. *(The corpus's exact `min|δY| = 1/1000` result is the sister
bound for a **direct scorer-plane integer** step; mine is the bound for the **camera 4-tap
lift**, which is the path an archive must actually take.)*

---

## §2 — Capacity of the wall (MEASURED, n600, scorer-free)

Source: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (sha
`cf8d83605d…`) — `gt_f1 (600,874,1164,3) uint8`, `lstars (600,384,512) int64`,
**`margins (600,384,512) float32`**. The cached per-pixel GT margin field is what makes this
section scorer-free. Separatrix `= |margin| ≤ 0.801940` (the 2.16% quantile) →
**2,566,212 / 117,964,800 px = 2.175%**, reproducing `sx1`'s separatrix mass.

**Class indices verified independently by spatial signature** (never luma-sort, per CLAUDE.md):
mean row — MyCar **334.6** (bottom) ✓, Undrivable **95.0** (top/sky) ✓; shares Road 23.23% /
Lane 0.586% / Undrivable 49.52% / Movable 1.238% / MyCar 25.43%. Canonical comma10k order. ✅

### 2.1 — **The decisive measurement: how much of the discriminative direction survives the pose-free constraint**

The `Q3` actuator moves **exactly** within the isoluminant plane `{δ : δ·n = 0}`,
`n = (0.299, 0.587, 0.114)`. So its efficacy against an unconstrained move of the same magnitude
is `|P_n⊥ g| / |g|`, where `g` is the margin gradient. Using the **cross-boundary RGB difference
as the proxy for `g`** (n600, scorer plane, GT frame_1):

| edge | boundary px | share | rms \|Δ\| | luma-parallel (energy) | **ISOLUMINANT EFFICACY** |
|---|---:|---:|---:|---:|---:|
| **Road↔Lane** | 814,066 | **50.25%** | 36.27 | 76.6% | **48.4%** |
| Road↔MyCar | 317,679 | 19.61% | 9.48 | 63.8% | 60.1% |
| Road↔Undrivable | 290,167 | 17.91% | 8.81 | 58.4% | 64.5% |
| Undrivable↔Movable | 99,530 | 6.14% | 54.25 | 65.2% | 59.0% |
| Road↔Movable | 90,322 | 5.58% | 21.58 | 60.1% | 63.2% |
| Lane↔MyCar | 5,112 | 0.32% | 31.56 | 76.8% | 48.2% |
| Lane↔Undrivable | 1,587 | 0.10% | 27.41 | 76.8% | 48.2% |
| Lane↔Movable | 1,297 | 0.08% | 29.55 | 73.6% | 51.4% |

> **Boundary-mass-weighted isoluminant efficacy = 55.0%.**
> Independent cross-check: the corpus's *direct* n96 SegNet margin-gradient split (78.8% luma /
> 21.2% chroma) gives `sqrt(0.212) =` **46.0%**. Two independent methods on two different
> vehicles, agreeing within **1.2×**.

**This overturns a standing corpus claim.** *"Road↔Lane (41% of Road's flips) is LUMA-separable
— bright lines on dark road — and CANNOT be chroma-repaired"* is **too strong**: the edge is
luma-*led* (76.6% of energy) but retains **48.4%** of its discriminative amplitude in the
isoluminant plane. Lane markings are bright paint on grey asphalt, so the difference is largely
but **not purely** achromatic — and critically, the luma normal `n` is *not* parallel to
`(1,1,1)`, so even a perfectly neutral grey→white step is only 85% luma-parallel in amplitude.

**Honest limit on this number.** The proxy `g ≈ Δ` is the weak link, and it is load-bearing here:
if SegNet's true `g` at this edge were exactly parallel to `n`, efficacy would be **0**, not
48.4%. My measurement cannot distinguish `g ∥ n` from `g ∥ (1,1,1)`. What resolves it is the
corpus's *direct* gradient measurement (46.0%) and its *direct* authority sweep — an ideal
isoluminant chroma checkerboard at amp 32 moves **Δd_seg = 2.73e-3** at n96, which is **68% of
our entire current seg gap** (4.015e-3). The actuator demonstrably moves argmax at scale.

**This also resolves the corpus's explicitly-contested reading** (§9.9 of the quartering audit:
§8 "shared blind space, chroma is not a free seg actuator" vs §3b "chroma is a real actuator"):
**§3b is right.** The actuator exists (294,912 exact dims), has gamut to spare (§2.3), retains
~half the discriminative direction, and has measured authority worth 68% of the gap. §8's
"blind space" reading conflated *luma-led* with *luma-only*.

### 2.2 — Lane has no interior, and it is GEOMETRY not low confidence

| class | % of separatrix | separatrix rate | mean Y (all) | mean Y (sep) |
|---|---:|---:|---:|---:|
| Road | 48.16% | 4.509% | 44.5 | 40.3 |
| **Lane** | **17.18%** | **63.840%** | 71.5 | 59.9 |
| Undrivable | 15.54% | 0.683% | 13.8 | 35.0 |
| Movable | 8.38% | 14.730% | 74.6 | 57.1 |
| MyCar | 10.74% | 0.919% | 9.8 | 31.1 |

**63.84% of all Lane pixels are decision-boundary pixels.** I attacked my own claim — is this
thinness (geometry) or is SegNet simply unconfident about Lane everywhere (confidence)? Erosion
shells inward from each class's own boundary, n600:

| depth | Lane px | Lane sep% | Road px | Road sep% |
|---:|---:|---:|---:|---:|
| 0 | 518,278 | 83.66% | 1,251,268 | 80.33% |
| 1 | 124,537 | 5.66% | 1,243,000 | 9.60% |
| 2 | 40,799 | **0.44%** | 1,182,243 | 3.63% |
| ≥2 total | **47,823 = 6.92% of Lane** | | **4,365,275 = 63.64% of Road** | |

**GEOMETRY, decisively.** At depth 0 Lane and Road are *equally* uncertain (83.66% vs 80.33%);
two pixels inside a Lane region SegNet is **more** confident than inside Road (0.44% vs 3.63%).
Lane's separatrix rate is high purely because it has **6.92% interior against Road's 63.64%**.
A region-paint primitive has essentially nothing to paint for Lane — it can only produce Lane
where two painted regions happen to meet. This is a primitive-level explanation for why
Road↔Lane is the largest flip family and why Lane's temporal IoU is 0.263 against ≥0.90
elsewhere. **It also means a primitive that gets Lane geometry right would produce confident
Lane pixels** — the classifier is not the problem.

### 2.3 — The pose-free actuator is NOT gamut-limited (my own hypothesis, refuted)

I predicted before measuring: *lane markings are bright ⇒ low chroma authority ⇒ the actuator
cannot reach the biggest flip family.* **Measured: false.** Lane has the *highest* authority of
any class; its mean separatrix luma is 59.9 — mid-tone, not blown out. `A = min(Y,255−Y)`:

| | A<10 | A<25 | A<35 | A≥35 |
|---|---:|---:|---:|---:|
| **separatrix, ALL** | **3.19%** | 28.31% | 45.93% | **54.07%** |
| all pixels, ALL | 35.39% | 73.81% | 80.78% | 19.22% |

> **The separatrix is 2.81× enriched in high-chroma-authority luma (54.07% vs 19.22% at A≥35),
> and only 3.19% of separatrix pixels are authority-dead.**

The dark classes are authority-dead in *bulk* (MyCar mean Y 9.8, 73.13% at A<10) but not at
their *separatrix* (12.41%). The flip-prone pixels sit where the actuator has the most room —
and §2.1 shows that room is in the wrong direction. **This fills the corpus's named
`OPEN LIMIT 2`** (the uint8-reachable / gamut-limited fraction of `Q3`, previously UNMEASURED):
the answer is *gamut is not the limit*.

### 2.4 — Reachability curve (`INFERRED`, biased optimistic — see §5)

Estimator `|∇_spatial I| / |∇_spatial m|` ≈ RGB units per logit. A proxy for `∂m/∂RGB`, **not**
the Jacobian. Fraction of separatrix px where guaranteed reach `A` ≥ RGB needed for `T` logits:

| class | sep share | T=0.25 | T=0.5 | T=1 | T=2 | T=4 |
|---|---:|---:|---:|---:|---:|---:|
| Road | 48.16% | 94.9% | 87.9% | 72.0% | 48.1% | 25.8% |
| Lane | 17.18% | 91.3% | 76.5% | 49.9% | 25.1% | 9.9% |
| **ALL** | 100% | **91.9%** | **81.9%** | **63.7%** | 40.9% | 21.4% |

**Prefix control applied** (memory `bp2`): the n=200 *prefix* gave ALL/T=1 = 63.47%, the full
n=600 population **63.70%** — 0.4% relative. Representative *for this quantity*. Reported because
the control was run, not because it was assumed.

### 2.5 — Three budget denominators, and separatrix addressing is FREE

`sx1`'s 5.91 bits/flip **assumes the identity of the flipping pixels is free**. A decoder cannot
run SegNet. But I attacked that too, and it is *half* free: the **dilated spatial label boundary
— computable from the decoder's own `L*` with no scorer —** recovers **93.53%** of the margin
separatrix at 51.24% precision, occupying 3.971% of pixels.

| denominator | count | budget |
|---|---:|---:|
| gap-flips (identity free — optimistic) | 473,651 | 0.7382 B = **5.91 bits/flip** |
| margin-separatrix px | 2,566,212 | 0.1363 B = **1.09 bits/px** |
| **free-addressable boundary band** (recall 93.53%) | 4,684,382 | 0.0746 B = **0.60 bits/px** |

**So: addressing is free; only the nudge *value* costs.** And 0.60 bits/px means a **sparse
ternary** (most sites "no nudge") — 1 flat bit/px would cost 585,548 B = 1.67× the entire
residual budget. Any attack costed against 5.91 bits must first say how the decoder learns
*which* sites flip.

---

## §3 — The ranked attack table

### The rule that ranks them — generalized from `ddm_pz1`

`pz1` removed rasterization error and **lost** (Δd_seg +0.000394, Δd_pose +0.0310217, 43.3% of
pairs improved / 49.3% worsened). §1 R1 says why a priori: `D` is a lossless-onto partition, so
the removed component was never the wall. **Important correction to my own first draft:** pz1
did **not** compute a cosine against the GT error (corpus §3, exhaustively confirmed absent),
and MAIN's own stop-hook downgraded "orthogonal" to **wrong-signed, 788× the reproducibility
floor**. So the effect was *anti*-aligned, not null.

> **REALIZATION LAW.** An attack that reduces `‖ΔX‖` without being **positively margin-aligned**
> buys a random or adverse sign in d_seg — and, because `D` is SHARED, still pays full pose.
> `pz1`'s pose loss was **79×** its seg gain; that is the standing prior for any non-pose-null
> camera-plane edit.
>
> **Cheapest thing in this memo:** every future realization proposal must state (i) its
> `cos(direction, ∇m)` and (ii) its pose leakage, *before* it costs anything. No one has yet
> computed (i) for any landed proposal.

### 3.1 — **A1. Stop discarding the realized margin.** Rank 1. Zero counted bytes.

- **Mechanism.** `direct_description_joint_descent.py:2359` trains CE on the *label*; four lines
  later `:2419` computes the realized `d_seg = mean(argmax != targets)` **on the same logits and
  discards it** (`_loss:2349` unpacks `seg, pose_mse, _`). Replace the CE surrogate with the
  exact power-diagram margin on a forward pass that **already realizes the whole trip**.
- **Cost.** Zero counted bytes. Per `ddm_er1`: "a materially smaller and better-targeted build
  than the charter assumed."
- **Why rank 1 given §2.1.** The wall is a luma↔pose constraint collision. The only thing that
  can *price* luma against pose is a **joint realized objective**. A label objective cannot: it
  has no notion of how much margin survives, so it cannot know when a luma move is worth its
  pose cost. This is the same discipline CLAUDE.md makes non-negotiable for every other training
  path in the repo, and the describe path is the one path that still optimizes a surrogate.
- **Falsifier (pre-registered).** Matched pair at identical description budget and seed:
  (a) CE-on-label, (b) realized-margin. Compare d_seg **and** d_pose at n600 through the real
  byte-close (`tac.contest_score --n-pairs 600 --device cpu`). **Kill if** Δd_seg ≤ the
  single-seed noise floor. **Kill if** the margin objective improves proxy margin but not
  realized d_seg. **Report d_pose beside d_seg against `joint_finish_d_pose_max = 0.001610`** —
  `uv1` measured a **3,019× d_pose separation** between two bases under an otherwise identical
  solver, so a seg-only A/B here is forbidden.
- **Scope.** Alive for the **describe path**. **`DERIVED-REDUNDANT` for TR1**, whose seg form is
  already a margin objective (`tau_softplus`, `train_witness_realized_through_R_mlx.py:1454-59`).
- **Honest.** This is a retrain. If the answer is "realization is attackable only by a retrain",
  A1 *is* that answer — and it **corroborates `br1`'s independent conclusion from the rate side**
  ("the live direction is the lattice SHAPE — needs a retrain + the scorer") rather than
  contradicting it. Two arms, two axes, same verdict, again.

### 3.2 — **A2. Pose-free chroma steering at the boundary band.** Rank 2 — **re-promoted by round 3.**

- **Status.** I opened this as the leading candidate, demoted it to rank 4 on a §2.1 model that
  round 3 found to be **wrong** (it applied the luma/chroma energy split *twice*, squaring the
  penalty into a spurious "4.6%"), and re-promoted it on the corrected projection. The audit
  trail is left visible on purpose: the corrected number is **55.0%**, cross-checked at 46.0%.
- **Why rank 2.** It is the only attack that is **exactly pose-free by construction**
  (`Q3`: pose max\|Δ\| = 5.684e-14, seg max\|Δ\| = 6.0). Against `pz1`'s measured **79× pose:seg
  penalty** for luma edits, paying a **2× directional loss to buy exact pose-freedom is a good
  trade** — and that trade has never been priced this way.
- **Everything it needs, it has.** Capacity 294,912 exact dims/frame · gamut **2.81× enriched**
  at the separatrix with only 3.19% authority-dead (§2.3) · addressing **free** from the
  decoder's own `L*` at 93.53% recall (§2.5) · budget **0.60 bits/px** = a sparse ternary ·
  measured authority **2.73e-3 d_seg** = 68% of the current seg gap (n96, amp 32).
- **Realizability — a genuine upgrade this arm hands over.** The corpus records "camera-res
  realization **fails**: a period-2 camera-res luma-preserving chroma cosine arrives at the yuv6
  plane as 50% luma." **R1 explains it exactly and cures it.** A camera-plane pattern that is
  zero-mean over a 2×2 block is *not* zero-mean under the **unequal** bilinear tap weights, so it
  folds into that scorer pixel's own luma — the fold is **within** the block, never across it
  (supports are disjoint). Cure: design in the **scorer plane**, then lift by setting all four
  private camera pixels to the same target (weights sum to 1 ⇒ exact). The corpus's prescription
  "band-design at the scorer's 384 grid" is right; R1 supplies the mechanism and the exact lift.
- **Two threats I did NOT clear.** (i) `ddm_fd2` measured a **staircase** — description steps at
  ×0.5 and ×0.25 changed the frames and produced **zero argmax flips**; a 0.60-bit nudge may land
  below the first gate. A2 partly escapes this by writing at scorer-plane sites rather than
  through the description coder, but `OPEN LIMIT 1` stands: reaching SegNet's input at 6.0/255
  ≠ moving its argmax. (ii) The corpus's precision-coupling rule binds — **any `Q3` correction
  must carry a margin floor**, else a near-tie edit is a device-dependent coin flip, not a gain.
- **Falsifier (pre-registered, ONE probe, returns two answers).** On the live-best decode (`cx1`,
  sha `1d3ab694`, S=0.8264972), apply a ±1-step isoluminant chroma nudge along `P_n⊥` of the
  local target-class direction at every free-addressable boundary-band pixel of frame_1, with a
  margin floor. Measure Δd_seg via `experiments/ddm_pz1_dseg_window_solve_n600.py` (~20 min, $0)
  and Δd_pose via `experiments/ddm_pz1_dpose_window_solve_paired.py` (~12 min, $0).
  **Kill the direction if** Δd_seg ≥ 0. **Kill the exactness claim if** |Δd_pose| > 1e-6 — which
  simultaneously tests R4's prediction that the uint8 camera lift breaks exact pose-nullity at
  the ≤0.9 LSB scale. **Kill the amplitude if** Δd_seg = 0 exactly (the fd2 staircase).
- **Consumer.** A decode-side correction section; `ddm_pu2` or the next scorer holder.

### 3.3 — **A6. Lane needs a curve primitive, not a region primitive.** Rank 3.

- **Mechanism.** §2.2: Lane is 6.92% interior — geometrically, not by low confidence. §2.1:
  Road↔Lane is 50.25% of boundary mass and **76.6% luma-parallel in energy** — the most
  luma-led edge in the frame. A curve rasterizer delivers *luma
  structure at geometric precision*, which is exactly and only what this edge responds to.
- **Rate.** CLAUDE.md already classifies a parametric lane rasterizer (openpilot polynomial +
  homography, `tac.lie` SE(3)) as a **generic, video-independent algorithm — free in
  `inflate.py`**; only the ~8-dim per-frame coefficients are counted.
- **Prize.** Road↔Lane = 49.2% of flips = **22.1% of the entire gap** (`ddm_pc2`).
- **Falsifier (pre-registered).** Rasterize Lane from GT lane geometry alone at the archive's
  byte budget, composite over the current decode, measure at n600 via
  `experiments/ddm_pz1_dseg_window_solve_n600.py`. **Kill if** Δd_seg does not improve by ≥ the
  Lane share of the separatrix — that would mean Lane flips are driven by Road-side context, not
  by Lane's own realization. **Also report d_pose:** a luma-carrying primitive is *not* pose-free.
- **Owner.** The renderer primitive set. **Not this arm** — routing it is the point.

### 3.4 — **A3. Luma for seg in frame_1, pay the pose debt in frame_0.** Rank 4 — highest capacity, untested.

- **Mechanism.** SegNet reads **only** `x[:,-1,...]` (`modules.py:108`). PoseNet reads both
  frames. Therefore the corpus's `Q2` (frame_0 non-null, **294,912 dims, POSE-only, seg max|Δ|
  exactly 0.000e+00**) is an exact compensator for the pose debt incurred by a luma edit in
  frame_1 (`Q4`). **d_pose is an MSE over 6 scalars per pair = 3,600 numbers at n600, against
  294,912 compensating dims per pair — over-determined ~49,152×.** The pose debt should be
  fully cancellable; the only question is bytes, and the corpus already has
  `e_p RANK-1 ~2KB MEASURED-CLOSED` plus `R1 dxi 7.2KB` as existence proofs of cheap pose
  sections.
- **What this reframes.** This is *why* the standing "pose AFTER frozen seg" ordering (#383)
  works. It has not previously carried a capacity argument; §2.1 supplies the reason it is
  **mandatory** rather than merely convenient — chroma cannot do the seg job, so luma must, so
  a pose compensator is not optional.
- **The one blocking unknown, and I did not resolve it.** If our vehicle emits frame_0 as a
  **warp of frame_1**, `Q2` is not independently settable and `pz1` §7.1's law applies
  ("null-space membership does not survive a change of lattice"; measured attenuation only
  1.662×, frame_0 delta 2.12× the frame_1 debt). Then the compensator must be an **additive
  frame_0 residual**, which costs bytes but still only has to fix 6 numbers per pair.
  **Resolve this repo fact before designing anything.**
- **Falsifier (pre-registered, 2 existing $0 instruments).** Apply a luma-only frame_1 edge
  sharpening at Road↔Lane sites; measure Δd_seg with
  `experiments/ddm_pz1_dseg_window_solve_n600.py` (~20 min) and Δd_pose with
  `experiments/ddm_pz1_dpose_window_solve_paired.py` (~12 min). **Then** fit the minimal frame_0
  compensator and re-measure d_pose. **Kill if** the residual d_pose after compensation exceeds
  `0.001610`, or if the bytes to cancel it exceed the seg gain in `W`-equivalents.

### 3.5 — **A4. A terminal per-site solve.** Rank 5, conditional. Do not re-open naively.

`ddm_fd2` diagnosed `fd1`'s zero-accept as **`SEG_REALIZATION_GAP_AT_UINT8_DOMINANT`** —
canary PASSED, pose-veto **REFUTED** (0/6 seg-only accepts; 5/6 bit-identical d_seg), no locality
signature, and at the amplitude where flips finally appear the realization is already
**anti-gradient**. "Amplitude tuning alone is measured NOT to recover fidelity." **Diagnosed,
never cured.** §1 adds an independent reason to expect failure: a solve whose residual is a
**reconstruction norm** is norm-aligned, and the REALIZATION LAW predicts a random/adverse sign.
**A solve is admissible only if its residual is the margin hinge itself**, and only if it meets
fd2's named precondition — integer-aware proposals at named near-margin sites, or training
feedback that includes realized flips (which is A1).

### 3.6 — Predicted NULL, with reasons (do not spend on these)

| proposal | predicted | reason |
|---|---|---|
| pre-compensate / inverse-filter the resize | **null — already MEASURED null** | R1: `D` is a lossless-onto partition. `pz1` retired it. Do not re-run. |
| dither / anti-alias the raster | **null** | `ddm_ra1`: 93.5% of camera-raster debt is in exact float; uint8 is 6.0%. (n=4 strided.) |
| spend bits on the 22.70% blind camera pixels | **null by construction** | R1(c): read by neither scorer. |
| index a per-pixel correction in camera coordinates | **5.17× waste** | R1(b). |
| ~~chroma repair of Road↔Lane~~ | **RETRACTED by round 3** | I listed this as null on the erroneous 0.7%. Corrected: Road↔Lane retains **48.4%** isoluminant efficacy (§2.1). Not null. |
| any non-pose-null camera-plane seg edit **without** a frame_0 compensator | **dominated** | shared `D`; `pz1`'s 79× pose:seg prior. |

---

## §4 — Diff against our corpus (derivation was done first)

**Corroborated, independently:** blind set 22.6969% / 230,904 px ≡ corpus; resize nullity
**80.6742%** ≡ corpus #580 to 4 dp; `Q3` = 294,912 exact pose-free dims ≡ corpus quartering;
`W`, flip counts and **5.91 bits/flip** ≡ `sx1`; shared `D` ≡ `pz1` §0(4); `pz1`'s failure
predicted a priori from R1.

**Two corrections to the charge / to my own first draft:**

1. **The charge's premise "the describe path has no `eval_roundtrip` equivalent" is REFUTED**
   by `ddm_er1`. The trip *is* realized inside the forward
   (`direct_description_joint_descent.py:2345-2363`: STE round to uint8, `fused_r_roundtrip`,
   real MLX SegNet, real MLX PoseNet on YUV6 from the same round-tripped pair), and
   `apply_eval_roundtrip_during_training` already exists at
   `src/tac/differentiable_eval_roundtrip.py:213`. Only the *label-not-margin* clause is true.
   A1 is rescoped accordingly and is much cheaper than the charge assumed.
2. **My own leading hypothesis was wrong.** I opened expecting the pose-free chroma actuator to
   be gamut-blocked at bright lane markings. §2.3 refutes it (authority 2.81× *enriched*); §2.1
   then supplies the real limit (orientation). Recorded because a refuted hypothesis that
   changes the ranking is the finding.

**Contributed to open corpus questions:** filled `OPEN LIMIT 2` (gamut is not the `Q3` limit);
resolved the §9.9 contested reading (both readings correct, about different things — the limit
is orientation, named by neither); quantified "Road↔Lane is luma-separable" into a per-edge
n600 table; explained *and cured* "camera-res realization fails" via R1.

**Inherited, not re-measured:** Road↔Lane = 49.2% of flips = 22.1% of gap (`ddm_pc2`); the
78.8/21.2 margin-gradient energy split (n96); `v14`'s 1700× (a ratio between two measurement
surfaces, never stage-localized — its own memo declines to localize it, correctly).

---

## §5 — Assumption ledger

| # | assumption | status |
|---|---|---|
| 1 | `D` shared, bilinear, antialias=False, 874×1164→384×512 | `VERIFIED_VIA_SOURCE_INSPECTION` (`modules.py:73,109`; `frame_utils.py:13-15`) + re-derived numerically |
| 2 | each camera px feeds ≤1 scorer px; blind 0.226969; nullity 80.6742% | `VERIFIED_VIA_SOURCE_INSPECTION` + brute-forced (200 sites) + matches corpus #580 |
| 3 | yuv6 null = 6 dims / 2×2 block, exactly pose-invisible | `VERIFIED_VIA_SOURCE_INSPECTION` (`frame_utils.py:48-79`) + matches corpus `Q3` |
| 4 | reach `= min(Y,255−Y)`; gamut ≥25% peak only `Y∈[34,221]` | `VERIFIED_VIA_SOURCE_INSPECTION` + brute-forced on the uint8 cube |
| 5 | `U,V` clamp breaks exact nullity within 0.5/255 of pure-blue/red | `VERIFIED_VIA_SOURCE_INSPECTION` (`clamp_` in `rgb_to_yuv6`) |
| 6 | separatrix 2.175% @ \|m\|≤0.80194; per-class shares; authority tables; erosion depths; **per-edge luma shares** | `VERIFIED_VIA_EMPIRICAL_ANCHOR` — n600, all 117,964,800 px, cached GT, 0 scorer forwards |
| 7 | class index ↔ name mapping | `VERIFIED_VIA_EMPIRICAL_ANCHOR` — self-detected by spatial signature (MyCar row 334.6 bottom, Undrivable 95.0 top), never luma-sort |
| 8 | `W`, flip counts, 5.91 / 1.09 / 0.60 bits denominators | `VERIFIED_VIA_EMPIRICAL_ANCHOR` — re-derived from live constants; matches `sx1` and the pointer to the digit |
| 9 | RGB-per-logit sensitivity (§2.4) | **`INFERRED`** — chain-rule spatial proxy, **biased optimistic**: where margin varies faster than the image because of *context* rather than the local pixel, `\|∇m\|` is large and the ratio understates the true cost. Ranks; does not bound. |
| 10 | isoluminant efficacy column (§2.1) | **`INFERRED`** — the **decomposition is exact** (orthogonal projection off the true luma normal, which *is* the actuator's constraint) but it is applied to the cross-boundary RGB difference as a **proxy for the margin gradient**. Load-bearing: if `g ∥ n` exactly, efficacy is 0, not 48.4%. Cross-checked at 46.0% by the corpus's *direct* n96 gradient split and by its measured 2.73e-3 authority sweep. |
| 11 | d_pose perturbation after the uint8 camera lift ≤0.9 LSB luma | `DERIVED`; **size in d_pose units `ASSUMED_AWAITING_VERIFICATION`** |
| 12 | Road↔Lane = 49.2% of flips = 22.1% of gap; 78.8/21.2 split | inherited (`ddm_pc2`; n96 chroma probe) — **not re-measured here** |
| 13 | whether our vehicle emits frame_0 independently or as a warp of frame_1 | **`UNKNOWN` — blocks A3, not resolved by this arm** |

**Review counter: 0 clean passes of 3.** Rounds and what they changed:

- **R1** attacked five of my own claims and changed four: class order upgraded to
  self-detected-by-spatial-signature; separatrix addressing found to be **free** from `L*`
  (93.53% recall) — which added the 0.60-bit denominator; Lane's 63.84% resolved as
  **geometry not confidence** by erosion shells; my leading gamut hypothesis **refuted**.
- **R2** found the §2.4 optimistic-bias direction and the §2.1 model-vs-measurement split, and
  corrected `pz1` "orthogonal" → **"anti-aligned"** (MAIN's own stop-hook: wrong-signed, 788× the
  reproducibility floor).
- **R3** found a **material error in my own §2.1**: the "chroma-reachable" column applied the
  luma/chroma split *twice*, squaring the penalty into a spurious **4.6%**. Corrected by an exact
  orthogonal projection off the luma normal → **55.0%**, cross-checked at 46.0%. This
  **reversed the ranking** — A2 went from rank 4 back to rank 2.

R3 found a material error, so per CLAUDE.md ("round-finished ≠ clean pass"; the counter resets on
every finding) this memo is **round-3 output at 0 clean passes, NOT sealed work** — the same
honest status `sx1` carries. The single most likely place for a fourth finding is assumption
**#10**: everything in §2.1, and therefore A2's rank, rests on the cross-boundary colour
difference standing in for the margin gradient.
