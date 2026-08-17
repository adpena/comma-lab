---
arm: ddm_rn1
title: "hv1 is a TASK-SPACE WITNESS (luma corr -0.480 vs GT, 5.95x SHARPER at the boundary, not blurrier) so the GT-referenced blur/registration questions are ill-posed; the real mechanism is that the receiver is BLIND to the scorer, which closes the undirected decode-side family by a measured symmetry (rho(0.01)=0.985, a fair coin) and leaves POSE, not seg, as the binding constraint -- a +-1 LSB dither is seg-neutral (56 fixed / 55 broken) and still costs +0.0024 S on pose, 25% of the whole gap; the prize that remains is a training-side ring-0 margin hinge worth 1.53x the gap at +0.1 logits"
utc: 2026-08-16
parent: ".omx/research/ddm_rt1_seg_roundtrip_decomposition_20260816.md"
axis: "[macOS-CPU advisory] frozen CPU-torch SegNet + PoseNet -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "INSTANCE on the hv1 ep0634 vehicle; family verdicts only where named"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_rn1 — why the seg axis routes to the renderer

STORES CONSULTED: parent `ddm_rt1_seg_roundtrip_decomposition_20260816.md` (all sections) ·
`ddm_ll1_window_solve.py` + `ddm_pz1_dseg_n600_cx1_20260803.json` ·
`l28_engineered_corrections_witness_remeasure_negative_20260701.md` ·
`signal_processing_filter_levers_derived_20260701T014119Z.md` ·
`dseg_side_feasibility_corners_verdict_20260619.md` (#149) ·
`ddm_lr2_legal_realization_ladder_20260804.md` (A1/A2/A3) ·
`ddm_ra1_rasterization_crossing_20260802.md` ·
`reports/levelset_render_side_sizing_l7best_n600_20260701.json` (AA-on-witness) ·
`reports/aa_sdf_observation_render_verify_n600_20260701.json` ·
`ddm_hv1_pointer_move_and_wd2_advisory_chain_20260815.md` · `upstream/modules.py` ·
memories [[m88]] [[m96]] [[m91]], CLAUDE.md "SegNet vs PoseNet importance — operating-point
dependent".

## ANSWER FIRST

**rt1's routing verdict is right, and the reason is stronger than rt1's.** The seg axis routes
to the renderer not because the post-hoc levers happened to lose, but because **the receiver is
structurally blind to the scorer**, and every decode-side lever collapses onto that one fact.

1. **hv1 is a TASK-SPACE WITNESS, not a video reconstruction — MEASURED, and it voids two of
   my three charter questions as posed.** n=16 seeded-random pairs: luma correlation between
   our decode and GT is **−0.480** (identical at camera resolution and in the scorer lattice),
   RMSE **133.7 / 255**, luma mean **+103.5**, and our field carries **2.22×** GT's gradient
   energy over the whole frame and **5.95×** on the label-boundary ring. Our render is not
   blurrier than GT. It is **anti-correlated with GT and six times sharper at the edge**, and
   it still reproduces GT's SegNet argmax to 99.97%. So "is our render blurrier / shifted /
   photometrically off vs GT" is not a well-posed question on this vehicle, and my own opening
   hypothesis was **backwards**. I found that by measuring it, and I record the diff below.

2. **The mechanism is not blur, not uint8, not R, not tie-breaking. It is missing SIGN
   information.** SegNet weights are **38,502,892 B = S 25.64** if shipped — 2,672× the whole
   remaining gap — and CLAUDE.md forbids shipping them at all. The receiver therefore cannot
   evaluate its own argmax, cannot know where it is wrong, and cannot know which way to move.
   Every decode-side operator is consequently either **UNDIRECTED** (carries no sign) or
   **DIRECTED** (the sign must be paid for in bytes). Both halves are now closed.

3. **The undirected half is closed by a measured symmetry, not by any operator's bad luck.**
   New quantity, never measured before: the **exchange rate**
   `rho(delta) = #{correct boundary px with margin < delta} / #{flips with deficit < delta}`.
   At n=96 seeded-random pairs, **rho(0.01) = 0.985** — a statistical tie (404 flips vs 398
   correct at risk). An undirected perturbation at the boundary is a **fair coin**, exactly as
   continuity of the margin field demands. Reach and cost are then coupled: to reach half the
   flips you need delta = 0.1, where **rho = 2.14**; at delta = 0.3 you reach 85.6% at
   **rho = 7.06**. There is no delta where an undirected operator is a net seg supplier.

4. **The race confirms it on 14 operator rows, and the model predicts the damage.** Ten
   operators at n=12 and five at n=24: `dither +-1/+-2`, `gain 0.98/1.02/1.05`, `gamma 1.02`,
   `unsharp` x2, `blur`, `shift 0.25px`. The measured broken/fixed ratio tracks rho: the
   gentlest operator lands at the fair coin (`dither +-1`, n=24: **56 fixed / 55 broken**), the
   harshest at 2.97 (`shift 0.25 px`). No operator reduces flips beyond noise.

5. **POSE, not seg, is the binding constraint on this vehicle — and that is the new
   operating-point fact.** Every operator's seg effect is bounded by rho and is small; its pose
   effect is not. `dither +-1 LSB` is seg-**neutral** (56 fixed / 55 broken) and still costs
   **+0.0024 S** on pose — **25% of the entire remaining gap for a one-LSB change.**
   `gain 1.02` is the one row that *won* on seg (34 fixed / 27 broken, within noise) and lost
   **+0.0104 S** on pose. At hv1's operating point the pose marginal is **6.03× seg's**
   (`dS/d(d_pose) = 5/sqrt(10 d_pose) = 602.8` against seg's constant 100), so the decoded frame
   field is effectively **frozen**: it cannot be touched at all, in any direction, for any
   reason. Those pose figures are the **optimistic** bound — my advisory pose instrument carries
   a measured **18.2×** baseline offset that suppresses the damage by roughly that factor
   (§3.2b, measured and worked through, not waved at).

6. **The directed half was already closed by rt1** — eta 0.6235 (n=9) against a required 0.753,
   32,270 B of real coded bytes, a non-supplier at +0.0025 S. I re-price nothing; I supply the
   *reason* rt1's channel must exist at all: the sign is not free, and rho says nothing cheaper
   than the sign will do.

7. **What is left is a training-side prize, and it is now quantified.** Only training knows the
   sign, because only training has GT. A **signed** margin shift of +delta at ring-0 pixels
   recovers every flip below delta and breaks nothing, because the same shift pushes correct
   pixels further from the boundary. Measured deficit ladder, n=96:
   **+0.1 logits recovers 49.7% of the axis = 0.014711 S = 1.53× the remaining gap**;
   +0.3 recovers 85.6% = 2.64×. That is the number the wd3 / ns1-P1 line should aim at, and it
   is the first time the prize has had a coefficient.

8. **R independently re-derived, and it is a selector, not a filter.** `F.interpolate(...,
   mode='bilinear')` with `antialias` unset gives an exact **disjoint 2×2 block sampler**:
   786,432 of 1,017,336 camera pixels are read exactly once (**77.3031%**) and **230,904
   (22.6969%) are invisible to both SegNet and PoseNet**. This reproduces ll1's measured
   read-histogram to the pixel from pure geometry. R cannot supply error, because it is a
   deterministic selection applied identically to our frames and to GT.

**Pointer UNMOVED.** hv1 ep0634 remains S 0.15959729295498598 @ 182,759 B [contest-CUDA T4].
This unit measured a mechanism and closed a family. It did not lower the score.

## §0.5 The mechanism ledger — the charter's question (a), answered

The charter asked me to decompose the 33,743 round-trip flips **by mechanism, not by location**.
Every candidate mechanism is now measured, and none of them is the answer:

| mechanism | probe | measured | verdict |
|---|---|---:|---|
| **M-A** uint8 quantization at camera res | ±1 LSB dither, n=24 | 56 fixed / 55 broken, Δflips −1 | **not the mechanism** — a fair coin on seg, and ra1 already bounded isolated quantization at 6.0% of the debt |
| **M-B** global sub-pixel misregistration | `shift:dy=0.25`, n=12 | +335 flips, broken/fixed 2.97 | **not the mechanism** — and ill-posed anyway, there is no registration to a field we do not reconstruct (§1) |
| **M-C** boundary blur / HF deficit vs GT | gradient ratio at ring 0, n=16 | ours is **5.95× SHARPER** | **refuted, and backwards** (§1) |
| **M-D** global photometric drift vs GT | `gain`, `gamma`, n=24 | luma mean off by +103, corr −0.48 | **ill-posed** — the fields are not comparable (§1) |
| **M-E** exact argmax ties | tie census, n=24 | **0 exact ties in 4,718,592 px** | **not the mechanism** |
| **R** the resize/uint8 operator | exact tap derivation | disjoint 2×2 selector, 22.6969% of camera px unread | **not a supplier** — common to both sides (§4) |
| **the actual mechanism** | rho + the scorer-weight arithmetic | rho(0.01) = 0.985; SegNet weights = S 25.64 | **the decoder cannot know the SIGN of its own error** (§2) |

The tie census also serves as a second positive control: median flip gap **0.1064** at n=24
against rt1's n600 median **0.1051** — a **1.2%** match on an independently coded instrument.

## §0 Prior-law prediction lines — stated BEFORE the measurements

Per the anti-re-anchor law, and because two of these were wrong.

1. **Charter premise (from rt1 §2.6): our render erodes Lane by 1,773 px, the signature of
   blur.** PREDICTION: our decode will read blurrier than GT at the boundary, and an unsharp
   mask will be the zero-byte cure. **WRONG, and backwards** — measured 5.95× *sharper*
   (§1). The erosion is real; blur is not its cause.
2. **rt1 §2.5b: R supplies exactly zero, scoped to flat paint.** PREDICTION: an independent
   geometric derivation will show R is a selector and confirm zero supply. **HELD** (§4).
3. **rt1 §2.8: "the residual is a TIE, not a wall … the most favourable shape a residual can
   have for a cheap cure."** PREDICTION: I will find the other side of that trade, and it will
   be at least as populous. **HELD, decisively** — 78:1 by count (§2).
4. **Recall (ll1, L28, odd_luma_bias, AA-on-witness): every global decode-side transform ever
   measured in this repo raised S, on three vehicles.** PREDICTION: the same sign at hv1's
   14×-lower d_seg, with rho as the coefficient that explains all of them at once. **HELD**
   (§3), with the refinement that pose, not seg, does the killing.
5. **My own pre-registered race bar, written before the race: "every undirected operator will
   show broken/fixed > 1 and none will reduce flips."** **FALSIFIED on the seg leg** by
   `gain:g=1.02` (34 fixed / 27 broken) and `dither:amp=1` (56 / 55). The correct statement is
   the one rho actually implies and which I had stated too strongly: at small delta the trade is
   a **fair coin**, so the seg sign is zero-mean and flips between samples. It did so: `gain
   1.02` measured 19/21 at n=12 and 34/27 at n=24. §3.3 records this against me.

## §1 hv1 is a task-space witness — the measurement that voids the charter's framing

Receipt `RN1_SPECTRUM.json`. n=16 seeded-random pairs (seed 20260816), GT decoded ONLY via
`frame_utils.yuv420_to_rgb`, both fields pushed through the scorer's own
`SegNet.preprocess_input` so the comparison is in the lattice SegNet actually reads.

| quantity | measured |
|---|---:|
| luma correlation ours vs GT, scorer lattice | **−0.4797** |
| luma correlation ours vs GT, camera resolution | **−0.4804** |
| luma mean, ours − GT | **+103.49** |
| luma std ratio ours / GT | 2.838 |
| RMSE ours vs GT in the scorer lattice | **133.70** (of 255) |
| gradient energy ratio ours / GT, whole frame | **2.218** |
| **gradient energy ratio ours / GT, label-boundary ring 0** | **5.947** |
| d_seg of the same field vs the same GT | 2.96e-04 (99.97% argmax agreement) |

**Positive control, paid before any of this was claimed.** My GT decode reproduces the cached
`gt_argmax_n600.npy` to **18–32 pixels of 196,608** on pairs 0, 1 and 5, so the decode path is
the canonical one and the indexing is right. Cross-check on brightness: PyAV's own `rgb24`
conversion gives GT per-channel means [31.5, 23.1, 20.2] against `yuv420_to_rgb`'s
[33.3, 24.1, 21.5] — the source video is genuinely dark, and our decode's [100.3, 132.3, 158.4]
is genuinely not a picture of it.

**Reading.** A video reconstruction correlates ~+1 with GT. Ours correlates **−0.48** and still
reproduces the frozen head's argmax to 99.97%. That is the campaign's own non-RGB task-space
witness thesis, confirmed on the live frontier vehicle for the first time by direct
measurement. The consequence is immediate and it is what killed my opening hypothesis: **any
lever justified by "make our frames more like GT" has no purchase here**, because our frames
were never trying to be like GT. This also retires, for this vehicle, the reading that our
render is a low-pass of the scene — at the boundary it carries six times GT's gradient energy.

⚠ **Scope.** n=16 seeded-random pairs, one vehicle. The correlation is stable pair to pair
(every pair negative), and the sign is not a close call, so I treat the qualitative verdict as
settled and the coefficients as n=16 estimates.

## §2 The exchange rate — the number rt1 did not take

rt1 measured the logit deficit **at the flips** (median 0.105, wanted class already runner-up
98.3% of the time) and read it as "a tie, not a wall … the most favourable shape a residual can
have for a cheap cure." That is one side of a two-sided trade. The other side is: **how many
CORRECT boundary pixels sit at an equally thin margin and would break under the same
perturbation?**

Definition. `gap = top1 − top2` SegNet logit. At a flip it is rt1's deficit — how far a cure
must move the head. At a correct pixel it is the **headroom before that pixel breaks**.

Receipt `RN1_EXCHANGE_n96.json`. n=96 seeded-random pairs; 5,448 flips, 408,678 ring-0 pixels,
403,254 of them correct.

| delta | flips below | correct ring-0 below | **rho (ring-0)** | rho (whole frame) |
|---|---:|---:|---:|---:|
| 0.01 | 404 | 398 | **0.985** | 0.988 |
| 0.03 | 1,070 | 1,320 | **1.234** | 1.242 |
| 0.10 | 2,706 | 5,785 | **2.138** | 2.147 |
| 0.30 | 4,662 | 32,906 | **7.058** | 7.111 |
| 1.00 | 5,431 | 212,166 | 39.07 | 41.72 |
| 3.00 | 5,448 | 403,184 | 74.01 | 161.10 |

Median gap at flips **0.099**; median gap at correct ring-0 pixels **0.970**. Individually a
correct boundary pixel is ~10× safer than a flip is wrong. **Collectively it does not matter,
because there are 74× more of them** (403,254 vs 5,448), and the lower tail of the correct
distribution reaches straight down into the flip range.

**The law.** An undirected perturbation of scale delta helps about half the flips it reaches and
breaks about half the correct pixels it reaches, so its expected effect is
`0.5 · cum_flip(delta) · (rho(delta) − 1)`. It is a net win only where `rho < 1`.

**rho(0 +) = 1, and that is not an accident.** The margin field is continuous across the
decision boundary, so the density of just-wrong pixels equals the density of just-right pixels
in the limit. Measured 0.985 at delta = 0.01 — a tie within Poisson error on 404 and 398
counts. **An undirected operator at the boundary is a fair coin, by symmetry.** A fair coin
cannot reduce an error count. And buying reach makes it worse, not better: the same table that
gives the fair coin at 7.4% reach gives 2.14 at 49.7% reach.

This is the single coefficient that explains, at once, ll1's window solve (+0.00039 seg),
L28's channel offset (+5.9e-8), `odd_luma_bias` +-1 (both worse), AA-on-witness (+49%), and
every row in §3 — without any of them needing its own story.

## §3 The race — 14 operator rows, all zero-byte or near-zero-byte

Tool `experiments/ddm_rn1_render_boundary_mechanism.py`, stage `race`. Same instrument pins as
rt1 (frozen CPU-torch SegNet, batch = 1 pair, upstream preprocess verbatim), so these rows are
leg-to-leg comparable with rt1's ledger. Every operator is a **generic algorithm** — free in
inflate.py under rule 118 — carrying only 0–24 B of tuned scalars, which are counted honestly
below and are negligible (24 B = 1.6e-5 S).

### §3.1 Seg screen, n=12 seeded-random (`RN1_RACE_screen_n12.json`)

| operator | flips | fixed | broken | broken/fixed | Δflips |
|---|---:|---:|---:|---:|---:|
| identity | 631 | 0 | 0 | — | 0 |
| `gain:g=1.02` | 633 | 19 | 21 | **1.11** | +2 |
| `gain:g=1.05` | 636 | 39 | 44 | 1.13 | +5 |
| `gamma:gamma=1.02` | 636 | 13 | 18 | 1.38 | +5 |
| `dither:amp=1` | 644 | 27 | 40 | 1.48 | +13 |
| `dither:amp=2` | 654 | 50 | 73 | 1.46 | +23 |
| `unsharp:sigma=2.3,amount=0.3` | 675 | 64 | 108 | 1.69 | +44 |
| `unsharp:sigma=1.0,amount=0.5` | 675 | 56 | 100 | 1.79 | +44 |
| `blur:sigma=1.0` | 837 | 164 | 370 | 2.26 | +206 |
| `shift:dy=0.25` | 966 | 170 | 505 | **2.97** | +335 |

The ordering is monotone in perturbation strength, exactly as rho requires: the gentlest
operator sits at the fair coin, the harshest at 2.97.

**A one-parameter test of rho's shape.** For each operator, infer its effective delta from
`fixed = 0.5 · cum_flip(delta)`, then *predict* `broken = 0.5 · cum_corr_ring0(delta)` from the
n=96 table and compare:

| operator | fixed | implied delta | predicted broken | measured broken |
|---|---:|---:|---:|---:|
| `gain:g=1.05` | 39 | 0.017 | 43.9 | **44** |
| `blur:sigma=1.0` | 164 | 0.097 | 348.8 | **370** |
| `gain:g=1.02` | 19 | 0.008 | 18.7 | 21 |
| `unsharp:sigma=2.3,amount=0.3` | 64 | 0.029 | 78.8 | 108 |
| `shift:dy=0.25` | 170 | 0.100 | 365 | 505 |

One free parameter per row predicts the second number to within 10% on the smooth photometric
operators. It **under**-predicts on the spatially structured ones (`unsharp`, `shift`) because
rho only counts ring-0 pixels at risk and those operators also break pixels off the ring. So
rho is a **lower bound** on the damage, and every measured operator lands at or above it.

### §3.2 Joint seg+pose, n=24 seeded-random (`RN1_RACE_joint_n24.json`)

Per the no-axis-priority law, no operator gets a verdict on seg alone. d_pose is aggregated in
the **scorer's convention** — the mean of d_pose itself, never a mean of per-pair ratios (rt1
§6.2b: the two disagree in sign).

| operator | Δflips (24 pr) | fixed | broken | ΔS_seg (n600 eq) | d_pose × | ΔS_pose | **NET ΔS** |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity | 0 | 0 | 0 | 0 | 1.0000 | 0 | 0 |
| `dither:amp=1` | −1 | 56 | 55 | −0.000021 | 1.664 | +0.002406 | **+0.002385** |
| `gain:g=0.98` | +14 | 24 | 38 | +0.000297 | 3.224 | +0.006598 | **+0.006897** |
| `gamma:gamma=1.02` | +5 | 30 | 35 | +0.000106 | 4.271 | +0.008846 | **+0.008955** |
| `gain:g=1.02` | **−7** | 34 | 27 | **−0.000148** | **5.077** | **+0.010394** | **+0.010249** |

**The pose leg is 25–70× the seg leg on every row.** The operator that wins on seg
(`gain:g=1.02`) loses hardest overall. And the cheapest possible perturbation — a **one-LSB
dither** — is seg-neutral and still costs **+0.0024 S**, a quarter of the entire remaining gap.

**Why.** At hv1's operating point d_pose = S_pose²/10 = **6.880e-06**, so the pose marginal is
`dS/d(d_pose) = 5/sqrt(10·d_pose) = 602.8` against seg's constant **100** — pose is **6.03×**
seg per unit of distortion. CLAUDE.md's operating-point section already predicted this flip and
measured it at 2.71× on PR106; hv1 sits further down the same curve. Structurally: seg's argmax
**quantizes away** almost every perturbation, so only the rho-governed sliver of boundary pixels
responds; pose is a **continuous regression on a near-zero residual**, so every pixel responds
and nothing is quantized away.

**The decoded frame field on this vehicle is effectively frozen.** It cannot be touched in any
direction, by any operator, for any reason — the pose term forbids it before the seg term has an
opinion.

### §3.2b The pose instrument carries an 18× offset — and it makes §3.2 the OPTIMISTIC bound

My advisory `d_pose_before` at n=24 was **1.83e-04** against hv1's n600 [contest-CUDA] d_pose of
**6.88e-06**. I did not let that pass. Candidate causes were an unlucky draw (per m96, pose
per-pair values span ~70×) or a genuine advisory-CPU vs contest-CUDA pose-axis gap. **I measured
it** (`RN1_RACE_posebase_n96.json`, n=96 identity):

| baseline | advisory d_pose | vs contest-CUDA n600 |
|---|---:|---:|
| n=24 | 1.830819e-04 | 26.6× |
| **n=96** | **1.251833e-04** | **18.2×** |
| contest-CUDA n600 | 6.880000e-06 | 1× |

It **did not converge** — quadrupling n moved it 32%, not 26×. So this is an **instrument axis
gap, not a subset draw**. The seg half of the same instrument is sound: 56.75 advisory flips per
pair at n=96 against 58.23 for the n600 base (**2.5%**), and rt1's n600 instrument check pinned
advisory-vs-CUDA seg at **0.021%**. The gap is pose-only. Most likely cause is the GT decode
path — `yuv420_to_rgb` is the CPU reimplementation of nvdec, and the contest CUDA axis decodes
through DALI — with CPU/CUDA scorer-forward drift as the other candidate. **I did not separate
them, and I do not need to.**

**What it does to §3.2, worked through.** Write our true pose error as `e` and the instrument's
extra decode offset as `d`, so my measured `d_pose = |e+d|² ≈ 18.2·|e|²`, i.e. `|d| ≈ 4.15·|e|`.
An operator adds `Δ`, uncorrelated with `d`. Then my measured ratio is
`1 + |Δ|²/|e+d|²` while the true ratio is `1 + |Δ|²/|e|²` — **the true ratio's excess over 1 is
~18× larger than the one I measured.** For `dither:amp=1`: measured excess 0.664 → true excess
≈ 12.1 → true ratio ≈ 13.1 → ΔS_pose ≈ **+0.0217**, against the +0.0024 in the table. The
independent absolute-Δd_pose reading gives **+0.0276** — same order, same direction.

**So the §3.2 pose column is the OPTIMISTIC bound, and the real loss is roughly an order of
magnitude worse.** The verdict does not merely survive the ambiguity; the ambiguity was hiding
damage, not inventing it. Anyone quoting the pose coefficients must quote them as an optimistic
bound with this 18.2× offset attached.

### §3.3 Where my own pre-registered bar was wrong

I pre-registered, before the race: *"every undirected 0-byte operator will show broken/fixed > 1
and none will reduce flips."* Two rows falsified it on the seg leg — `gain:g=1.02` at 34/27 and
`dither:amp=1` at 56/55.

That bar was **too strong, and rho is what says so**. rho(0.01) = 0.985 means the small-delta
trade is a fair coin, so the seg sign is zero-mean and must flip between samples. It did:
`gain:g=1.02` measured **19 fixed / 21 broken at n=12** and **34 / 27 at n=24** — opposite signs,
same operator, both consistent with rho = 1. The honest claim is not "every operator loses on
seg"; it is **"no undirected operator is a reliable seg supplier, because the trade is
symmetric"** — and separately, that every one of them loses badly on pose. I am recording the
correction rather than restating the bar to fit the result.

## §4 R is a selector, not a filter — the charter's question (c), settled by geometry

rt1 measured `S3 = 0` but scoped it to flat paint and wrote honestly that R "remains unmeasured
for the render." The question is answerable without any scorer at all, because R is pure
geometry.

`SegNet.preprocess_input` (`upstream/modules.py:107-109`) is `x[:, -1]` then
`F.interpolate(x, size=(384,512), mode='bilinear')`. **`antialias` is not passed**, so it
defaults False. Deriving the tap set for `align_corners=False` at scales 874/384 = 2.276042 and
1164/512 = 2.273438:

| quantity | derived |
|---|---:|
| camera rows ever read | 768 / 874 = 87.8719% |
| camera columns ever read | 1024 / 1164 = 87.9725% |
| **camera pixels ever read** | **786,432 / 1,017,336 = 77.3031%** |
| **camera pixels invisible to BOTH scorers** | **230,904 = 22.6969%** |
| taps per scorer pixel | 786,432 / 196,608 = exactly **4** |

786,432 = 196,608 × 4 exactly, so the windows are **disjoint**: every scorer pixel owns a
private 2×2 camera window, and 22.70% of camera pixels belong to no window at all. This
reproduces ll1's measured read-histogram (`0×: 230,904 px (22.70%)`, `1×: 786,432 px (77.30%)`)
to the pixel, from geometry alone and by an independent route.

**Consequence.** R does not blur, mix or displace; it **selects**. It therefore cannot be a
supplier of error, because it is a deterministic selection applied identically to our frames and
to GT. rt1's `S3 = 0` is confirmed and its scope caveat can be lifted: R supplies zero on *any*
content, textured or flat, because the operator is common to both sides of the difference.
Independently, the measured R MTF (1.00 to 16 cyc/unit, 0.997 at the dash scale) says the same
thing from the frequency side and already pre-killed R-deconvolution.

## §5 What is left — the training-side margin hinge, quantified

Only training can supply the sign, because only training has GT. A **signed** margin shift of
+delta at ring-0 pixels — toward the correct class — recovers every flip whose deficit is below
delta and **breaks nothing**, because the identical shift moves correct pixels further from the
boundary. This is precisely the asymmetry the decoder cannot have.

From the n=96 deficit ladder:

| signed margin hinge | share of the seg axis recovered | n600 flips | S units | **× the remaining gap** |
|---|---:|---:|---:|---:|
| +0.01 logits | 7.4% | 2,591 | 0.002196 | 0.23× |
| +0.03 logits | 19.6% | 6,862 | 0.005817 | 0.61× |
| **+0.1 logits** | **49.7%** | **17,354** | **0.014711** | **1.53×** |
| +0.3 logits | 85.6% | 29,897 | 0.025344 | 2.64× |
| +1.0 logits | 99.7% | 34,829 | 0.029525 | 3.08× |

Per GT class, share of that class's flips recoverable at +0.1 / +0.3 logits (canonical comma10k
order; flips at n=96):

| class | flips | +0.1 | +0.3 |
|---|---:|---:|---:|
| 0 Road | 2,172 | 52.9% | 87.5% |
| 1 Lane | 1,376 | 48.1% | 86.1% |
| 2 Undrivable | 931 | 52.3% | 87.4% |
| 3 Movable | 758 | 45.3% | 83.0% |
| 4 MyCar | 211 | **30.3%** | **63.5%** |

Road, Lane and Undrivable are equally cheap to buy; **MyCar is the expensive class** — its flips
sit at roughly twice the deficit of the others, so a uniform hinge buys it last. That is a
useful scheduling fact for the objective, and it is new.

⚠ **This is DERIVED and it is an UPPER BOUND, not a prediction.** A real trained model produces
a margin *distribution*, not a uniform shift; the hinge costs capacity and may trade against
pose and rate. The table bounds the prize and gives the objective a coefficient; it does not
promise the prize. Anyone citing it must carry that label.

**Routing.** This is rt1's follow-on #3 (edge-weighted objective into the wd3 / ns1-P1 line),
now with a target: **a ring-0 signed-margin hinge, +0.1 logits, worth 1.53× the remaining gap**,
scheduled Road/Lane/Undrivable first and MyCar last. It is not a new arm.

## §6 What this unit did NOT establish

- **No score.** Every number is `[macOS-CPU advisory]`. The pointer is unmoved and this unit was
  not permitted to move it.
- **The CAUSE of the 18.2× advisory pose offset is not separated** (§3.2b). I established that
  it is an instrument axis gap and not a subset draw, and I established its direction — it makes
  my pose column optimistic. I did not separate GT-decode-path drift from CPU/CUDA
  scorer-forward drift. That separation is a real owed measurement for anyone who needs advisory
  pose *coefficients* rather than advisory pose *signs*. It does not touch the seg axis, which
  is validated to 2.5% at n=96 and to 0.021% at n600 by rt1.
- **rho is measured at n=96 seeded-random pairs, not n600.** Per m96 a seeded subset may refute
  a bar but may not license a LIVE verdict. rho refutes the undirected family; it does not
  characterize the margin distribution across the full 600.
- **No n600 `leg` row was taken.** The `leg` stage is built and pinned to rt1's instrument, but
  no operator earned one: the best joint candidate loses by 25× on pose at n=24, so spending
  6 minutes of n600 SegNet on it would be measuring a settled loss.
- **The operator space is not exhausted.** I raced photometric (gain, gamma, dither), spatial
  (blur, unsharp, shift) families. Untested: per-class or spatially adaptive operators — but
  those need to know *where*, which is the sign problem again, and rho applies to them
  unchanged.
- **No claim that the witness finding transfers to other vehicles.** It is measured on hv1
  ep0634 at n=16.

## §7 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_rn1_render_boundary_20260816/` (APDataStore: 240 GiB free;
VertigoDataTier is at 893 MiB).

| artifact | what it is |
|---|---|
| `RN1_SPECTRUM.json` | the witness measurement, per-pair rows retained |
| `RN1_EXCHANGE_n12.json` · `RN1_EXCHANGE_n96.json` | the rho ladder + per-class deficits + the hinge bound |
| `RN1_RACE_screen_n12.json` | the 10-operator seg screen |
| `RN1_RACE_joint_n24.json` | the 5-operator joint seg+pose race |
| `RN1_RACE_posebase_n96.json` | the owed pose-baseline row |
| `exchange_n96.log` · `race_joint_n24.log` · `posebase_n96.log` | run logs |

The `leg` stage retains the full n600 argmax field with sha256 + bytes when it is used; no leg
was run, so no field is claimed. Consumed unmodified: the wc1 retained decode `0.raw`
(3,662,409,600 B, sha `e5539653…`, custody verified in-tool), the hv1 ep0634
`decoded_spatial_tokens.rc64.bin`, the qs3 `gt_argmax_n600.npy`, and `upstream/videos/0.mkv`.

Tool: `experiments/ddm_rn1_render_boundary_mechanism.py`
(stages `spectrum` / `ties` / `exchange` / `race` / `leg`).
