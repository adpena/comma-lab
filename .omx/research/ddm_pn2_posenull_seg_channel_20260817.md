---
arm: ddm_pn2
title: "the pose tax that killed every seg-edit candidate this campaign is REMOVED inside PoseNet's null space -- matched A/B over the COMPLETE pre-registered sample (n=12, same pairs/solver/support/seed, only the projection differs): unprojected d_pose x4.6089 vs projected x0.7935, worth 0.010423 S = 1.09x the whole remaining gap -- and the seg leg does not pay for it (pooled eta 0.5651 -> 0.6111, +8.1%, 10 of 12 pairs, exact sign p=0.0386), with the 2x2-snap confound isolated to COMPLETION (n=12) and measured running the OTHER way (snap -0.0286 eta vs projection +0.0746, so the bare A/B understated the projection; pre-registered FO-2 CLOSED); but the channel still does not deliver, because the binding constraint moves from POSE to the seg x rate arithmetic where sr1's waterfill is only a marginal supplier (-0.000526 S, 5.5% of the gap) on IDEAL-entropy bytes no real coder has yet priced (18.5% headroom); separately rt2's DERIVED x1.004 small-support pose leak is accidentally right at one alpha and wrong by 10.4-24.4x at its neighbours, and its alpha=1.0 row is an estimator artifact in which one pair carries 710% of the aggregate excess"
utc: 2026-08-17
charter: "operator/MAIN charter to ddm_pn2, 2026-08-17"
axis: "[macOS-CPU advisory] frozen CPU-torch SegNet + PoseNet -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "INSTANCE on the hv1 ep0634 vehicle at the measured n; mechanism verdicts only where named"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_pn2 — does realizing a seg edit INSIDE PoseNet's null space remove the pose tax?

STORES CONSULTED: rt2 `ddm_rt2_manufactured_seg_mechanism_20260817.md` (its §5 FO-A, read at source) ·
rt1 `ddm_rt1_seg_roundtrip_decomposition_20260816.md` §5/§6/§6.1/§6.2/§6.2b + its retained
`eta_gate_null` / `eta_gate_free` rows + `coder_race/RT1_CODER_RACE.json` · sr1
`ddm_sr1_manufactured_seg_recovery_20260816.md` §3.3/§4 + `SR1_WATERFILL.json` + its sister arm
`ddm_a1s_foa`'s `foa/FOA_VERDICT.json` · **Q3** `ddm_control_surface_exact_quartering_20260731.md`
+ memory `[[control-surface-exact-dof-quartering-q3-seg-only-pose-null]]` · sq1
`ddm_sq1_pose_null_constrained_paint.py::pose_null_projector` read at source · rn1 · `upstream/
frame_utils.py:51-76` and `upstream/modules.py:73,108-109` read at source · memories [[m88]]
[[m96]] [[m91]].

## ANSWER FIRST

1. **The pose tax is REMOVED, and it is the largest single number this unit produced.** Matched
   A/B — same seeded-random pairs, same solver, same support, same seed, only the projection
   differs — the unprojected solve costs `d_pose` **×4.6089** and the pose-null-projected solve
   costs **×0.7935**, over the **complete pre-registered sample (n=12, both arms finished)**.
   Converted to contest S against hv1's own n600 pose base, that is **+0.009516 S vs −0.000906 S:
   the projection removes 0.010423 S of pose cost = 1.09× the entire remaining −0.0095973 gap.** Every seg-edit candidate this campaign refused died on
   exactly this term (rt2's de-blur +0.2475 S, sf1 +0.0622 S, rt2 α=0.25 +0.0241 S, qs4's
   +2.396e-4 d_pose). **Inside the null space that blocker is gone.**
2. **The seg leg does not pay for it — η RISES.** Pooled η **null 0.6111 vs free 0.5651, +8.1%**,
   and the projection raised η on **10 of 12** matched pairs — **exact sign test p = 0.0386**. A
   constraint that makes the objective *better* is the opposite of the expected trade.
   **The confound I found in my own design (§4) is measured and it does not carry the result:**
   `null` mode also snaps the edit support to whole 2×2 blocks (snap_tax **1.77×**), so the bare
   A/B varies projection and support size together — but the isolating control (`--snap-support`,
   built and run to **completion, 12/12**) measures the snap alone at **−0.0286 η** — it *hurts* —
   against the projection's **+0.0746** at matched support. **The projection carries 162% of the
   observed gain**, i.e. the bare A/B *understated* it; on pose the snap alone is **1.57× worse**
   while the projection is worth **9.1×**. This closes the pre-registered FO-2.
3. **The channel still does not deliver the gap.** Removing pose moves the binding constraint, it
   does not close the arithmetic. At the measured projected η, rt1's describe-everything channel
   is a **NON-SUPPLIER (+0.004171 S pose-neutral)** and sr1's waterfill is a **marginal SUPPLIER
   (−0.000526 S pose-neutral, 5.5% of the gap)** — on bytes that are an **ideal
   conditional-entropy ceiling, not a real coder**, with only **18.5% real-coder headroom**. The
   pose blocker is retired; the seg×rate blocker is not.
4. **FO-A, adjudicated: rt2's DERIVED ×1.004 is accidentally right and mechanically wrong.** The
   energy argument ("leak ∝ support fraction") predicts the measured ring-0 ratio to **0.7×** at
   α=1.0 but misses by **24.4×** at α=0.5 and **10.4×** at α=2.0. Independently corroborated:
   sr1's sister arm `ddm_a1s_foa` measured band-restriction retaining **82.5%** of the full
   actuator's pose drift, and my α=0.5 row retains **79.0%** — not the 3.2% the energy model
   demands. **No downstream claim may cite ×1.004 as a law.**
5. **Two design errors of my own, caught and recorded** (§0, §4) — plus a P0 payload defect in the
   tool I inherited and fixed (§1).

**Net: the campaign's #1 named blocker is retired on measured evidence, and the object it unblocks
is a marginal supplier on unpriced bytes. Pointer UNMOVED.**

## §0 Prediction lines, and the honest provenance of each

Per the anti-re-anchor law, written before the corresponding measurement — and I distinguish a
blind prediction from a direction I already knew from retained data.

1. **NOT a blind prediction — declared.** The direction of the η result was already visible in
   RETAINED data before I ran anything: rt1's `eta_gate_free` holds exactly one row (pair 33,
   η +0.5714, `d_pose` ×396.157) against `eta_gate_null`'s matched pair 33 (η +0.6531, ×1.582),
   with identical `n_described_ring0=49`, `support_px=499`, `d_pose_before` and `snap_tax`. rt1's
   memo never reported this comparison. My run tests whether an n=1 retained coincidence survives
   at n>1; it is not a discovery of direction.
2. **PREDICTED: FO-A would REFUTE ×1.004**, because confining a de-blur to a thin support forces
   large camera-space excursions (deconvolution amplification, sr1's κ 2.34) which hit the
   [0,255] box and clip, and clipping — not support area — is what breaks the projection.
   **HELD on the mechanism, and the mechanism is the finding**: my n=2 smoke showed
   `realized_max_abs_err_vs_target` **28.3** with `frac_out_of_box_pre_clip` 5.8e-4 on the
   support-confined edit, against rt1's solved paint which realizes at **exactly 0.0** on every
   pair. **PARTLY WRONG on the number**: the aggregate ratio at α=1.0 lands at ×1.0044, inside
   rt2's "DERIVED holds" band, for a reason that is an estimator artifact (§2).
3. **PREDICTED: the projection's η advantage would shrink as n grows** (regression to the mean,
   the way rt1's pose payback went ×0.431 → ×0.713 → ×0.794 across n=6/9/12). **HELD.** The η
   advantage fell +15.6% (n=4) → +14.2% (n=7) → +11.7% (n=10) → **+8.1% (n=12, complete)**, and the
   pose gap-multiple wandered 1.65× → 1.53× → 0.81× → **1.09×** — noisy, not monotone, because the
   aggregate is a ratio of means over a heavy-tailed per-pair distribution. The **direction** never
   moved and the sign test stayed significant (**10/12, p=0.0386**). Recording this because it is
   the reason the headline is a direction and an order of magnitude, never a level.
4. **PREDICTED (rt2 §7's own caveat, inherited): ratios transfer across the advisory pose
   instrument's offset, absolutes do not.** **HELD and quantified**: rt1's 12 pairs carry mean
   `d_pose` 1.30107e-04 against hv1's contest n600 aggregate 6.885643e-06 — a factor **18.90**,
   which independently reproduces rn1's measured ~18.2× instrument discrepancy. Every pose number
   below is carried as a RATIO and converted once, explicitly (§1).

## §1 Instrument, controls, and a P0 defect I fixed in the tool I inherited

Reference form imported, never reimplemented: rt2's `Deblur` preimage + `project_pose_null`;
rt1's `ddm_rt1_eta_gate_pose_constrained` solver, which itself carries sq1's `pose_null_projector`,
`project_null`, `snap_band_to_blocks` and `realize_scorer_paint_to_camera`. Frozen CPU-torch
SegNet + PoseNet, batch-1, upstream preprocess verbatim, GT decoded only through
`frame_utils.yuv420_to_rgb`.

| control | measured | verdict |
|---|---|---|
| de-blur ladder `α = 0`, every arm (n=2, n=24 ×2) | **0 flips changed, 0 fixed, 0 broken, `d_pose` ratio exactly 1.0000** | **PASS** |
| free-mode solver reproduces rt1's retained pair-33 row | η **+0.5714**, 49→21, `d_pose` **×396.157** — identical to the retained row | **EXACT** — config matched and the solver is deterministic |
| pose-null nullity under support masking, n=24 | max abs `dY` **6.96e-15**, max abs 2×2 block-mean **4.31e-15** | exact; asserted in-tool, fail-closed |
| projection confined to the snapped support | max abs leak outside support **0.0** (unit test) | **PASS** |
| my `snap_to_blocks` vs sq1's `snap_band_to_blocks` convention | both grow to whole 2×2 blocks | matched |
| two independent derivations of the 6-of-12 kernel | rt2's `project_pose_null` (analytic) vs sq1's `pose_null_projector` (`P = I − pinv(A)A`, rank 6) | agree |
| `f1` vs `both` frame convention, identical seg | Δflips, fixed, broken **identical** (61/62 at α=0.5, 114/113 at α=1.0) | consistent with `modules.py:108` `x[:, -1]` — SegNet reads only frame_1 |

**Spurious-warning note.** `delta @ K_YUV` raises `divide by zero` / `overflow` / `invalid value`
RuntimeWarnings from the macOS Accelerate BLAS matvec path — the same spurious warning sq1's
`pose_null_projector` docstring already documents. Verified NOT real: the field is finite
everywhere, and an `einsum` recomputation agrees with the matmul to 8e-15. No number is affected.

**⚠ P0 defect fixed, not worked around.** rt2's ladder computed a per-pair `d_pose` and then
discarded it into a mean, persisting only the aggregate — the measure-and-discard shape the
ALWAYS-KEEP-THE-PAYLOAD rule forbids at the typing moment. It is also what hid §2's finding: the
aggregate is a ratio of MEANS, so a handful of heavy pairs can dominate it. I added per-pair
retention and re-ran. **Without that fix I would have reported ×1.0044 as "confirms the DERIVED
×1.004".**

## §2 FO-A — measured, and adjudicated against its pre-registered bands

rt2's FO-A: the realized `d_pose` of a pose-null-projected, preimage-realized edit restricted to
rt1's retained ring-0 support (`free_band_mask.npy`, sha `649dd26f0843…` — **verified, matches
rt2's citation exactly**), n=24 seeded-random (seed 20260817 — the SAME 24 pairs as rt2's
full-frame ladder, so the only difference is the support). I built the one owed `--support` flag,
plus `--edit-frames {both,f1}` because SegNet reads only frame_1 and a seg channel has no reason
to touch frame_0.

Snapped support = **3.242%** of the scorer field (2.16% unsnapped, grown to whole 2×2 blocks).

| α | full-frame ratio | ring-0 ratio | full excess | ring-0 excess | ring-0/full | energy model predicts | miss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | ×1.0385 | ×1.0304 | +0.0385 | +0.0304 | **0.790** | ×1.0012 | **24.4×** |
| 1.0 | ×1.2052 | ×1.0044 | +0.2052 | +0.0044 | 0.021 | ×1.0067 | 0.7× |
| 2.0 | ×2.5103 | ×1.5079 | +1.5103 | +0.5079 | 0.336 | ×1.0490 | **10.4×** |

**Verdict, split between letter and substance.**

- **By the letter of the pre-registered bands** (which were written on the aggregate ratio): α=1.0
  gives ×1.0044, inside the "≤×1.01 → the DERIVED ×1.004 holds" band. α=0.5 gives ×1.0304, which
  falls in a **gap the pre-registration does not cover** — the bands named ≤1.01 and >1.05 and
  said nothing about (1.01, 1.05]. I am recording that gap rather than resolving it in my favour.
- **By substance the mechanism is REFUTED.** rt2's ×1.004 comes from "the leak is a property of
  how many camera pixels the edit touches", i.e. leak energy ∝ support fraction. That model is
  right to 0.7× at exactly one α and wrong by **10.4–24.4×** at both neighbours. A model that
  survives at one point of a three-point ladder is a coincidence, not a law.
- **The α=1.0 row is an estimator artifact and I can show it.** With per-pair retention: at α=1.0
  the single largest-moving pair contributes **710.7%** of the total aggregate excess — the
  aggregate is a near-cancellation of large opposite-signed per-pair moves, not a small leak.
  Per-pair ratios at α=1.0 span **0.208 to 3.797** (median 1.0018). At α=0.5 the top pair is
  59.9% of the excess; at α=2.0, 20.9%. **At n=24 the aggregate ring-0 ratio is not resolved.**
- **Independent corroboration of the substance.** sr1's sister arm `ddm_a1s_foa` measured, on a
  different instrument (pose-drift rms rather than realized `d_pose`) and a different α (0.25),
  `band_share_of_full_drift` = **0.8248**. My α=0.5 row gives **0.790**. Two arms, two
  instruments, same answer: **band restriction retains ~80% of the pose drift while touching
  ~3% of the field.** The energy model demands ~3%.

**Mechanism, measured.** The support-confined de-blur does NOT realize its target: on ring-0
support it shows `realized_max_abs_err_vs_target` **28.69** and `frac_out_of_box_pre_clip`
**6.16e-04** at α=1.0, because a thin-support target forces large camera excursions
(`camera_max_delta` 74) that hit the [0,255] box. rt1's SOLVED paint, by contrast, realizes at
**exactly 0.0 max abs error on every pair** because it solves with the clamp in the loop and never
demands an out-of-box preimage. **The binding variable is excursion size and clipping, not support
area** — which is why the de-blur (undirected, large) leaks and the solved edit (directed, small)
does not. That reconciles rt2's leak finding with rt1's exact realization in one mechanism.

**Seg leg on the same object** (ring-0, projected): Δflips **+1 / −1 / +36** at α = 0.5 / 1.0 /
2.0, with `broken/fixed` **1.016 / 0.991 / 1.195**. The fair-coin signature rn1's law predicts,
on a sixth operator. The de-blur is not a seg supplier at ring-0 either.

## §3 The matched A/B — the seg reach of the pose-null subspace

**Design, and why it is this design.** Q3 (2026-07-31) already measured the same 6-of-12 kernel
from the other side — a frame_1 yuv6-null perturbation reaches SegNet at amplitude 6.0/255 while
PoseNet sees 5.7e-14, *exactly* zero, a linear kernel at any amplitude — and left two limits open:
(1) reaching SegNet's input is not moving its argmax; (2) realizability at camera-res uint8 was
UNMEASURED. rt2 closed (2). This section answers (1), and Q3's own law dictates the arm:

> *"a reader holding only the dimension count would design the Q3 probe ISOTROPIC — the generic
> control our own standing law forbids — and a null result would close a FAMILY off a rung-1
> design. Perturb along rung-3-ranked directions; isotropic is the CONTROL arm, never the
> treatment."*

So the **undirected de-blur of §2 is the CONTROL** and the **directed solver is the TREATMENT** —
run in both modes on the SAME pairs, which is the one thing neither rt1 nor rt2 ran (rt1's support
ladder was unconstrained, its gate was projected, on different pairs).

Config, matched exactly to rt1's retained `eta_gate_null` run: `steps=30, lr=6.0, eval_every=2,
focus_weight=500, radius=1, starts=2`, pairs by seeded random choice (seed 20260816, n=12, never a
prefix per m96).

| statistic | unprojected (`free`) | pose-null projected (`null`) | delta |
|---|---:|---:|---|
| pooled η, **all 12** matched pairs | **0.5651** | **0.6111** | **+0.0460 (+8.1%)** |
| pairs where the projection raised η | — | — | **10 of 12** (exact sign test **p = 0.0386**) |
| `d_pose` ratio, scorer convention | **×4.6089** | **×0.7935** | **5.8× less pose** |
| ΔS_pose against hv1's n600 pose term | **+0.009516** | **-0.000906** | **0.010423 S removed** |
| that, as a multiple of the whole remaining gap | — | — | **1.09×** |

This is the **complete pre-registered sample** — all 12 seeded-random pairs, both arms finished.
The projected pooled η here (0.6111) is *identical* to the full null run's own n=12 figure, as it
must be, which is a closure check on the join.

Per-pair Δη (null − free), **10 of 12 positive**: `33` +0.0816, `66` +0.0959, `81` +0.1220, `89` +0.0600, `280` -0.0217, `299` +0.0816, `322` +0.1562, `353` +0.0333, `410` +0.0213, `438` +0.0411, `474` -0.0847, `538` +0.0247.
Pairs 280, 474 are the two where the projection cost seg; they are recorded rather than
smoothed away.

**Reading.** The unprojected solve buys its seg flips and hands back **+0.009516 S** on pose —
**0.99× the whole remaining gap** — which is precisely why every prior seg-edit candidate was
refused. The projected solve fixes *more* flips and pays **-0.000906 S**. On this evidence the pose
tax is not reduced; it is gone.

**η trajectory as rows landed** (the regression-to-the-mean check of §0.3, which **HELD**): the
η advantage fell +15.6% (n=4) → +14.2% (n=7) → +11.7% (n=10) → **+8.1% (n=12)**, and the pose
gap-multiple wandered 1.65× → 1.53× → 0.81× → **1.09×** — noisy, not monotone, because the
aggregate is a ratio of means over a heavy-tailed per-pair distribution. **The direction never moved**
and the seg sign test stayed significant (10/12, **p = 0.0386**). Quote the direction and the
order of magnitude — the pose removal is *on the order of the whole remaining gap* — never a level.

## §4 The confound in my own A/B, and the control I built for it

`null` mode does not only project — it also snaps the edit support to whole 2×2 blocks, because
two of the six pose constraints are BLOCK-mean conditions and masking a projected delta at pixel
granularity re-introduces a block mean. Measured `snap_tax` = **1.7715** on pair 33. So the bare
null-vs-free A/B varies **the projection and the support size together**, and rt1 §6.1 measured
that support size alone is worth ~0.65 η (r=0 → 0.52–0.54, r=1 → 0.62–0.69, r=2 → 0.32–0.55).

**Consequences as stated BEFORE the control ran, kept for provenance:** the η claim was
CONFOUNDED — a +14.2% η gain from a treatment that also grows the support 1.77× could not be
attributed to the projection. The pose claim was never confounded, and the confound ran *against*
it: the projected arm edits a **1.77× larger** support — more pixels changed, which should cost
*more* pose — and still pays 12.8× less.

**The control, built and run:** `--snap-support {true,false}` forces the 2×2 snap
independently of `--mode`, applied to BOTH the solve mask and the realize mask, defaulting to
`None` = legacy behaviour so no existing invocation changes. `--mode free --snap-support true`
isolates the projection.

**Measured, all 12 matched pairs — the control is COMPLETE** (12/12, both arms finished):

| arm | support | projection | pooled η | `d_pose` ratio |
|---|---|---|---:|---:|
| `free` | pixel | none | 0.5651 | ×4.609 |
| `free --snap-support true` | **2×2 blocks** | none | 0.5365 | ×7.243 |
| `null` | 2×2 blocks | **pose-null** | 0.6111 | ×0.7935 |

- η attributable to the **support snap alone**: **-0.0286** — it *hurts*
- η attributable to the **projection alone**, at matched support: **+0.0746**
- the projection therefore carries **162.1%** of the observed gain — more than all of it, because
  the snap it drags along is a drag
- pose: the snap alone makes it **1.57× WORSE** (×4.609 → ×7.243); the projection alone is worth **9.1×**

**The confound is real, it is larger than I first measured, and it runs entirely the OTHER WAY.**
Growing the support to whole 2×2 blocks costs **-0.0286 η** and makes pose **1.57× worse**;
the projection at matched support is worth **+0.0746 η** and **9.1×** on pose. So the bare
null-vs-free A/B **understated** the projection: the true projection effect is larger than the
confounded estimate, not smaller. **The η claim survives de-confounding with room to spare.**

⚠ **This resolves the pre-registered FO-2 (§7) in its second branch.** The bar was: *snapped-free η
at or above projected η → the gain is support growth; snapped-free η below projected η → the
projection genuinely improves seg.* Measured: snapped-free **0.5365** vs projected
**0.6111** — **below, by +0.0746**. FO-2 is **CLOSED: the projection genuinely
improves seg.** The mechanism — that a chroma-at-fixed-luma edit is a lower-collateral seg actuator
because it moves the argmax without disturbing the luminance structure SegNet's stride-2 stem keys
on — remains a HYPOTHESIS this unit did not test.

## §5 Joint arithmetic at n600 scale

Exchange rates, exact contest arithmetic, reused not re-derived: seg `ΔS/flip` = **8.477105e-07**,
rate `ΔS/byte` = **6.658590e-07**, so the seg market bar is **1.273108 B per scored flip**. Pose
converted once, explicitly, as `ΔS_pose = (√ratio − 1)·√(10·d_pose_n600)` with
`d_pose_n600 = 6.885643e-06` (hv1's contest-CUDA row) giving a pose contribution scale of
**0.0082980**. **This revises rt2's absolute `ΔS_pose` figures downward** — rt2 priced against the
advisory instrument's own 24-pair base (2.2557e-04), which is **32.8×** the contest n600
aggregate, so its absolute pose costs overstate the contest term by ≈√32.8 ≈ 5.7×. rt2's *ratios*
stand; its absolute pose S figures do not transfer.

Priced at the projected pooled η over **all 12 matched pairs, η = 0.6111** — which is also the full
null run's own n=12 figure, so there is no subset-vs-full ambiguity left — and the pose ratio ×0.7935.

| channel framing | pose leg | ΔS_seg | ΔS_rate | ΔS_pose | **net ΔS** | verdict |
|---|---|---:|---:|---:|---:|---|
| rt1 describe-everything (real M7 coder) | measured ×0.7935 | -0.017959 | +0.022130 | -0.000906 | **+0.003265** | NON-SUPPLIER |
| rt1 describe-everything (real M7 coder) | pose-neutral | -0.017959 | +0.022130 | +0.000000 | **+0.004171** | NON-SUPPLIER |
| sr1 waterfill (IDEAL entropy, unpriced) | measured ×0.7935 | -0.003374 | +0.002847 | -0.000906 | **-0.001433** | SUPPLIER |
| sr1 waterfill (IDEAL entropy, unpriced) | pose-neutral | -0.003374 | +0.002847 | +0.000000 | **-0.000526** | SUPPLIER |

- **rt1's describe-everything channel is a NON-SUPPLIER**, confirming rt1's CLOSED verdict on
  independent rows — and for a reason that has nothing to do with pose: even with the pose leg set
  to exactly zero it loses (**+0.004171 S**).
- **sr1's waterfill is a marginal SUPPLIER**: **-0.000526 S** pose-neutral
  (5.5% of the gap), **-0.001433 S** (14.9%) if the measured pose *gain* survived
  to n600 — which §9 says must not be assumed.
- η is **below rt1's 0.753 describe-everything bar** (0 of 12 null pairs above it) and **above sr1's
  0.3871 guarded waterfill supplier margin** (12 of 12 above it). Both prior verdicts reproduce
  exactly; my rows change neither.
- Break-even on the waterfilled support is **5066 B** against sr1's ideal **4276 B** —
  **18.5% real-coder headroom**.

## §6 The rate leg — priced honestly, and the one thing that is NOT priced

- **rt1's real coder is the only real coder measured on this object.** `RT1_CODER_RACE.json`, all
  payloads roundtrip-verified by decoding back through the same online context machine: best
  **M7 CABAC boundary-walk (88 contexts) = 32,270 B = 7.447 bits/flip** on the full band, against
  a static-AC i.i.d. realization of 33,087 B = 7.636 bits/flip. **A real context coder beat i.i.d.
  by 2.47%** on this object — not the 2× the pre-§5 speculation assumed.
- **sr1's waterfilled bytes are NOT a real coder.** `SR1_WATERFILL.json` is an *ideal conditional
  entropy* limit over 41 cells with a 148 B model cost — sr1 says so itself ("No real coder for
  the waterfilled support"). At 4,276 B for 6,512 described flips that is **5.253 bits per
  described flip**, well under M7's 7.447 because the waterfill keeps only the densest cells.
- **The headroom is measured, the coder is not.** At the measured projected η the waterfilled
  channel breaks even at **5,066 B** against sr1's ideal **4,276 B** — a real coder may be
  **18.5% worse than the ideal and still break even**. That is real headroom, and M7's 2.47%
  edge over i.i.d. on the denser support is plausibly enough to sit inside it. **Plausible is not
  measured.** Coding the waterfilled support with a real coder is the named owed row (§7 FO-1).
- **The projection does not change the rate leg.** The mask names which band pixels flip; whether
  the realization is projected or not costs the same bytes. My finding moves pose, not rate.

## §7 Sealed follow-ons — a PRICE and a CONTROL, no cure is fired

No cure cleared, so no cure is fired. No Modal. MAIN owns all paid fires.

**FO-1 — the real coder on the waterfilled support ($0, scorer-free, local desk).** The single
number sr1's supplier claim rests on and has never had. Reconstruct the waterfill's 41-cell
selection (`cell_definition` in `SR1_WATERFILL.json`: own class × lowest differing 4-neighbour
class × min(degree, max_degree) × row band — every factor a deterministic function of the
transmitted labels, so the receiver recomputes it free), restrict rt1's mask to it, and race the
same M0–M7 coders with roundtrip verification.
Pre-registered bar, written before the run: **real coded bytes ≤ 5,066 B at the measured η → the
waterfilled channel is a real supplier; > 5,066 B → sr1's −0.000595 S is an artifact of the ideal
entropy and rt1's CLOSED verdict is restored on better evidence than it had.**
Owner: a $0 desk arm.

**FO-2 — the snap-isolating control — RESOLVED, no longer owed.** `--mode free --snap-support true`,
same 12 pairs, **completed 12/12**. Pre-registered before the run and not amended: snapped-free η at
or above projected η → the gain is support growth; below → the projection genuinely improves seg.
**Measured: snapped-free 0.5365 vs projected 0.6111 — below by 0.0746. CLOSED in the second branch:
the projection genuinely improves seg**, and the support snap it drags along is a −0.0286 η drag
that made the bare A/B *understate* the effect. See §4. The open successor is the MECHANISM
(chroma-at-fixed-luma as a lower-collateral seg actuator), which this unit did not test.

**NOT queued, and why.** A larger-n rerun of the pose leg is the obvious next thought and is worth
less than it looks: the pose result is already ~14× with the confound running against it, and n600
of this solver is ~25 h of local CPU. The binding uncertainty is FO-1's bytes, not the pose leg.

## §8 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_pn2/` (APDataStore; VertigoDataTier has 893 MiB free and is
read-only for this arm).

| artifact | bytes | sha256 (prefix) |
|---|---:|---|
| `eta_gate_free_n12/ETA_GATE_ROWS.jsonl` | 6638 | `e1f784e08b8c…` |
| `eta_gate_free_n12/ETA_GATE_VERDICT.json` | 10176 | `228649e64dc4…` |
| `eta_gate_free_snapped_n12/ETA_GATE_ROWS.jsonl` | 6665 | `c6a0baeb675a…` |
| `eta_gate_free_snapped_n12/ETA_GATE_VERDICT.json` | 10213 | `768e7b2d33f9…` |
| `PN2_VERDICT_partial.json` | 5305 | `ad0f969b9478…` |
| `PN2_VERDICT.json` | 5426 | `7704e4802ddf…` |
| `RT2_DEBLUR_LADDER_n2_s20260817_posenull_supf1.json` | 4653 | `e51fc93c9e41…` |
| `RT2_DEBLUR_LADDER_n24_s20260817_posenull_supboth.json` | 6681 | `f751bb9f83b1…` |
| `RT2_DEBLUR_LADDER_n24_s20260817_posenull_supf1.json` | 33382 | `da5d1a77868b…` |

`PN2_VERDICT.json` is the machine-readable verdict (matched A/B + three-way decomposition +
joint arithmetic). The `RT2_DEBLUR_LADDER_*_sup*` receipts carry per-pair `d_pose` and flip rows,
not only aggregates. **Both eta-gate arms are COMPLETE at 12/12**: `eta_gate_free_n12` (the
unprojected A/B arm) and `eta_gate_free_snapped_n12` (the snap-isolating control).

Tools landed this unit (commit `44742c4bb1`): `experiments/ddm_rt2_deblur_ladder.py`
(`--support`, `--edit-frames`, per-pair retention, fail-closed nullity assertion) ·
`experiments/ddm_rt1_eta_gate_pose_constrained.py` (`--snap-support` confound isolation, receipt
provenance) · `experiments/ddm_pn2_posenull_seg_channel.py` (matched-A/B aggregation + joint
arithmetic; runs no scorer).

Consumed unmodified: rt1's `free_band_mask.npy` (sha `649dd26f0843…`, verified), rt1's
`eta_gate_null/ETA_GATE_ROWS.jsonl` and `argmax_base.npy`, rt2's
`RT2_DEBLUR_LADDER_n24_s20260817_posenull.json` (sha `df53598c4f4bf540…`), the wc1 retained
decode `0.raw`, the qs3 `gt_argmax_n600.npy`, and `upstream/videos/0.mkv` via
`frame_utils.yuv420_to_rgb`. **`upstream/` was read, never written.**

## §9 What this unit did NOT establish

- **No score, no pointer move.** Every number is `[macOS-CPU advisory]`, `score_claim=false`.
- **The η half of the seg-reach claim is CONFOUNDED** by the 1.77× support snap (§4). Only the
  pose half is clean. The isolating control (§4) measures the snap at −0.0286 η and the
  projection at +0.0746 at matched support — so the confound runs *against* the claim, not for it,
  and the bare A/B understated the projection. That decomposition is **COMPLETE at n=12**.
- **n=12 (solver) and n=24 (ladder) are SCOPE reductions**, seeded-random, never a prefix. Per m96
  a random subset may REFUTE a bar but may not license a LIVE verdict. No LIVE verdict is claimed.
  Per m96's sister finding, prefix bias on the pose axis runs 2.5–4.2× — which is *why* both arms
  are seeded-random.
- **The n=24 aggregate ring-0 pose ratio is not resolved** (§2): one pair carries 710% of the
  α=1.0 excess. The α=1.0 row must not be cited as a measurement of the leak.
- **The pose instrument's absolute level is not trusted** (18.90× vs the contest n600 base, §0.4).
  Every pose number is a RATIO against the same instrument's own base, converted once against
  hv1's contest pose term. A ratio survives a multiplicative offset; it is not proof the offset is
  purely multiplicative.
- **sr1's waterfilled bytes remain an IDEAL entropy ceiling.** The supplier verdict in §5 is
  conditional on FO-1, and I have not run FO-1.
- **No claim that the pose-null subspace is a seg SUPPLIER.** It is measured to remove the pose
  COST of a seg edit. Whether the seg edit pays for its own bytes is the seg×rate arithmetic of
  §5, and that arithmetic is marginal at best.
- **The 6-of-12 kernel is exact only pre-quantization.** Realized nullity is 7e-15 in float and
  then passes through uint8 and the [0,255] box; §2 measures what that costs for a large
  undirected edit and rt1's solver measures 0.0 realization error for a small directed one.
  Nothing here characterises the intermediate regime.
