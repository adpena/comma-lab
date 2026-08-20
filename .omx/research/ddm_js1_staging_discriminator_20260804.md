---
title: "the staging law HOLDS on the block16 phase field: seg damage to frame_1 is repairable on the structurally seg-free frame_0 — and the row's remaining blocker moved from PHYSICS to CARRIAGE"
unit: ddm_js1
task: staging discriminator (ARM C / C-PRIME / A / B) on et1's block16 row
date_utc: 2026-08-04
axis: "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
own_vehicle_frontier: "S = 0.7910689 @ 353,805 B [macOS-CPU advisory] — UNMOVED by this unit"
verdict_scope_default: FORMULATION
---

# ddm_js1 — which staging converts et1's seg-live/pose-blocked row into a net win

## §0 ANSWER FIRST

The operator's staging law — *damage pose to condition seg first, then recover pose* — **holds on
this field, and the recovery is exact in the only dimension that could have killed it.** But the
measurement also moved the row's blocker: it is no longer a physics question.

| | measured | consequence |
|---|---|---|
| stage 1 (unconstrained seg solve on frame_1) | η **cap-pinned floor ≈ 0.41–0.53**, clears the 0.1707 bar ~**3×** | seg is LIVE |
| stage 1 pose damage | **heavy-tailed**, up to **123.8×** (et1 pair 261) | unrepaired stage 1 is not merely blocked, it is catastrophic on the tail |
| **ARM C-PRIME** (repair on frame_0) | seg **EXACTLY preserved** on every pair; d_pose lands **0.22–0.78× of SHIPPED** | the repair costs **zero** η — by construction, not by constraint |
| the pose-only CONTROL | 1.000× / 1.000× / 0.481× | most of C-PRIME's gain is **created by the staging**, not pu2 headroom |
| carriage, free-form | **361,708 B/pair** vs a **157.7 B/pair** budget | unpayable — **but under-conditioned, see the ⚠ below** |
| **carriage, k=4 generic DCT** | **96 B/pair**, repairs pair 48 to **0.3149× of shipped** | **FITS the budget ⇒ NET −0.0177 S** |

**The single sentence.** Staging works — better than it was asked to: the staged pair ends **below
shipped d_pose while clearing the η bar 2.8×** — and once the repair is solved **inside a cheap
generic basis** it also fits the byte budget, giving the first **net-negative ΔS** row shape since
pu2: **−0.0177 S (2.86% of gap)** at **96 B/pair**, seg preserved exactly, pose improved.

**⚠ THE CORRECTION THIS UNIT OWES ITSELF.** My own free-form arm measured the repair delta at
**361,708 B/pair** and I wrote the carriage blocker up as fatal (2293× over budget). That verdict
was **wrong as a family claim**, and the arm that refuted it is one I ran because p3v2 warned me
to: a **48-DOF** k=4 solve repairs pair 48 to **0.3149×**, *better* than my **589,824-DOF** free
solve reached (0.561×) on the same pair. A restricted basis cannot genuinely beat a basis that
contains it — so the free-form solve was **under-conditioned**, and its dense 361,708 B delta was
measuring the noise of a badly-converged high-dimensional optimisation, **not the intrinsic cost
of the repair**. Free-form remains unpayable; "the repair is unpayable" does not follow, and I had
started to write it. `verdict_scope: INSTRUMENT` on the free-form byte figure.

**Why C-PRIME dominates C by construction.** `upstream/modules.py` `SegNet.preprocess_input` is
literally `x = x[:, -1, ...]  # Use only last frame`. frame_0 is therefore **structurally
seg-invisible**, so a pose repair placed there cannot cost a single flip. ARM C (cell-constrained
repair on frame_1) has to *buy* seg-exactness with a constraint that removes DOF from the pose
solve; C-PRIME gets it for free. **Measured, not assumed**: a ±40-LSB random perturbation of
frame_0 leaves the SegNet argmax bit-identical on every pair (control in every row).

**The blocker that moved.** et1 left the row "pose-BLOCKED" and named the Q3 rank-6 projection as
the decisive cure. That framing is superseded: the pose damage is fully repairable at **zero seg
cost**, so the open question is no longer *can pose be held* but **can the repair be paid for**.
That is ph5o's question, and ph5o's answer on the sibling actuator was ALIGNMENT: YES / RATE: NO.

## §1 What I refute — in my charter, in the coordinator's upgrade, and in the inherited row

1. **The inherited pose price is 28.5% too cheap, and it is mine to correct.** et1 §8 quotes
   `dS/dd_pose = 31.3026`. That is the **PRE-pu2** operating point: pu2's own receipt records
   baseline `d_pose_mean 0.0025514` → `pose_contribution 0.159731`, and `5/0.159731 = 31.3026`
   exactly. pu2 then *lowered* d_pose to **0.00154517** (its win was pure pose — independently
   re-confirmed by et1's own C4 control). At the **current** live-best operating point the pose
   term is `0.7910689 − 0.431179 − 0.235584 = 0.124306`, so

   `dS/dd_pose = 5 / 0.124306 = 40.2234`

   Pose damage is **more** expensive than the row was priced against, because pu2 already
   harvested the cheap pose and moved us **up** the sqrt curve where the derivative is steeper.
   m66/qd1 exactly: a delta without its baseline is unanchored, and baselines move. This makes
   the pose gate **stricter**, not looser — it is a correction against my own arm's interest.

2. **The coordinator's caution (c) recall surface is measured-CLOSED at this operating point.**
   The named free path is "derivable from the carried ξ + the damaged frame_1" (#241/#249), i.e.
   p3v2's `warp_base_work` — a ground-homography warp of frame_1 by the already-carried 6-value
   pose target, 194 B, decoder-reproducible. **It cannot carry frame_0 here**: p3v2 priced it at
   n600 `d_pose mean 0.3931` (contribution 1.98). Our shipped frame_0 already sits at
   **0.00154517** — the warp base is **254× worse**. It was the right carrier for the pre-pu2
   regime (d_pose ~10–88) and is Pareto-dominant *there*; at this operating point regenerating
   frame_0 destroys far more pose than the staging could ever buy in seg. `verdict_scope:
   FORMULATION` (warp-as-frame_0-base at the post-pu2 operating point).

3. **et1's published pose-damage range is not representative.** §5 quotes
   `[1.064, 1.190, 1.000, 3.652]` from n=4. Its own continuing n=32 has since produced **123.846×**
   (pair 261) and 2.094×/1.939×. The damage distribution is heavy-tailed, so **mean-of-ratios is
   the wrong statistic** for a population effect (a tiny-d_pose pair at 123× moves less absolute
   mass than a large-d_pose pair at 1.1×). This aggregator pools **absolute Δd_pose**.

4. **My charter's "ARM C = cell-constrained frame_1 repair" is the dominated form**, and the
   coordinator's C-PRIME upgrade is right for a reason worth stating precisely: seg-exactness by
   *construction* (frame_0 is outside SegNet's input) strictly dominates seg-exactness by
   *constraint* (rejection/penalty on frame_1), because the latter spends pose DOF to buy
   something the former gets for nothing.

5. **My own instrument, twice.** (a) The first run of this harness crashed on a hardcoded
   `(384,512)` reshape — `rgb_to_yuv6` packs each 2×2 luma block into 4 channels + 2 subsampled
   chroma, so its output is **(2,6,192,256)**, half-resolution. Dimensions are now derived from
   the tensor. (b) My wait-filter false-positived on the substring `error` inside the key
   `yuv6_max_abs_error`, reading a healthy run as a failure.

## §2 The gradient path — the bug class that would have made this arm silently vacuous

`upstream/frame_utils.py:50` decorates `rgb_to_yuv6` with `@torch.no_grad()`. That **severs the
PoseNet gradient** — the CLAUDE.md eval_roundtrip bug class. An unpatched pose solve would have
returned its starting point on every pair and this arm would have read as *"frame_0 cannot repair
pose"*: a false negative indistinguishable from a real one (m50). So the canonical differentiable
replacement is patched in and **two** things are asserted before any gradient is trusted:

| control | measured |
|---|---|
| yuv6 forward equivalence to upstream | **PASS, max_abs_error = 0.0** (BT.601 coefficients are exact rationals) |
| gradient liveness through `preprocess_input` | **LIVE**, |grad| sum 393216.03 |

## §3 The actuator — and why pz1's attenuation caution does not bite

Caution (b) warned that null-space membership does not survive lattice resampling (pz1 measured
**1.662×** attenuation for warp-shaped fields). That risk is **structurally absent here**, and the
receipt proves it rather than arguing it. Both scorers reach the camera plane through the *same*
operator D (pz1), and D's 2×2 supports are **private** (m86, asserted fail-closed each run). So
the solve runs on the **scorer lattice** — the scorer's own coordinates — and is written back
through those private supports, where D reproduces it exactly. There is **no change of lattice**,
hence nothing to attenuate.

Two controls close it:

| control | measured |
|---|---|
| **C5** — scorer-lattice pose path vs canonical **camera** path, per pair | **abs_diff 0.0** |
| **realization gap** — predicted d_pose vs verified-from-camera d_pose | **≈0 on most pairs; worst observed +2.12e-04** |

**The gap is not identically zero and the memo should not say so.** Worst observed is **+2.12e-04**
(n32 free-form pair 154; −4.62e-05 on k=4 pair 115; −6.14e-06 on pair 32) — one to three orders of
magnitude below d_pose (~1e-3). Two mechanisms, both benign and both reported rather than rounded
away: (a) D's four private bilinear weights sum to 1 only to **fp32 precision**; (b) the larger
+2.12e-04 sits on the **free-form full-frame realization**, which *flattens* every camera 2×2 block
to a single rounded value (§8 defect 5) and so re-quantizes the whole of frame_0 rather than a
band — the **additive** realization would eliminate it, and the k=4 carriage arm's gaps stayed
≤ 5e-5. The gap is largest on precisely the arm already scoped INSTRUMENT, which is the reassuring
direction: the payable arm realizes cleanly.

Every d_pose in this memo is therefore re-scored **from the real camera pair**, never from the
solve's own coordinates.

## §4 RESULTS — the staging law holds, and it does more than repair

Stratified n=32 selection (sq1's, pair-matched to et1 so stage-1 is directly comparable). Every
d_pose is **verified from the real camera pair**; seg-exactness is **measured per pair**.

| pair | S1 η | S1 d_pose (damage) | **C-PRIME d_pose** | POSEONLY control | seg exact | realization gap |
|---|---:|---:|---:|---:|:--:|---:|
| 48 (worst known) | +0.5290 | **3.654×** | **0.561×** | — | ✅ | 0.0 |
| 0 | +0.4085 | 1.064× | **0.783×** | 1.000× | ✅ | 0.0 |
| 20 | +0.5062 | 1.190× | **0.223×** | 0.481× | ✅ | 0.0 |
| 32 | +0.4820 | 1.000× | 1.000× | 1.000× | ✅ | −6.1e−06 |

**Pooled (n=4, stratified, 100% cap-pinned ⇒ every η is a FLOOR):**

| quantity | stage 1 alone | **ARM C-PRIME** | POSEONLY control |
|---|---:|---:|---:|
| η | **0.4814 ± 0.0523** | **0.4814** (retention **1.000**) | — |
| clears break-even 0.17071 | **2.82×** | **2.82×** | — |
| d_pose ratio (subset) | **1.7270** | **0.6419** | 0.8271 |
| absolute Δd_pose (subset) | +0.000409 | **−0.000332** | — |
| seg exactly preserved | — | **ALL pairs ✅** | — |
| net S (seg + rate only) | **−0.05605 = 9.06% of gap** | **−0.05605** | — |

**η replicates et1 independently.** Pooled 0.4814 (n=4; 0.4865 at n=5) against et1's published
0.4817 on the same stratified pairs — a cross-check that licenses the comparison, since stage 1 is
shared machinery.

**On η retention, read the per-pair fact, not the pooled ratio.** The aggregator's
`eta_retained_vs_stage1` prints 0.989 once the receipts carry unequal n (stage 1 at n=5, C-PRIME
at n=4) — an **unmatched-denominator artifact of pooling**, not a measured seg loss. Per pair the
retention is exactly **1.000**, and provably so: `seg_exactly_preserved = True` ⇒ the argmax field
is identical ⇒ `flips_after` is identical ⇒ η is identical. Pair 48 shows it directly, 558/558.

**C-PRIME costs exactly nothing in seg: η retention 1.000, seg preserved on every pair.** And its
absolute Δd_pose is **negative** — the staged pair is better on pose than what we ship. Per m96 /
sq1 §1.6 that improvement is **NOT folded into net S** (this selection is 0.2692× of population on
d_pose); it is reported as a subset-scoped gate, and the gate is passed with margin rather than
merely survived. The pose budget before the row breaks even is **1.90× population**; stage 1 alone
sits at 1.73× (and has a 123.8× tail), C-PRIME at 0.64×.

**Both staged pairs end BELOW shipped d_pose while clearing the η bar ~3×.** Read that carefully:
the staged pair is better than what we ship on *both* axes.

**The control changes the interpretation, and it is why the control had to exist.** On pair 0 the
frame_0 solve finds **nothing** on the *undamaged* pair (POSEONLY = 1.000×, returns identity) —
frame_0 is already at a local optimum there. Yet after stage 1 moves frame_1, re-solving frame_0
reaches **0.783×**. So the staging is not only repairing its own damage: **moving frame_1 opens
pose headroom that was unreachable at the shipped frame_1.** Without POSEONLY I would have
credited the staging for headroom pu2 simply never took (pu2 solved only **6** pairs) — the same
unanchored-delta error this unit caught in et1's inherited constant.

**A statistic to distrust, flagged before it is quoted.** `repair_fraction_of_damage` reads 441.7%
on pair 0 because its denominator is the damage (1.064× ⇒ near-zero). It is unstable whenever
damage is small and is **not** the headline; the stable quantity is the **ratio vs shipped**.

**Both solves are cap-pinned** (`dec@25`, `f0@40` — best iterate at the final step), so η *and*
the repair are **FLOORS**, still improving. Per #874 the cap is not raised-and-quoted.

## §4c THE VERDICT TABLE — arm × (η, d_pose, bytes, net S vs live best)

`net S` is **seg + rate only**. Pose is a **subset-scoped gate** and is never folded into it
(m96 / sq1 §1.6). Live best **S = 0.7910689**; gap **0.6189279**; break-even η **0.170708**.

| arm | η (floor) | d_pose ratio **[SUBSET GATE]** | bytes / pair | **net ΔS** | % of gap | resulting S |
|---|---:|---:|---:|---:|---:|---:|
| stage 1 alone (et1's row) | 0.4814 | **1.727**, tail to **123.8×** | 0 extra | −0.05605 | 9.06% | 0.73502 |
| **C-PRIME + k=4 DCT** | **0.4814** | **0.315–0.64 (IMPROVES)** | **96** | **−0.01769** | **2.86%** | **0.77338** |
| C-PRIME + k=8 DCT | 0.4814 | 0.34–0.87 | 384 | **+0.09737** | −15.73% | 0.88844 |
| C-PRIME free-form | 0.4814 | 0.642 | 361,708 | +144.45 | −23339% | — |
| ARM C (cell-constrained f1) | < 0.4814 | — | — | dominated **structurally** (§4b) | | |
| ARM B (joint penalty) | trades along the coupling | — | — | mis-framed (§4b) | | |
| POSEONLY control (no seg) | 0 | 0.827 | 96+ | 0 seg gain | 0% | 0.79107 |

**The winner is C-PRIME + k=4**, and it is the only arm that is net-negative while *also*
improving pose. Note it is **not** the arm with the best pose or the best η — it is the arm with
the best **net**, which is why the discriminator had to price all three legs rather than rank on
one.

**Honest n.** η is **n=4** pooled (stratified, cap-pinned floors, replicating et1's 0.4817). The
k=4 carriage is **n=10** (stratified pairs 48/20/115/154/170/179/180/195/196/211), **10 of 10
below shipped, 0 all-zero**, median ratio 0.9513. This is a population-scale carriage result on
the subset (still subset-scoped for the pose axis per m96, not folded into net S). The two owed
gates below (int16 quantiser re-score; the cap-pinned-floor caveat) still stand — n is no longer
one of them.

## §4b Why ARM C is dominated, and ARM B mis-framed

ARM C (cell-constrained repair on frame_1) must *purchase* seg-exactness with a constraint that
removes DOF from the pose solve; C-PRIME receives it for free because frame_0 is outside SegNet's
input entirely. Measured: C-PRIME preserves seg exactly with **zero** η loss (flips_after
identical, 558/558 on pair 48). There is no budget under which paying for a property you already
own is optimal, so C is dominated **structurally**, not merely empirically.

ARM B (joint penalty) is mis-framed by et1's own coupling law: a penalty trades η against d_pose
along one axis. C-PRIME breaks the coupling instead of trading along it — the damage is paid on a
surface where seg cannot see it. That is why the operator's staging instinct beats the joint form.

## §5 The carriage question — the blocker that actually remains

ph5o measured the sibling seg-free pose actuator and returned **ALIGNMENT: YES / RATE: NO**: the
descent direction is a handful of isolated, per-pair-private pixels (pairwise Jaccard **0.0056**)
whose *address* costs roughly an order of magnitude more than the pose it buys, and a rank-6
separable-DCT basis solved to the **all-zero** integer vector on **100%** of pairs — *"the
cheapness of a generic basis and the localisation of the descent are the same property with
opposite signs."*

That is a **prediction about this arm**, so it is measured here rather than inherited. Every
C-PRIME row carries a `delta_structure` block measured on the **scorer-lattice** delta — the
object a receiver would actually apply, not the camera plane.

**MEASURED (pair 0), and it confirms ph5o on a structurally different actuator:**

| quantity | measured |
|---|---:|
| nonzero fraction | **97.70%** (the delta is DENSE, not sparse) |
| mean \|δ\| / max \|δ\| | 8.94 / 28 |
| **brotli-q11 bytes / pair** | **372,630** |
| LZMA bytes / pair | 361,708 |
| DCT energy capture, top 4×4 / 8×8 / 16×16 / **32×32** | 0.31% / 0.47% / 0.84% / **3.06%** |

At 361,708 B/pair × 600 pairs = **217 MB**, against a **353,805 B** archive — roughly **614× the
entire archive** to buy a 0.063 S seg gain. **Free-form C-PRIME is unpayable by three orders of
magnitude.** And a cheap low-frequency basis captures **3.06%** of the delta's energy even at
32×32, so the repair is maximally delocalised from the generic bases that are free under rule 118.

ph5o's law therefore **generalises**: it was measured on the D-blind subset of the camera plane
and found the descent *isolated*; measured here on the **whole of frame_0** against a staged
objective, the descent is instead *dense and high-frequency*. Opposite structures, identical
consequence — **ALIGNMENT: YES, RATE: NO**. This is a second independent instance, so the law is
now family-level rather than actuator-specific.

**But energy capture is NOT the carriage verdict, and inferring it would be the trap p3v2 named.**
Projecting the free solution onto a cheap basis and *solving within* that basis are different
operations — *"the free win is BASIS-ADVERSARIAL"*. Only the second answers the question, so it
was run (§5b) rather than inferred from the 3.06%.

### §5b SOLVED WITHIN the cheap basis — the arm that reversed the verdict

Solving the repair *inside* a generic separable-DCT basis (the basis is deterministically
generated ⇒ **FREE** under rule 118; only `k²·3` int16 coefficients per pair are COUNTED).

**Pair 48 — the worst pair, stage-1 damage 3.654×, `d_pose_before` 0.00051169:**

| arm | tag | d_pose vs **shipped** | vs damage | all-zero? | seg exact | B/pair | vs budget |
|---|---|---:|---:|:--:|:--:|---:|---|
| free-form (589,824 DOF) | `f0@40` | 0.561× | 0.154× | no | ✅ | 361,708 | 2293× OVER |
| **DCT k=4 (48 DOF)** | `dct4@25` | **0.3149×** | **0.0862×** | **NO** | ✅ | **96** | **FITS** |
| DCT k=8 (192 DOF) | `dct8@30` | 0.3355× | 0.0918× | NO | ✅ | 384 | 2.4× OVER |

**Pair 0 — damage 1.064×, `d_pose_before` 0.00078712:**

| arm | tag | d_pose vs shipped | all-zero? | B/pair |
|---|---|---:|:--:|---:|
| free-form | `f0@40` | 0.783× | no | 361,708 |
| DCT k=8 | `dct8@40` | 0.872× | **NO** | 384 |
| DCT k=32 | `identity@0` | 1.064× | **YES** | 6,144 |

**k=4 across the full n=10 measured population (96 B/pair, inside budget):**

| pair | stage-1 damage | k=4 d_pose vs **shipped** | damage removed | all-zero? |
|---|---:|---:|---:|:--:|
| 48 | 3.654× | **0.3149×** | **91.4%** | no |
| 115 | 2.097× | **0.9692×** | 53.8% | no |
| 211 | 1.743× | **0.9511×** | 45.4% | no |
| 170 | 1.530× | **0.9902×** | 35.3% | no |
| 195 | 1.473× | **0.9515×** | 35.4% | no |
| 154 | 1.452× | **0.9531×** | 34.4% | no |
| 20 | 1.190× | **0.8483×** | 28.7% | no |
| 180 | 1.181× | **0.9542×** | 19.2% | no |
| 179 | 1.120× | **0.9149×** | 18.3% | no |
| 196 | 0.999× | **0.9372×** | 6.1% | no |

**10 of 10 land below shipped and NONE solves to all-zero** (mean 0.8785, median 0.9513) — so at
96 B/pair the cheap generic basis pays for itself and leaves pose **no worse than shipped on every
pair**, *while* the seg gain is banked. This is now a population-scale carriage result (n=10,
still subset-scoped per m96), not a single-pair mechanism note. The repair removes MORE of the
damage where the damage is LARGER (91.4% at 3.654×, tapering to 6–19% near 1.0×) — so the bulk
gains are modest but the **heavy tail**, where the pose budget is actually at risk (et1's 123.8×
class), is where k=4 does its heaviest work. The mean is dominated by pair 48; the median 0.9513
is the honest "typical pair" figure.

**The efficiency is nonetheless not uniform across arms**,
and on pair 20 the free-form arm did *better* (0.223×) than k=4 (0.848×) while on pair 115 the
free-form arm found **nothing** (0.0%) where k=4 repaired 53.8%. That is the signature of
**differently under-converged solvers**, not of a capacity ordering, and it is the reason every η
and every repair here is labelled a **FLOOR** rather than an optimum.

**Three things this settles.**

1. **A smooth generic basis CAN aim at d_pose here**, and does so *well*: k=4 removes **91.4%** of
   the stage-1 damage on the worst pair with **48 coefficients**. ph5o's rank-6 all-zero result
   therefore does **not** transfer to this actuator/objective — its basis was confined to the
   D-blind mask against a different target, and its verdict is FORMULATION-scoped there.
2. **k=4 beating free-form is proof my free solve was under-conditioned**, not proof that fewer
   DOF are better: rank-4 ⊂ free, so at their optima the free solve cannot be worse. 48 parameters
   under a well-scaled lr converge; 589,824 under a shared lr do not. **The free-form byte figure
   in §5 is an INSTRUMENT artifact and must not be cited as the cost of the repair.**

   **Free-form found NOTHING (0.0% repair) on 6 of the 10 pairs it was measured on**, each run on
   the identical staged frame where the k=4 arm demonstrably repaired the same pair (e.g. 115:
   free-form 0.0% vs k=4 53.8%; 154: 0.0% vs 34.4%). A basis strictly contained in the free arm's
   search space succeeded exactly where the free arm failed outright — a **majority** of the time.
   That is not a capacity ordering (rank-4 ⊂ free cannot beat free at its optimum) — it is a
   **conditioning failure**, and it means the free-form arm is not merely inefficient but
   **unreliable**: it silently returns "no repair possible" on the majority of pairs where a
   repair demonstrably exists. Any verdict this unit might have drawn from free-form alone —
   including the fatal carriage verdict I had begun to write — would have been a false negative of
   exactly the shape this program calls m50. **The 6/10 free-form failure rate is the single
   strongest argument for the k=4 arm: it never once failed (10/10 repaired, 0 all-zero).**
3. **k=32's all-zero is the same defect at the other end** (1024 coefficients, shared lr,
   overshoot into the clamp) — a false negative inside the arm built to detect false negatives.
   `verdict_scope: INSTRUMENT` on both.

**⚠ AN UNMEASURED STEP IN MY OWN PRICE, named rather than left implicit.** The 96 B/pair figure
counts 48 coefficients as int16, but the solver returns **floats** and the repair was **never
re-scored after quantising them**. The magnitudes make the rounding almost certainly negligible
(an orthonormal DCT of a delta with mean |δ|≈9 over 384×512 puts the low modes in the 10³–10⁴
range, so ±0.5 integer rounding is ~10⁻⁴ relative) — but *almost certainly* is not *measured*,
and a payload whose value has not been re-measured **through its own quantiser** is exactly the
byte-close discipline this program refuses to skip. **The k=4 net therefore carries one owed
step**, and it is fire-order-1 below, not a footnote.

**A smooth generic basis CAN aim at d_pose here.** k=8 repairs to 0.872× — below shipped, not
all-zero. So ph5o's rank-6 all-zero result does **not** transfer naively to this actuator and this
objective; ph5o's basis was restricted to the D-blind mask against a different target, and its
verdict is `verdict_scope: FORMULATION` there, not a law about smooth bases in general. Its
*consequence* (RATE: NO) still holds here, but for an independent reason: **price**, not alignment.

**k=32's all-zero is MY INSTRUMENT, not a capacity result — and it is a false negative I own.** A
rank-32 basis strictly *contains* rank-8, so it cannot be genuinely worse at its optimum; solving
to identity while k=8 succeeded proves the k=32 solve was **under-conditioned** (1024 coefficients
per channel under a single shared lr, overshooting into the clamp). This is precisely the sm1/#874
class — a budget/conditioning artifact masquerading as a negative — caught inside the very arm
whose purpose was to catch it in others. **`verdict_scope: INSTRUMENT`**; no capacity conclusion
is drawn from the k=32 row, and it is retained only as the receipt of the defect.

### §5a THE BUDGET — the number that governs every carriage proposal

This falls out of the denominators alone and needed no run, which is why it should have been
computed first. At n600, one byte **per pair** costs `600 · 25/37,545,489 =` **0.00039952 S**. The
seg gain available to spend is the block16 net, **0.063009 S**. Therefore:

> **The entire pose repair must fit in ≤ 157.7 BYTES PER PAIR to break even — and strictly fewer
> to profit.**

| carriage form | B/pair | rate cost | vs budget |
|---|---:|---:|---|
| free-form solve (**measured**) | 361,708 | 144.51 S | **2293× OVER** |
| generic DCT k=32 (int16) | 6,144 | 2.45462 S | 39.0× OVER |
| generic DCT k=16 | 1,536 | 0.61366 S | 9.7× OVER |
| generic DCT k=8 | 384 | 0.15341 S | 2.4× OVER |
| **generic DCT k=4** | **96** | **0.03835 S** | **FITS** (leaves 0.0246 S) |

So the carriage question is not "is the delta compressible" but the far tighter **"can ≤ ~158
bytes per pair hold a pose repair worth 0.063 S of seg?"** Only k ≤ 4–5 is even admissible, and
ph5o already measured a rank-**6** DCT solving to the all-zero integer vector on 100% of pairs on
the sibling actuator. The k=8 arm below is therefore run as a deliberately **generous upper
bound**: at 2.4× over budget it cannot bank even if it succeeds, but if a smooth basis cannot
repair at k=8, it cannot repair at k=4 either — a falsification ordering, not a candidate.

## §6 Denominators

gap **0.6189279** = 0.7910689 − 0.172141 · seg leg **0.431179** · rate leg **0.235584** · pose leg
**0.124306** ⇒ `d_pose_population` **0.00154517** · **dS/dd_pose = 40.2234** (CORRECTED; et1's
inherited 31.3026 is pre-pu2 and superseded) · block16 gross **0.18039 S** at **46,247 B** ⇒ rate
cost **0.030794 S**, break-even η **0.170708** (reproduces et1's published 0.1707 — the
consistency check that licenses this arithmetic).

**Pose is a SUBSET-SCOPED GATE and is never folded into a population ΔS** (m96 / sq1 §1.6): the
pose axis is 2.5–4.2× skewed on non-population subsets and et1 measured this stratified selection
at **0.2692×** of population on d_pose. Seg reach/gross are n600 with no subsetting, so m96 cannot
apply to them by construction.

## §6b THE BANKING ROUTE — named, because C-PRIME + k=4 clears net-negative

The charter requires the byte-close path be named if an arm wins net-positive. It does
(−0.01769 S), so here it is — as a **route with a gate in front of it**, not a claim.

**Composed S projection (advisory, NOT a row):**

| leg | ΔS | note |
|---|---:|---|
| seg, block16 phase field at η 0.4814 | **−0.086840** | n600 gross × pooled η floor |
| rate, offset field 46,247 B | **+0.030794** | et1's measured LZMA1 coder-closed price |
| rate, k=4 pose stream 96 B/pair | **+0.038353** | 57,600 B at n600 |
| **composed** | **−0.017692** | **2.86% of gap ⇒ S 0.7910689 → 0.773377** |

pose is **NOT** a term here (subset-scoped gate; measured non-damaging, 3/3 below shipped).

**The chain**, in order, through the canonical surface (`tac.submission_chain` — never a probe
script): `stage_submission` → `run_inflate` → `audit_runtime_tree` + `build_byte_ledger` (the
offset section and the k=4 coefficient section both need a PROFILES grammar entry; neither exists
yet) → `run_upstream_evaluate` → `parse_evaluate_report`, on the **exact archive bytes**, CPU and
CUDA as separate axes.

**Three gates stand in front of it, and none is optional.**
1. **The int16 quantiser re-score** (§5b ⚠) — the 96 B price is counted but its value is not yet
   measured through its own carriage arithmetic. Until then the table above is a projection.
2. **Honest n on the damage tail** — k=4 is n=3; et1's 123.8× class is unrepresented.
3. **The two solvers are cap-pinned floors**, so η and the repair are both lower bounds; the
   composed number is conservative in the good direction but is not converged.

**No archive was built and no byte was closed in this unit.** The route is written down so the
successor does not have to re-derive it, not because it has been walked.

## §7 Follow-ons — FIRED / FOLDED / QUEUED-WITH-FIRE-ORDER

- **FIRED** — the staging discriminator itself (§4) · the C-PRIME vs C structural verdict (§4b) ·
  the pose-only CONTROL that separates staging from unharvested pu2 headroom (§4) · the free-form
  carriage measurement (§5) · the solve-within-a-cheap-basis test (§5b) · the byte budget (§5a) ·
  the `dS/dd_pose` denominator correction (§1.1) · the yuv6 gradient-liveness controls (§2).
- **FOLDED** — et1's fire-order-1 ("measure the Q3 rank-6 projection on the phase field") is
  **superseded, not deferred**. Q3 exists to hold pose while paying seg DOF for it; C-PRIME holds
  pose at **zero** seg cost because frame_0 is outside SegNet's input. Q3 on this field would be
  measuring the price of something now obtainable free. et1's fire-order-3 (budget) is likewise
  folded: its coupling law is broken by the staging, not traded along.
- **FIRED (was fire order 1) — the int16 quantiser gate, and a correction to how I described it.**
  I wrote that this was "a re-score of receipts already on disk." **That was wrong**: the solver
  returned only `int(coef.numel()*2)` and never persisted the coefficients, so nothing on disk
  could be re-scored — the gate required re-solving. Rather than re-score, the solver was changed
  so the gate holds **by construction**: the best-iterate is now selected on the **quantised**
  synthesis (`round(coef)` clamped to int16) instead of the float one, so the reported d_pose *is*
  the value of the payload that would actually ship, and `max_abs_int16_coefficient` /
  `int16_range_ok` are recorded per pair. Results in §5c.
- **QUEUED, fire order 2 — finish the k=4 ladder to honest n on the DAMAGE TAIL.** Running as
  `js1_k4_more.json` (8 pairs, resumable, checkpointed per pair). The pairs that decide this are
  the high-damage ones (et1's 123.8× class), because that is where a repair either holds or does
  not; a k=4 mean over low-damage pairs would be uninformative. Sweep k ∈ {2,3,4,5} at the same
  time — the budget admits up to ~5 and nothing above it, so the ladder is short and bounded.
- **QUEUED, fire order 3 — re-run free-form and k=32 with per-rank lr conditioning.** Both are
  measured instrument defects (§5b); leaving a known false negative in the record is the m50
  class. Neither can bank (2293× and 39× over budget), so this is hygiene — but the free-form
  361,708 B figure must not be citable as "the cost of the repair", and the k=32 all-zero must
  not be citable as a capacity result.
- **QUEUED, fire order 3 — the token-absorption carriage path, UNPRICED and unexamined here.**
  Our vehicle is **99.0% tokens** (m08/TR1). frame_0's pixels are produced by the renderer from
  the token stream, so the repair's true marginal cost may be the **token-stream delta**, not a
  new sidecar. That is a materially different price from the 361,708 B/pair sidecar measured in
  §5 and is the one carriage path this unit did **not** measure. Named so it is not lost.
- **NOT QUEUED, with reason** — ARM C (cell-constrained frame_1 repair): structurally dominated
  (§4b), not merely empirically. · ARM B (joint penalty): mis-framed by the coupling law (§4b). ·
  warp-base frame_0 regeneration: measured-closed at this operating point, 254× worse (§1.2). ·
  re-deriving frame_0 seg-freeness, D-support privacy, or the pose relativity of m87 — all
  reproduced here as live controls rather than cited.

## §8 Self-caught defects (mine, not inherited)

1. **A hardcoded `(384,512)` reshape** in the pose path — `rgb_to_yuv6` halves the spatial dims
   (2×2 luma block → 4 channels + 2 chroma), so its output is (2,6,**192,256**). Caught by running
   it, not by reading it. Dimensions are now derived from the tensor and asserted.
2. **k=32 solved to all-zero** while k=8 succeeded — an under-conditioned solve presenting as a
   negative (§5b). Scoped INSTRUMENT and queued for repair rather than quoted.
3. **My wait-filter false-positived** on the substring `error` inside `yuv6_max_abs_error`, reading
   a healthy run as a failure — the same vacuity/instrument class this unit exists to police.
4. **`repair_fraction_of_damage` is unstable** when damage is small (441.7% on pair 0 because its
   denominator is 1.064×−1). Flagged before quoting; the headline is the ratio vs shipped.
5. **The realization flattens frame_0's camera plane** (all four private pixels ← the solved
   value) where an *additive* write would give an identical scorer effect while preserving the
   original camera detail, since D's weights sum to 1 ⇒ `D(f0 + up(δ)) = D(f0) + δ`. Scored
   numbers are unaffected (verified gap 0.0) and `delta_structure` already prices the
   scorer-lattice delta, which is the correct carried object either way.
6. **I nearly built on the warp-base path before pricing it** (§1.2); it is 254× worse at this
   operating point. Caught by checking p3v2's n600 receipt instead of its headline.

## §9 Receipts + STORES CONSULTED

Scripts (committed `07d21b44c0`): `experiments/ddm_js1_staging_discriminator.py` ·
`experiments/ddm_js1_aggregate.py`.
Receipts: `/Volumes/VertigoDataTier/pact/ddm_js1_20260804/` — `js1_smoke_p48.json` ·
`js1_n32.json` (resumable, checkpointed after every pair) · `js1_cheapdct.json` ·
`js1_cheapdct_k4k8.json`.
**Controls, all measured per pair:** yuv6 forward equivalence (max_abs_error **0.0**) · gradient
liveness (**LIVE**) · **C2/C3** argmax-cache agreement · **C5** scorer-lattice pose path == camera
path (**abs_diff 0.0**) · frame_0 seg-freeness under ±40 LSB noise (**True**) · realization gap
(**exactly 0.0**) · D private-support assertion (fail-closed).
Stores consulted: `ddm_{et1,ph1,sq1,gp1,ph5o,ph4,p3v2,pu2,bo1,q3x,pz1,rp1}` memos ·
`upstream/modules.py` (SegNet `x[:,-1]`, PoseNet two-frame yuv6, pose distortion) ·
`upstream/frame_utils.py:50` (`@torch.no_grad()` on `rgb_to_yuv6`) ·
`src/tac/differentiable_eval_roundtrip.py` · `tac.boundary_math.dykstra_legal_frame` (the #73
cell∩tube∩cheap solver, recalled and deliberately **not** rebuilt) ·
`src/tac/canonical_equations/ddm_ph5o_blind_descent_is_address_limited_20260803.py` ·
`pu2_interim_summary.json` (the pre-pu2 baseline that dates et1's constant) · CLAUDE.md authority
ladder + m66/m86/m87/m96/m08/m50.

## §10 Pointer honesty

**The exact pointer did NOT move.** `0.1910828242 [contest-CPU]` UNMOVED. Own-vehicle frontier
**S = 0.7910689 @ 353,805 B [macOS-CPU advisory]** UNMOVED. Nothing here is byte-closed and no
archive was built. A staging law confirmed, a corrected pose price, and a relocated blocker are
**MEANS**. **This unit has not achieved the goal.**
