---
arm: ddm_gt3
title: "The addressing IS free and that is not enough: a receiver-recomputable basis needs eta >= 0.7164, and the only realizer that reaches it is pose-catastrophic"
utc: 2026-08-03
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
own_vehicle_frontier: "S = 0.7910689 @ 353,805 B [macOS-CPU advisory] UNMOVED this unit"
axis: "[macOS-CPU cache-derived + certified-decode advisory] NON-PROMOTABLE"
slot: "held the n600 scorer slot; ZERO scorer forwards fired -- the decisive measurement turned out to be cache-derived. Slot returned unspent."
baseline_named: "live best S = 0.7910689 @ 353,805 B (pu2, archive sha c72ef357); gap to the PR130 bar 0.172141 = 0.6189279; 1% of gap = 0.0061893 S = 9,295 B; W = 1.273108215332031 B/flip (recomputed from components; agrees with the banked m66 value 1.2731082153320312 to 1 ULP -- a float-repr artifact, not a disagreement)"
verdict_scope: "FORMULATION (photometric+geometric free features on decoded RGB). NOT the free-basis family."
consumes:
  - ".omx/research/ddm_ob1_band_price_n600_and_ordering_break_even_20260803.md (the receiver audit + the 544,499 B hand-built legal floor I beat by 13.1%)"
  - ".omx/research/ddm_sq1_eta_seg_and_hinge_ab_20260803.md (both etas, TRANSFERRED not re-measured)"
  - ".omx/research/ddm_gt1_upstream_gt_unmined_inventory_20260803.md (the free-basis/counted-coefficient law this unit operationalizes)"
  - ".omx/research/ddm_mg1_margin_geometry_cure_20260803.md (the GT-margin enrichment curve, cited as the illegal ceiling)"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_gt3 — the free address, priced exactly

## §0 ANSWER FIRST

The operator's remark — *"the addressing is very cheap and kind of falls out"* — is **TRUE on
bytes and FALSE on selectivity**, and the distinction is the whole result.

Under `gt1`'s law I built the scheme where addressing genuinely costs **zero**: the receiver
recomputes a bin from its own decoded RGB + the fixed operators + its own token lattice, so
naming the corrected set costs no counted bytes at all. Only coefficients are counted. The
address term does not merely get cheap — it **vanishes**.

It still does not pay, and the reason is not the address:

> **A free address is free but not SELECTIVE.** The cost migrates out of the address and into
> the **waste**: the best free bin is **95.8% non-flips**. Break-even now lands entirely on the
> realizer, and the threshold is **eta >= 0.7164**. The measured *pose-neutral* realizer is
> **0.5406** (1.33x short, buys **ZERO** bins). The only realizer that clears it is sq1's
> seg-only paint at **0.7895**, which carries a **56.5x d_pose regression** and is therefore
> not a row.

**Typed outcome:** `ROW_BLOCKED_ON_A_SINGLE_MEASURED_NUMBER(eta_pose_neutral, threshold=0.7164)`
+ `SPECIFICATION_ISSUED` (§6). No family killed. The exact pointer did not move, nothing here is
byte-closed, and **this unit did not achieve the goal**.

**I also refute two things I wrote myself earlier in this same unit** (§2.4, §2.5). That matters
more than the headline, because the first of them would have produced a confident false kill.

---

## §1 Positive controls — four PASS, one EXACT-IDENTICAL to a sister arm

| # | control | result |
|---|---|---|
| C1 | `d_seg` + flip count reproduced from the argmax caches | **PASS EXACT** — 508,640 flips, `0.004311794704861111` (= gp1, ob1, sq1, pu2, `report.txt`) |
| C2 | my independent inflate of archive `c72ef357` | **PASS BIT-IDENTICAL** — 3,662,409,600 B, sha256 `0323386f54f5…ac36`, **exactly ob1's certified decode sha**. Any difference below is a difference in the OBJECT, never in the pipeline. |
| C3 | train/test split representativeness (`m88`) | **PASS** — interleaved split, test/pop **1.00731**, train/pop **0.99269** (a contiguous prefix[0:300] would have been 0.97134) |
| C4 | held-out vs in-sample optimism | **PASS** — **1.0010** (0.10%), matching ob1's independently measured 0.10% on its own table |
| C5 | vacuity (`m50`) | **PASS, denominators printed** — 117,964,800 sites, 58,982,400 held-out, 300+300 pairs, **1,659 of 3,600 bins carry support**. A zero-bins result below is a MEASURED zero on a populated scope, not an empty scan. |

C2 is the load-bearing one. Reproducing another arm's decode sha **bit-for-bit** is the strongest
plumbing control available on this vehicle, and it is why §2's departures from ob1 are real.

---

## §2 What I REFUTE

### 2.1 mg1 98.1x vs sg2 2.713x — NOT a conflict. Different quantities. Neither licenses a cheap free address.

Resolved at source, and the resolution matters more than the arithmetic:

- **`mg1` 98.1x** is *flip enrichment* in the low-**GT-margin** set (`mg1` §2.1: margin < 0.1 →
  0.2824% of sites carrying 27.69% of flips; §2.2 precision **42.51%**). It is computed on the
  frozen SegNet's margin evaluated on **ground truth**.
- **`sg2` 2.713x** is a *through-R input-leverage tilt* on the decoder-margin flip-capable set
  (`gt1` §2 GT1-3). Leverage, not flips.

Both are correct as measured. But the one relevant to addressing is `mg1`'s, and **its field is
GT-derived AND scorer-derived — the receiver cannot compute it, at any price short of the 38.5 MB
SegNet.** So the banked "98.1x = a cheap address mode" is **not a cheap address**: it is an
excellent *encode-side selector*, which is exactly and only the role `gt1`'s law assigns to GT.
Cheap-address claims must not cite it.

### 2.2 "Addressing is very cheap and kind of falls out" — true on bytes, false on what binds

I built the zero-byte address and it is real. What it buys is bounded by concentration, and that
is where the family fails. Measured at n600, held-out:

| ranking | flip density | enrichment over the 0.4343% base |
|---|---:|---:|
| best FREE bin (decoded RGB + `D` + geometry + token lattice) | **4.469%** | **10.29x** |
| `mg1` GT-margin oracle, margin < 0.096 (cited, ILLEGAL) | **42.51%** | **98.1x** |

**The oracle address is 9.5x more concentrated than anything free I could build.**

### 2.3 ob1's 544,499 B is not a floor — confirmed twice over, and beaten by 13.1%

MAIN told me the number is a today's-vocabulary artifact; `gt1`'s law adds that it prices the
BASIS as if counted. Both hold. Independently: on the same object, with two features ob1 did not
have (a **luma ridge** detector, motivated because Road<->Lane is 46-49% of flips and lane markings
are literally thin luma ridges; and **distance to the renderer's own 16x16 token-cell seam**, which
is pure receiver-known geometry), the legal indicator address costs **473,307 B vs ob1's 544,499 B
— 13.1% cheaper**. It changes no verdict, which is the point: the number was never the floor.

### 2.4 MY OWN first waterfill was a DOMINATED scheme and would have produced a false kill

My first pass priced *"ship the class at every site in a taken bin"*. It returned **zero bins at
every eta including 1.0**, and I was one step from writing that up as a clean structural negative.

It is wrong, because a strictly cheaper scheme exists: inside a taken bin ship a **flip INDICATOR**
coded at `P(flip|bin)`, and the class **only on flips**. Since `H(flip|bin) << H(gt|bin)`, this
dominates. Corrected, the same free basis takes **138 bins and −0.06031 S (9.7% of gap) at eta=1.0**
— the opposite of a structural impossibility.

Caught by cross-checking my own aggregate against ob1's 544,499 B and noticing my "dead" full-field
number implied an *affordable* address at eta=1. Both schemes are kept in the receipt
(`waterfill_class_everywhere_DOMINATED` vs `waterfill_indicator_THE_DECIDING_SCHEME`) so the
dominated one cannot be re-derived by a future arm.

### 2.5 MY OWN charter ranking — I said eta was not the binding unknown. It is exactly the binding unknown.

Off the dominated scheme I had concluded "it fails at eta=1.0, so no realizer can save it, so the
paint-and-read-back rate is not decision-relevant." Under the corrected scheme the entire verdict
is an eta threshold: **0.7164**. The charter's instinct (measure paint-and-read-back first) was
right and my intermediate reasoning was wrong.

---

## §3 THE LADDER — one object, one denominator, n600 held-out

Scheme: receiver recomputes bin j from free features; inside taken bins, flip indicator at
`P(flip|bin)` + target class on flips at `P(gt|bin,flip)`; outside, nothing (default = do not
paint). Counted = per-taken-bin table (12-bit id + 4x4-bit probs = **3.5 B/bin**) + the coded
stream. **Address bytes: zero, by construction.**

| eta | bins | sites painted | flips in set | precision | counted B | net S | % of gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| **0.5406** pose-neutral (sq1 P7) | **0** | 0 | 0 | — | 0 | **0.00000** | **0.00%** |
| 0.7895 seg-only (sq1, d_pose 56.5x) | 9 | 2,998,872 | 115,760 | 3.86% | 108,214 | −0.00542 | 0.88% |
| 1.0 hard upper bound | 138 | 20,066,358 | 410,216 | 2.04% | 431,669 | −0.06031 | 9.74% |

**Minimum break-even eta over all 1,659 supported free bins = 0.7164**, achieved on a bin of
401,825 held-out sites at density 4.199%. That single number is the whole verdict.

For contrast, the naive route (`class everywhere on the free basis`, no indicator) costs
**5,397,443 B = 15.26x the entire shipped archive**. Bulk GT is dead by a wide margin and this
unit reproduces that independently.

---

## §4 THE STRUCTURAL FINDING — why free features top out at 10x

Every feature I built measures **scene complexity**: gradient, ridge, distance-to-edge, row,
token-cell phase. Not one measures **render wrongness**. Scene complexity buys 10.3x. The scorer's
own margin buys 98.1x. The 9.5x gap between them is exactly the gap between *"where is this scene
hard"* and *"where is my render wrong"* — and the second requires knowing the truth.

Stated as the law, because it generalizes past this feature set:

> **A free address for our own errors is CIRCULAR.** The best possible receiver-side predictor of
> `flip` is `P(gt != L*(decoded) | decoded)`, which requires predicting `gt` from the decoded
> frames — the same problem as fixing the render. A decoder that could locate its own errors for
> free could have avoided them.

Measured in bits, without depending on my binning: `H(flip) = 0.040335` bits and the free basis
recovers `I(flip; free) = 0.008236` bits = **20.42% of the information about where the errors
are**. The remaining 79.58% is what the oracle has and the receiver cannot get for free.

The escape is not a better hand-feature; it is a **COUNTED predictor** — `gt1` §4's "small end,
unpriced". §6 prices it.

---

## §5 The largest carried assumption, stated rather than buried

**Both etas are TRANSFERRED from sq1, and both were measured on a DIFFERENT SET.** sq1's etas
(0.5406 pose-neutral, 0.7895 seg-only) were measured on the r=1 `L*`-boundary band — a set with
far higher flip density than my free bins. eta is mechanism-AND-set scoped.

**The direction of the bias is against me, and I will not round it toward my own result.** My best
bin is **95.8% non-flips**; sq1's band is far denser. A solved paint applied to a set that is
overwhelmingly *already correct* has far more opportunity for collateral damage, so the honest
expectation is that **eta on my set is BELOW 0.5406**, not above — which would put the row further
from 0.7164, not closer. I did not measure it, and I am not going to imply otherwise.

That measurement is the single decisive open number in this unit, it is a scorer job, and the
exact bin definition needed to run it is in the committed receipt.

---

## §6 SPECIFICATION — what would make this live (never a bare kill)

1. **Measure eta, pose-neutral, on THIS unit's free-bin set.** Threshold **0.7164**. This is the
   whole row. Reuse `experiments/ddm_sq1_stage_decomposition_and_solved_paint.py::solve_margin_optimal_paint`
   with the P7 yuv6-null rider and the 2x2-block snap (sq1 §2.7 caveat 1: the projection does NOT
   commute with a pixel-granular mask). Stratify by bin, report per-bin eta, never a pooled number.
2. **The three free features that could carry render-wrongness rather than scene-complexity**, none
   of which is in my set, each genuinely zero-byte:
   (a) **temporal self-inconsistency** between adjacent pairs' rendered frame_1 — all 600 token
   grids are in the archive, so the receiver can render neighbours and difference them;
   (b) the renderer's own **pre-sigmoid activation magnitude** (`ddm_tr1_runtime.py:1300`) — a
   native confidence the receiver computes anyway and currently discards;
   (c) **inter-cell token disagreement** across the 16x16 lattice seams.
   These are the candidates for the missing 9.5x. Each is $0 on the existing decode.
3. **The counted-student rung, priced.** `mg1`'s oracle at margin < 0.2 is worth **≈ −0.0299 S
   (4.82% of gap)** at eta=0.5406 (my arithmetic on mg1's counts, assuming 1.5 bits/site; the
   assumption is stated, not hidden). So a distilled error-predictor must fit in **< 44,829 B AND
   reach GT-margin-oracle quality**. SegNet is 38,502,892 B ⇒ **859x compression with no quality
   loss**. Severe, and it is a specification rather than an impossibility — but nobody should
   start it before item 2, which is free.
4. **Do NOT re-derive:** the receiver holds no label field (ob1 + my §1 independent confirmation);
   `D`'s 2x2 private supports and the 22.70% blind fraction; bulk GT is dead; the mg1/sg2
   quantities are different (§2.1); the class-everywhere waterfill is dominated (§2.4).
5. **Scope.** `verdict_scope: FORMULATION` — photometric + geometric free features on decoded RGB.
   The free-basis FAMILY is not refuted; item 2 is the untested part of it.

---

## §7 Pointer honesty

Exact pointer **UNMOVED**: `0.1910828242 [contest-CPU]`. Own-vehicle frontier **UNCHANGED at
S 0.7910689 @ 353,805 B [macOS-CPU advisory]**. Nothing here is byte-closed; no archive was built.
A priced ladder is a MEANS, and by the standing rule **this unit has not achieved the goal.**
The scorer slot was held and **returned unspent** — the decisive measurement was cache-derived,
and the one job that needs the slot (§6.1) is specified but not run.

## §8 Artifacts

Script (committed): `experiments/ddm_gt3_free_basis_address_n600.py`.
Receipt (SSD, rebuildable): `/Volumes/VertigoDataTier/pact/ddm_gt3_20260803/gt3_free_basis_n600.json`
— carries both waterfills, the per-bin table, the shortfall block, the information block, and every
denominator. Decode: `inflated/0.raw`, 3,662,409,600 B, sha256 `0323386f54f5…ac36`, rebuildable in
~4.5 min from archive `c72ef357` via the shipped `inflate.sh` (command in ob1's cleanup certificate).
