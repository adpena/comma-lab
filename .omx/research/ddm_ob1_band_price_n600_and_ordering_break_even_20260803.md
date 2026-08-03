---
arm: ddm_ob1
title: "The seg address at n600: gp1's free band is not free, not legal, and the ordering rung is the whole mechanism"
utc: 2026-08-03
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU cache-derived + frozen-scorer advisory] NON-PROMOTABLE"
slot: "held the one full-n600 scorer slot; 630 SegNet forwards, chunked 50/checkpoint, resumable"
baseline_named: "live best S = 0.7910689 at 353,805 B (pu2, archive sha c72ef357) [macOS-CPU advisory]; seg leg 0.4311790; gap to the PR130 bar 0.172141 = 0.6189279; 1% of gap = 0.0061893 S = 9,295 B"
verdict_scope: "see per-result scoping; no family negative is licensed by this unit"
consumes:
  - ".omx/research/ddm_gp1_selective_gt_student_pricing_20260803.md (the ladder being re-priced)"
  - ".omx/research/ddm_sq1_eta_seg_and_hinge_ab_20260803.md (the eta this unit carries as a multiplier)"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_ob1 — the seg address, re-derived at n600

## §0 ANSWER FIRST

My charter asked me to re-derive gp1/sq1's two band prices at n600 and price the ordering gap.
Both prices reproduce **as arithmetic** and both are **wrong as physics**, for the same reason,
and the reason is not sampling:

> **gp1's "FREE label-boundary band" is computed from a label field the receiver does not have
> and cannot compute.** The shipped decoder holds no class label field of any kind. `L*` is the
> argmax of the 73 MB frozen SegNet. Every row in gp1's ladder marked *"address legal, zero
> compliance question, legal today"* is, as measured, **not legal**.

That relocates the whole result. The charter's framing — free band DEAD as priced, oracle band
LIVE, difference = the student's ordering — survives in its *ordering* half and inverts in its
*legality* half. Three things are separately true and only the decomposition separates them:

1. **The charter's A3-is-dead verdict is CONFIRMED and is understated.** At n600 on the
   decoder's actual `L*`, with gp1's own dilation, the r=1 band captures **83.334%** of flips,
   not 97.264%. gp1's falsifier **F4 fires at n600** (bar: 90%). Re-priced honestly the free
   band's net is **+0.02670 S** at sq1's measured eta, not the **+0.018 S** the charter carried
   — 1.48× deader. sq1 got this only half-right: it measured eta on the real band but kept
   gp1's **proxy-derived gross**, mixing a measured multiplier with an unmeasured multiplicand.

2. **The hard band was never the right object.** Thresholding at radius *r* is a crude use of a
   ranking. An optimal coder assigns every pixel a probability. Priced that way on the *same*
   `L*` information, the address costs **327,405 B at FULL capture** — cheaper than the r=1
   hard band (331,824 B) *and* it throws nothing away. Break-even eta falls **0.6149 → 0.5056**,
   below sq1's measured 0.5406, so the row goes **LIVE at −0.01509 S (2.44% of gap)**. Same
   information, same legality, better coder.

3. **But that row is illegal, and the legal floor is dead by a wide margin.** Using only what
   the decoder actually has — its own decoded RGB — the best address I could build costs
   **544,499 B** and needs **eta 0.8409**. Measured eta is 0.5406. **DEAD, and not marginally.**

**So the ordering rung is not a bonus on top of a live row — it is the entire mechanism that
could make any row live.** gp1 priced it at 106,954 B as *"economically wide open… the ladder
does not depend on it"*. The ladder depends on nothing else, and the true bar is far higher:
measured against the frozen scorer's own margin field (§4), **a student must capture 76.3% of
the legal→oracle information gap, at zero byte cost, merely to break even — and 99.3% to reach
the −0.039 S the charter called LIVE.**

**Typed outcome:** `ROW_DEAD_AS_LEGALLY_PRICED(gp1 A3/A4/B2, scope=FORMULATION)` +
`SPECIFICATION_ISSUED(student must capture ≥76.3% of the measured legal→oracle gap at ~0 bytes)`.
No family is killed: the rung's own ceiling (254,753 B) does exceed its break-even requirement
(194,431 B), so this is a hard specification, not an impossibility. The exact pointer did not
move and nothing here is byte-closed.

---

## §1 Positive controls — five PASS, one NOT (nothing below is admissible without these)

| # | control | result |
|---|---|---|
| C1 | `d_seg` reproduced from the argmax caches | **PASS EXACT** 508,640 flips, `0.004311794704861111` (= gp1, pu2, sq1, and `report.txt`) |
| C2 | payload entropy `H(gt \| rendered)` | **PASS** 1.101152 bits/flip vs gp1's 1.1011521 |
| C3 | **gp1's row A3 reproduced end-to-end** | **PASS** band 3.95948% (gp1: 3.95948%), capture 0.972639 (gp1: 0.972639), 367,873 B (gp1: 367,523 B, +0.10%), net −0.17443 (gp1: −0.174663) |
| C4 | sq1's F4 capture reproduced on sq1's own band | **PASS** 0.86701 at n600 vs sq1's 0.8668 at n=32 |
| C5 | frozen SegNet argmax on decoded `f1` == `cx1_argmax_n600` | **599 / 600 — NOT a clean pass.** One pair's re-forward argmax differs. Reported, not rounded up. |
| C6 | deterministic decode | **PASS** — my independent inflate of the shipped archive is **bit-identical on 1194/1200 frames** to the canonical decode |

**On C5, stated plainly rather than smoothed:** sq1 reported 32/32 EXACT; at n600 I get **1
mismatch in 600**. I do **not** know which pair — the failing-pair tracking was added to the
harness *after* that pair had already been processed, and I chose not to spend slot re-running
600 forwards to name it. The honest bound on its effect: the flip mask and the distance field
are both read from the **cache**, so a differing re-forward changes only that one pair's
**margin** bucketing — at most 1/600 of the oracle rungs' contingency tables, and nothing at
all in the legal or `L*` rungs. The likely cause is CPU-kernel nondeterminism (thread count /
BLAS path), which is the same class as the `[macOS-CPU advisory]` label this whole memo
carries. It does not move any verdict, but it is not a PASS and is not written as one.

**Two harness failures worth recording rather than hiding.** (i) My first inflate exited with
the harness reporting **exit code 0** while the job had actually failed (`python: command not
found`); only reading the log caught it — a live instance of the `m50` / `dn1x` fail-open
class, on the *wrapper*, not the script. (ii) The scorer job was **SIGURG-killed (rc=144)
three times** at ~2–3 min, under foreground, under `nohup`+`disown`, and under the harness's own
backgrounding alike. It completed only because it checkpoints every 25–50 pairs and resumes
from disk. The per-stage-checkpoint non-negotiable is the only reason this unit has a result at
all; the script now carries `--stop-after` / `--report-only` so a bounded invocation can never
emit a report that looks complete.

**C3 is the load-bearing one.** Because gp1's published row reproduces to 0.1%, every departure
below is a real difference in the *object*, not a difference in my arithmetic.

**C6 is worth its own line.** The 6 frames that differ are *all* `frame_0`, on pairs
**[16, 21, 67, 71, 74, 523]** — exactly pu2's six re-solved tail pairs. This independently
confirms (a) the decode is deterministic, (b) pu2's correction touches frame_0 only, and (c)
*why* `d_seg` is bit-exact across cx1→pu2: SegNet reads `x[:, -1, ...]`, so it never sees
frame_0. My features are computed on frame_1, which is bit-identical on all 600 pairs — so
there is zero cross-vehicle contamination between the RGB I measure and the argmax cache I
score against.

---

## §2 What I REFUTE

### 2.1 In gp1 — the band is a GT band wearing a receiver's name

gp1's free band is `dilate(boundary(z["argmax"]))` from `ddm_b2b_qa75_field_20260730`. Measured
at n600, that field agrees with **GT on 99.884%** of pixels (and with the decoder's `L*` on
99.538%). A flip is by definition a pixel where the decoder disagrees with GT; a band drawn
around a 99.88%-GT field therefore surrounds the flips *by construction*. gp1 labelled it a
"proxy… BOUND-flavoured", which is honest, but the resulting rows were then carried forward as
**"legal today, zero compliance question"** — and that is the part that does not hold.

### 2.2 In sq1 — a different band than the one it claims, and a mixed price

`ddm_sq1_eta_seg_realization.py::dilate` applies a 4-neighbour OR **and then a vertical-only
OR** per iteration. Measured: at r=1 it is **2.20× gp1's band area** and anisotropic (rowspan 5,
colspan 3, vs gp1's 3×3). Its docstring says *"Chebyshev dilation by r — gp1's convention"* and
§1.2 claims the band is *"byte-for-byte the SAME object gp1 priced at 367,523 B"*. It is not
either of those things.

This matters twice. (i) sq1's F4 number (86.68%) is correct **for sq1's band**; on gp1's own
convention the same label field gives **83.334%** — F4 fires harder. (ii) sq1 §4.2 computes
A3's seg gain as `eta × 0.41938`, where 0.41938 is gp1's **97.26%-capture** gross, while `eta`
was measured on the **86.7%-capture** band. Correcting to a single consistent object moves A3
from **+0.018 S** to **+0.02670 S**.

### 2.3 In my own charter

The charter states A2 (oracle band) is LIVE at **−0.039 S = 5.96% of gap**. Two corrections:
the percentage was computed against sq1's **stale** gap (0.654355, the cx1 baseline) — against
the live pu2 gap 0.6189279 the same −0.039 S is **6.31%**; and more importantly A2's gross
rests on an oracle ranking taken from the *qa75* render's margin field, a different vehicle
from the decoder being corrected. §3 replaces it with the decoder's own margin.

---

## §3 THE LADDER — one object, one denominator, n600

Address cost priced identically everywhere as the ideal code length of the flip indicator,
`Σ_i −[f_i log₂ p_i + (1−f_i) log₂(1−p_i)]` over all 117,964,800 scorer pixels. Hard-band rows
carry **measured LZMA1-raw** address bits over all 600 pairs (no prefix scaling — gp1 scaled a
150-pair coder run; I ran the coder on the whole population, so the `#875` caveat is gone).

| rung | receiver-legal? | bytes | capture | net S @ eta 0.5406 | % of gap | break-even eta |
|---|:---:|---:|---:|---:|---:|---:|
| gp1 A3 AS PUBLISHED (proxy label field, gp1 dilation, r=1) | no | 367,873 | 0.9726 | +0.01823 DEAD | −2.95% | 0.5841 |
| sq1's band (actual `L*`, sq1 ANISOTROPIC dilation, r=1) | no | 369,414 | 0.8670 | +0.04388 DEAD | −7.09% | 0.6580 |
| **CORRECTED hard band** (actual `L*`, gp1 dilation, r=1) | no | 331,824 | 0.8333 | +0.02670 DEAD | −4.31% | 0.6149 |
| SOFT model on `L*` (d × own-class × edge) | no | 327,405 | 1.0000 | −0.01509 **LIVE** | +2.44% | 0.5056 |
| **SOFT model, RECEIVER-LEGAL** (decoded RGB only) | **yes** | 544,499 | 1.0000 | **+0.12946 DEAD** | −20.92% | **0.8409** |
| ORACLE: frozen SegNet margin (illegal ceiling) | no | 300,288 | 1.0000 | −0.03315 **LIVE** | +5.36% | 0.4637 |
| ORACLE: frozen margin × d(`L*` boundary) (illegal ceiling) | no | 289,746 | 1.0000 | −0.04017 **LIVE** | +6.49% | 0.4474 |

**Reading it:** every live rung is illegal; the only legal rung is dead by 20.9% of gap. The
oracle rows are given every benefit — their payload is charged at the *`L*`-conditioned* 0.2633
bits/flip, which is itself illegal; the honest legal payload is **1.5421 bits/flip** (5.9×
worse), because a receiver with no label field must be told the target class outright rather
than a correction to a class it already knows.

Note also that the **soft model beats the hard band on the very same information**: 327,405 B
at full capture vs 331,824 B at 83.3% capture. Thresholding at a radius was never the right
use of a ranking, and that alone is worth 0.109 in break-even eta.

---

## §4 THE ORDERING BREAK-EVEN — the charter's job 2, answered

The charter asked what an ordering model must achieve for the free row **+ ordering** to beat
A2's −0.039 S, net of the model's own bytes. Priced from the legal floor (the only honest
starting point), with the frozen scorer's own margin as the ceiling no student can exceed:

| quantity | value |
|---|---:|
| legal floor | 544,499 B |
| affordable at sq1's measured eta 0.5406 | 350,068 B |
| **saving required to reach net = 0** | **194,431 B** |
| saving required to match gp1's A2 (−0.039 S) | 253,002 B |
| oracle ceiling: frozen SegNet margin × d(`L*`) | 289,746 B |
| **MAX saving any student can ever deliver** | **254,753 B** |
| **fraction of that gap needed to BREAK EVEN** | **76.3%** |
| fraction needed to match A2 | **99.3%** |
| extra mutual information required to break even | 0.013186 bits/field-px = 3.0581 bits/flip |
| gp1's published student ceiling | 106,954 B — **2.38× understated** |

**This is the number the charter asked for, and it is severe.** A student must reproduce
**76.3%** of the entire measured gap between "decoded RGB alone" and "the frozen SegNet's own
margin field" — *and cost nothing* — merely to stop losing score. To reach the −0.039 S the
charter called LIVE it must capture **99.3%**, i.e. be an essentially lossless, essentially
free replica of the scorer's confidence. gp1's estimate that the student needed to capture
**0.04%–14%** was computed as the gap between two hard bands both drawn on a near-GT field; it
is not the same quantity.

**The rung is not refuted — it is specified, and the specification is hard.** Its own ceiling
(254,753 B) does exceed the break-even requirement (194,431 B), so it is not impossible; it is
a 76.3%-fidelity-at-zero-cost problem, which is a different research program from "a 1 KB
Rudin head, 104× under the ceiling".

---

## §5 The largest carried assumption, stated rather than buried

**`eta` is transferred, not re-measured.** sq1 measured `eta = 0.5406` (pose-neutral, P7) on
**n=32**, on **its own anisotropic r=1 hard band**, with the solved-paint realizer. Every net in
§3 multiplies by it. Two distinct risks:

1. **Sample.** sq1's 32 pairs were stratified-systematic and measured **0.9973** of population
   mean flips/pair, so on the seg governing quantity the subset is representative (`m88`
   satisfied; a prefix would have been 0.9160). This risk is small and is the one the charter
   worried about most.
2. **Object — this is the real one.** A soft full-field model addresses **every** flip,
   including the **16.67%** that lie beyond r=1 of the `L*` boundary. sq1's realizer was only
   ever measured *on* the band. sq1's own v0 locality curve shows eta is strongly radius
   dependent. If the realizer does nothing for off-boundary flips, the effective eta is
   `0.5406 × 0.8333 = 0.4505` and **every row in §3 flips to DEAD**.

| eta assumption | rung | effective eta | net S | verdict |
|---|---|---:|---:|:---:|
| 0.5406 uniform over the whole field | LEGAL floor | 0.5406 | +0.12946 | DEAD |
| 0.5406 uniform over the whole field | `L*` soft (illegal) | 0.5406 | −0.01509 | **LIVE** |
| 0.5406 within r=1 (83.33% of flips), **0 beyond** | LEGAL floor | 0.4505 | +0.16831 | DEAD |
| 0.5406 within r=1 (83.33% of flips), **0 beyond** | `L*` soft (illegal) | 0.4505 | +0.02376 | DEAD |
| 0.5406 within r=1, **half** beyond | LEGAL floor | 0.4956 | +0.14889 | DEAD |
| 0.5406 within r=1, **half** beyond | `L*` soft (illegal) | 0.4956 | +0.00433 | DEAD |

**This single unmeasured quantity decides the verdict**, and it is a cheap, pre-registerable
measurement: re-run sq1's solved-paint realizer with the correction set stratified by
distance-to-`L*`-boundary and report eta per bucket. I did not have slot left to run it; it is
the top NEXT-IF-RESUMED item.

**Second-order, checked and non-binding:** the soft rows are ideal-coder costs while the
hard-band rows are measured LZMA. Stressing the soft rows by LZMA's own measured 1.0520
modelling overhead moves the legal floor's break-even eta 0.8409 → 0.8846 (still **DEAD**) and
the `L*` row's 0.5056 → 0.5319 (still **LIVE**, margin thinned from 6.9% to 1.6%). No verdict
changes. The stress is also pessimistic: a *static* arithmetic coder with the shipped table
achieves cross-entropy + O(1) bits, and I measured the held-out cross-entropy directly
(310,964 B held-out vs 310,662 B in-sample for the `L*` model — fitting optimism is **0.10%**,
on a 90-cell table costing ≤180 B to ship).

---

## §6 SPECIFICATION — what would make this live (never a bare kill)

1. **Measure eta off-boundary** (§5). Decides everything. Cheap.
2. **A student is now load-bearing, not optional, and its bar is 76.3% of the legal→oracle gap
   at ~0 bytes** (§4). gp1's *"not economically close; the ladder does not depend on it"* is the
   claim this unit overturns. Before building one, measure the cheap thing first: how much of
   the frozen margin is predictable from decoded RGB *at all*. My legal feature set (distance
   to a thresholded gradient edge × row × gradient magnitude) is a hand-built **floor**, not the
   legal optimum — a real student may sit well above it, and that headroom is unmeasured.
3. **Cheapest single improvement available right now:** the soft coder. On identical
   information it beat the hard band by 0.109 in break-even eta. Any future address on this
   axis should be a per-pixel probability model, never a radius threshold.
4. **Do NOT re-run** any band-local *content* substitution (sq1 measured it anti-productive at
   every radius to 79.3% of the field), and do not re-derive `D`-support privacy / the 22.70%
   blind fraction / the rank-6 yuv6 null — all reproduced upstream.
5. **Scope note.** This prices *gp1's scheme* — per-pixel class corrections. A carrier that
   sends an RGB delta directly, with no class and no label field, is a different object and is
   NOT priced here (`dd1` / `sg3x` own that surface).

---

## §7 Pointer honesty

The exact pointer did **NOT** move: `0.1910828242 [contest-CPU]` UNMOVED. The own-vehicle
frontier is **UNCHANGED at S 0.7910689 @ 353,805 B [macOS-CPU advisory]**. Everything here is
advisory, `score_claim=false`, and **not byte-closed**. A re-priced ladder is a MEANS. No row
here was built into an archive, so by the standing rule this unit has **not** achieved the goal.

## §8 Artifacts

Scripts (committed): `experiments/ddm_ob1_band_reprice_n600.py`,
`experiments/ddm_ob1_ordering_ceiling_n600.py`, `experiments/ddm_ob1_legal_address_n600.py`,
`experiments/ddm_ob1_margin_oracle_n600.py`, `experiments/ddm_ob1_aggregate.py`.
Receipts (SSD, rebuildable from the committed scripts + existing caches):
`/Volumes/VertigoDataTier/pact/ddm_ob1_20260803/` — `ob1_band_reprice_n600.json`,
`ob1_ordering_ceiling_n600.json`, `ob1_legal_address_n600.json`,
`ob1_margin_oracle_n600.json`, `ob1_aggregate.json`, plus the job logs and the resumable
`ob1_margin_state.npz`.
