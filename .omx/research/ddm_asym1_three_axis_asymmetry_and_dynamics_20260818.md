# The three-axis asymmetry, and what it does to the routing

**Operator, 2026-08-18:** *"Repair and resolve is always possible"* + *"Remember asymmetry as well"*
+ *"And all related dimensions and dynamics."*

All numbers below are DERIVED from the exact score function at the LIVE pointer
(S 0.15771357797660338 @ 179,930 B, d_pose 6.880e-06, gap-to-0.15 = 0.00771). No fitting.

`S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489`

---

## 1. The shape asymmetry: two axes are LINEAR, one is CONCAVE

| axis | marginal | behaviour |
|---|---|---|
| rate | `25/D` = **6.6586e-07 S/byte** | constant, forever |
| seg | `100/(600·512·384)` = **8.4771e-07 S/flip** | constant, forever |
| pose | `5/√(10·d_pose)` = **602.8** at the pointer | **changes with where you stand** |

**1 byte = 0.785 seg flips.** That is the only fixed exchange rate in the problem — and it is
exactly the `0.785 flips/B` breakeven qs5 banked empirically. Independent confirmation that the
banked constant is STRUCTURAL (it falls out of the score function), not an artifact of qs5's object.

## 2. The concavity asymmetry: equal ratios are NOT equal score

- halve `d_pose` → **−0.002429** (29.3% of the pose term)
- double `d_pose` → **+0.003436** (41.4% of the term)

⇒ **a 2× worsening costs 1.41× what a 2× improvement buys.** √ is concave, so it punishes
upward moves harder than it rewards equal-ratio downward moves. This is the structural reason the
uncompensated-edit family kept refusing: it was paying on the expensive side of a concave axis.

## 3. The operating-point asymmetry: the marginal RISES while the total SHRINKS

| d_pose | marginal |
|---|---|
| 6.880e-06 (now) | 602.8 |
| 3.440e-06 (half) | 852.5 |
| 6.880e-07 (tenth) | 1906.2 |

Both statements are true and they are usually confused. Each further unit of `d_pose` is worth
MORE — but there is less of it left, so realized gains DIMINISH. Cite the marginal for *local*
decisions (is this edit worth its pose tax?) and the total for *routing* decisions (how far can
this axis take us?). Using the marginal for routing is the trap.

## 4. The composition dynamic: pose SUBADDS, rate/seg ADD

- 1st halving of `d_pose` buys 0.002429
- 2nd halving buys **0.001718 = 71% of the first**

Rate and seg gains compose at **exactly** face value; two −790 B wins are worth precisely twice
one. Pose wins cannot say that. **Compose rate/seg additively; never sum pose gains.**

## 5. THE ROUTING CONSEQUENCE — the pose-first doctrine needs its counterweight

Pose is **107.5% of the gap** if driven to zero. That is true, and it is why the campaign has
been pose-first. But the concavity means:

| route | what it takes to close the 0.00771 gap ALONE |
|---|---|
| **pose** | `d_pose` 6.880e-06 → 3.376e-08 = **203.8× reduction** |
| **rate** | **11,584 B** = 6.4% of the archive |
| **seg** | **9,099 flips** = 0.0077% of pixels |

**Pose owns the most gap and is the hardest axis to spend it on.** "108% of the gap" is a
statement about the INTEGRAL; 203.8× is the statement about the WORK. Both belong in every
routing decision from here.

**Where sa3 sits:** its −790 B = **5.260286e-04 S** ≡ a **1.14× pose reduction** ≡ **621 seg
flips** — and unlike pose, a second −790 B buys the same again.

## 6. The repair asymmetry — *"repair and resolve is always possible"*, with its measured caveat

Repair on this campaign has a sharp, measured split:

- **EXACT-SOLVE repair WORKS.** qs5's in-compile Schur compensation drove `d_pose` BELOW base
  (−3.814e-07, repeat identical), fully curing qs4's +2.396e-04 disaster.
- **FITTED repair FAILS.** pk4's linear frame-0 overlays were LOPO-positive in the modeled space
  and heldout NEGATIVE-or-zero in reality at all three rungs (deterministic instrument,
  repeat-noise 0.0 ⇒ the negatives are signal). pk3 reproduced it: 23/23 in-sample = 0/23 LOO.

⇒ Repair is always available — **through a solve, not through a fit.** That is the operative
form of the operator's principle here, and it is why iv1's refusal routes to compensation rather
than to closure. (Repairing iv1's own candidate does not rescue iv1: with pose taxed to zero it
still carries +2.0e-05 seg and +1,545 B ⇒ +1.05e-03, refused on what it FAILED TO BUY, not on
what it broke. The repair principle is real; it just does not make a candidate that buys nothing
into a winner.)

## 7. Consequences adopted

1. **Rate is the currently-honest axis** — linear, additive, and needing only 6.4% of the archive.
   sa3's −790 B is live and sits there.
2. **Every pose claim carries its reduction FACTOR**, never just its ΔS — because the ΔS depends
   on the operating point and the factor does not.
3. **Never sum pose gains.** Recompute `√(10·d_pose)` at the composed `d_pose`.
4. **Repair routes to exact solve.** A fitted corrector needs a heldout leg before it is believed
   (pk4's law).
5. The 191-flips-per-edited-pair break-even (ddm_iv1 §9) is the *local* form of §2 — the same
   concavity, priced per edit.

`verdict_scope`: **DERIVED-EXACT** from the score function at the stated operating point. The
numbers move with `d_pose`; the SHAPE (linear/linear/concave, add/add/subadd) does not.
