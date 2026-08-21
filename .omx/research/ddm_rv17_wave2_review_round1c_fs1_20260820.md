# ddm_rv17 — WAVE 2, ROUND 1c: fs1 — one HIGH finding inverts a CLOSED verdict; counter 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · item 1 of the wave-2 scope, completing rounds 1 + 1b.

## THE ANSWER, FIRST

**Counter 0/3 — one HIGH finding, three MED, and it is the HIGH one that matters: a units misread
turns a CLOSED verdict into a probable gain.**

fs1's arithmetic is otherwise excellent — I re-derived every number in §1–§5 from the retained arrays
without importing the arm's module, and the frozen S reproduces bit-exactly. The payload law is
fully satisfied. The withdrawn estimate was withdrawn honestly and never silently relabeled.

**RV17-W2-F8 (HIGH) — `jg1_resolve_midpoint = 10.5 B/pair` is a coefficient count read as a rate.**
I verified this at source rather than adopting the arm's report. `ddm_na10:562` reads:

> *"jg1's re-solve moves **9–12 already-shipped coefficients**, which up2 measured the Rice stream
> absorbing at +5 B for all 7,200 ≈ **0.83 B/pair**"*

The tool's comment truncates the quote at `"moves 9-12"` — exactly at the word carrying the units —
and enters the midpoint **10.5** into a dict whose docstring reads *"Edit-encoding costs MEASURED on
this vehicle, in archive bytes per edited pair. Each is a real receipt, not a model."* For that
entry the docstring is false: 9–12 is a **count of coefficients**, and na10's own price for the same
family, in the same sentence, is **0.83 B/pair**.

**The verdict inverts.** Re-pricing the memo's own blanket-27 row with its own measured pose legs:

```
encoding B/pair              rate S      net@median     net@mean
qs2 5.667   (memo)        1.0188e-04    +7.9207e-05    +6.3120e-05   LOSS
jg1 10.5    (MISREAD)     1.8877e-04    +1.6610e-04    +1.5001e-04   LOSS
na10 stated 0.83          1.4922e-05    −7.7537e-06    −2.3841e-05   GAIN
```

At the price na10 actually states, the blanket move **nets −2.38e-05 — about 6.8× the −3.5e-6 bar,
a gain.** §3 reads *"CLOSED … at **every** edit encoding measured on this vehicle"* with reactivation
at ≤1.429 B/pair. The universal quantifier fails against the very sentence the constants are drawn
from, and the reactivation trigger may already be met.

**Two things keep this from being a simple reversal, and both need saying.** `ddm_up3` §5 *corrects*
the up2 "Rice absorbs it free" claim — as up2 shipped it the candidate would have cost **48 B**, and
*"a one-coefficient flip already costs +3 B."* So the live price for this family spans **0.08 B/pair**
(48 B over 600 pairs) to **~27–36 B/pair** (3 B × 9–12 coefficients). And na10's own 0.83 is
internally inconsistent: 5 B / 7,200 coefficients over 600 pairs is **0.0083**, not 0.83 — a 100×
slip inside the source line. **Which price governs is precisely the adjudication fs1 never performed,
and it is the single measurement that decides whether §3 stands.**

---

## THREE MED FINDINGS

**RV17-W2-F9 — 3.97× is the wrong statistic for the claim it headlines.** 3.97 = cheapest encoding ÷
**median** credit budget — a statement about the median pair. What fs1 actually prices, and what its
title asserts, is a **blanket move over a population**, whose ΔS is additive and whose break-even is
therefore the **mean**: 2.909 B/pair → **1.95×**. Using the memo's own clamped realized credit gives
2.16 B/pair → 2.63×. The conclusion (net loss at 5.667) survives all three; the **published magnitude
does not**, and the headline, §3 header and commit message all carry the median-derived figure.

**RV17-W2-F10 — scope inflation on exactly the surfaces that get recalled.** The typed labels are
correct and complete (§3 FORMULATION, §4 INSTANCE, §5 FORMULATION, plus two *"not a family kill"*
disclaimers). But the title, the §3 header, ANSWER-FIRST, and **the commit message** all say
**"family"** — and the commit message is asymmetric, printing `FORMULATION` for js6b while omitting
it for the pose actuator. `git log` is what propagates; the body's correction three lines down does
not travel with it. This is the campaign's own
`corrections_land_in_bodies_headlines_keep_the_stale_number` genus.

**RV17-W2-F11 — §4's population control was not applied to §5, and js6b pre-registered exactly it.**
The re-screen applies one global scalar `c` to all 200 rows and never consults the per-pair `c_i` the
instrument already holds. Classified by their own pair's state: **18 of 200 rows sit on pairs that
were never edited in jg5** — no compensation measurement exists — and **one of the two rows admitted
at the median calibration is among them** (pair 517). That is the identical defect §4 refuses for the
pose actuator. js6b's own reopen trigger required *"candidate-specific Pose-vector evidence"*; fs1
supplies a population aggregate from a different edit family. **The REOPEN still stands** — the
strongest row (pair 176) has its own measured compensation land at-or-below base, so it is stronger
than reported — but the "2 rows admit" count is not uniformly measurement-backed.

## VERIFIED CLEAN — and substantial

S reproduces **bit-exactly** (0.14827847122030852); shares 81.0355 / 13.5819 / 5.3826 % match to
stated precision; `25/37545489 = 6.658589531221714e-07` ✓; the bar `3.5e-6 / 6.6586e-07 = 5.2564 →
5.26 B` ✓ (though its −3.5e-6 definition is uncited — it traces to the 8dp report band). The 1.429
median break-even, the full credit distribution, and the blanket-27 nets including the clamp all
re-derive from the arrays. The **withdrawn estimate is clean**: −7.7495e-05 / 22.1× reproduces from
the described wrong method, the withdrawal reason verifies against primaries (0 strictly-positive
credits among 118 dropped), it appears exactly twice and both are labelled withdrawn, and the
replacement is a refusal plus a hypothetical-labelled ceiling — **no silent relabeling**, with 18
tests passing including a synthetic-credit flip guard. The **js6b reopen arithmetic fully reproduces**,
including a positive control at `c=1` that lands on js6b's own retained `5.25553385416666e-07`.
**Payload law fully compliant** — four arrays on disk, digests matching the result JSON, inputs
hashed at read time.

Three LOW carried: `kept_pair_prior_is_admissible_for_unbanked` is a hardcoded `False` a reader will
take as derived; the result digest is path-coupled (content-identical modulo 4 embedded absolute
paths); and the S-split line misrounds the rate term and mixes two instruments' d_pose.

---

## COUNTER

**0 / 3.** Wave 2 round 1 total: **1 HIGH, 6 MED, 6 LOW** across four landings.

The wave's shape is now unmistakable and worth stating once: **every measurement in this batch is
sound; every finding is in what was concluded from it.** fs2's refusal verifies exactly and its
constant is mis-sourced. fs1's arrays re-derive exactly and its cheapest-encoding constant is a
units misread. Round 1 found a receipt writer with no home and an equations leg with no owner. Four
landings, one genus: **the arithmetic is trustworthy and the generalizations are not yet.**

W2-F8 is the one that needs an answer before anything cites §3: **which price governs the
carrier-perturbation family — 0.08, 0.83, or ~27–36 B/pair?** Until that is adjudicated on receipts,
"CLOSED at every measured encoding" is not a claim the evidence supports.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round1c_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
