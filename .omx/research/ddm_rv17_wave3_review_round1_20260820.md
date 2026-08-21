# ddm_rv17 — WAVE 3, ROUND 1: one MED finding on the projection's consumer surface; counter 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · opening round of the fs3 wave.

## THE ANSWER, FIRST

**Counter 0/3 — one MED finding.**

**Scope honesty:** I completed items **4** (law scoping) and **7** (mirror caveat) myself. Items
**1, 2, 3, 5, 6** are with two supervised arms still in flight and are **not certified by this
round**. My wave-2 instrument laws bound me throughout: every absence-type judgement below is
length-checked with `awk`, never read through a truncating `cut`.

**RV17-W3-F1 (MED) — the 12.72× projection is caveated at the table and bare at the headline, and
the memo's own gate forbids exactly that.**

```
L603 (len 101, table)   | **5.5722 (MEASURED marginal)** | … | **12.72** |
L606-610                **"Named caveat, not buried:"** measured just BELOW the cut, applied just
                        ABOVE it … "it **is** a transfer, of exactly the class this memo has now
                        caught twice, and it must be closed by a real re-encode of a tightened
                        field **before any of it is quoted**."

L10 (len 103, ANSWER-FIRST)
                        "opens a mirror worth more than the reopen was: **§R6, a −4.45e-05
                        tightening at 12.72× the bar.**"          ← no caveat, no marker
```

The table's caveat is **exemplary** — adjacent, explicitly labelled *not buried*, naming the transfer
class and its own precedent. Nothing about it is buried or hedged. But the ANSWER-FIRST block states
the projection with no marker at all, and the memo's own gate at L609 says the number must not be
quoted before a real re-encode. **The headline quotes it.**

That matters because of what this campaign has measured about consumer surfaces: the ANSWER-FIRST
block is the highest-traffic surface in these memos — it is the block that travels into FEEDs, task
ledgers, and coordinator summaries, stripped of its section. This is the W2-F13 genus exactly: the
caveat reached the document and not the surface that moves.

**CURE:** one clause at L10 — *"…a −4.45e-05 tightening at 12.72× the bar (PROJECTION, price
transferred across the cut; see §R6's named caveat)."* The table needs nothing.

---

## ITEM 4 — the law's scope — **honestly scoped; one presentational note, not a finding**

I went in expecting a scope-inflation finding and did not get one.

**The title does not overclaim** — and I checked it rather than assuming:

```
L1 (len 83): "# ddm_fs3 — the reopen was real on the average price and dies on the marginal one"
```

That describes what happened to *this* reopen. It is not a general law statement, which is what I
was looking for. **Suspicion disproved.**

**The per-claim scope block is thorough** (L673–677): §1–§5 FORMULATION *(jg3-class
edit-configuration re-selection under real prices, on the rc2-lineage body)* · §4's permutation tests
INSTANCE *(these 38 pairs)* · §6 FORMULATION · §7 FORMULATION · and the closing line **"None is a
family kill. None is a score."**

**The note.** The law fuses two clauses of different kinds:

| clause | kind | scope needed |
|---|---|---|
| *"the AVERAGE price of a set is not the price of its marginal member"* | **arithmetic** — true by the definition of average vs marginal; unfalsifiable | none |
| *"yield AND price degrade at the margin, in the same direction"* | **empirical** — measured on one edit family, one body, one greedy ranking | FORMULATION |

Stated together as one "law," the empirical half can borrow the tautology's certainty when quoted.
The memo's section scoping mitigates this and its title does not overreach, so I am recording it as a
note rather than a finding — but the honest form is to state the arithmetic as *framing* and the
degradation as the *FORMULATION-scoped measurement*, which is what it is.

**One dependency I am flagging rather than resolving:** the "same direction" generalization leans on
identifying the yield collapse with jg1's 0.390 and jg3's 0.406. Whether those three numbers are
commensurable — same units, level, aggregation — is item 3, which is with an arm. **If they are not,
the empirical clause rests on a single instrument and this note becomes a finding.** I am not
pre-judging it.

---

## COUNTER

**0 / 3.**

The opening shape of this wave differs from wave 2's, and it is worth noting once: wave 2's findings
were all *"the measurement is sound, the generalization is not."* fs3's memo is visibly built by
someone who had read that verdict — it caveats its own projection in a block headed *not buried*, it
scopes every section by claim, it self-corrects its own pose leg, and its title declines to
generalize. The one finding I have is not that a caveat is missing; it is that the caveat did not
reach the surface that travels.

Items 1, 2, 3, 5, 6 remain outstanding with arms in flight.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave3_round1_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
