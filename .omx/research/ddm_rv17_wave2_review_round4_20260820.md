# ddm_rv17 — WAVE 2, ROUND 4: W2-F12 cured at the defect; W2-F13's cure carries a denominator mix; counter RESETS to 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · cure-verification round.

## THE ANSWER, FIRST

**Counter resets to 0/3 — one MED finding, and it is inside the corrected range itself.**

**W2-F12 is cured at the defect, not the finding** — exactly what I asked for and better than I
specified. All three median-derived surfaces now carry the mean, the §3 table's `3.97×` cell sits
under a column explicitly headed *"over the median budget"* (the honest asymmetry), and the
reactivation trigger was upgraded unprompted to name **both** budgets: *"≤1.429 B/pair (median) or
≤2.909…"*. That last change was not in my finding and it closes the same defect one step further out.

**W2-F13's four consumer surfaces all landed** — the MEMORY.md hook, the topic file's frontmatter
*and* body, and an fs2 ERRATUM section. The na10:562 consumption class is closed at every surface I
named.

**But the corrected range mixes two denominators**, and the erratum is what makes that visible:

```
against 4.718 — which the SAME erratum declares is jg3's LogitPrice RANKER, not a price:
    jg5 3.8373/4.718 = 0.8133      jg2 4.1379/4.718 = 0.8770      jg3 3.6471/4.718 = 0.7730

against 4.1379 — jg3's ACTUAL flat price:
    the quoted fourth figure 0.922  (= 1/1.0850, the 8.50% overcharge)

published RANGE "0.77–0.92"  =  three ranker-based ratios + one price-based ratio
fully re-based on 4.1379     =  jg5 0.9274 · jg2 1.0000 · jg3 0.8814   →  ~0.88–1.00
```

---

## RV17-W2-F14 — MED — the cured range is three ratios against a discredited denominator plus one against the corrected one

The erratum's central correction is that **4.718 was never a price** — it is jg3's `LogitPrice`
ranker, per jg3's own docstring — and that the actual flat price is **4.1379**. That correction is
right, and it is the reason the fourth figure (0.922) exists at all: 0.922 is the realised cost
re-based on the *corrected* denominator.

The three original figures were not re-based. `0.813`, `0.877` and `0.773` are all
`realised ÷ 4.718` — ratios against the very quantity the erratum just disqualified. Publishing them
inside one range with `0.922` presents four numbers as comparable when three answer *"how does the
realised cost compare to a ranker?"* and one answers *"how does it compare to the price?"*

**The consequence is material, not cosmetic.** Re-based consistently, the away-trust range is
**~0.88–1.00** — the model over-charges by at most ~12%, and on jg2 it is exact. The published
**0.77–0.92** says it over-charges by up to 23% and never less than 8%. Those are different
statements about instrument trust, and the second is the one now sitting in the session-loaded hook.

**Direction-dependence survives either way** — 0.09× toward versus 0.88–1.00× away is still ~10×, so
no verdict moves. What moves is the number a future arm will price a lever with.

**This is the standing genus in its subtlest form yet.** The erratum corrected the *attribution* of
the constant and the *identity* of the denominator, then carried the values derived from the old
denominator into the corrected range unchanged. A correction that reaches the prose but not the
values derived under the old premise is the same shape as a heading that keeps a superseded number —
and it is genuinely hard to see, because the cure looks complete and its own text is transparent
about each figure's provenance.

**CURE:** re-base all four on 4.1379 and publish `~0.88–1.00× away (jg5 0.927 · jg2 1.000 · jg3
0.881, `delta_trustworthy: false`)`, or, if the ranker-based figures are wanted for continuity, print
them as a separate labelled series rather than as members of one range. The hook must carry whichever
range is the *price*-based one.

## What verified clean

**W2-F12, all three surfaces** — ANSWER-FIRST *"3.97× the MEDIAN budget, 1.95× the MEAN budget that
governs this blanket move"*; the under-table line *"Against the MEAN budget (2.909 B/pair — the one
that governs the blanket move): qs2 1.95×, jg5 3.14×, rc4 4.41× — every measured edit encoding loses
on the mean budget too"*; §7 carrying both. The table column label makes the retained `3.97×` honest
rather than stale.

**W2-F13's surface coverage** — MEMORY.md:84 now reads *"0.77–0.92× away (RANGE, jg5 own 0.813) /
0.09× TOWARD argmax"*; the topic file carries the range in **both** description frontmatter and body
with all four per-instrument figures and the ranker-not-price correction; the fs2 memo carries a
dated ERRATUM section scoped `verdict_scope: instance`. Four surfaces, four landings — the coverage
question is answered even though the range's content is not yet right.

**Standing caveats correctly carried, unchanged:** em1's `0/46` reachability leg remains
never-re-derived, and the equations-leg registration still rides a future landing — now correctly
noted as owing the **range** form. W2-F14 changes which range it owes.

---

## COUNTER

**0 / 3 — reset.**

Round 3 found that a cure scoped to the findings it was handed is not a cure scoped to the defect.
Round 4 finds the next layer: **a cure can be scoped to the defect and still carry forward values
computed under the premise it just refuted.** W2-F13's cure reached every consumer surface — that
part is complete and I want it credited — while the numbers it published on those surfaces inherit a
denominator the same document disqualifies two lines earlier.

Seven of my own outputs have now been corrected in this campaign, and the reason I found this one is
that the erratum did its job: by naming 4.718 a ranker, it gave me the tool to check what else was
divided by it.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round4_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
