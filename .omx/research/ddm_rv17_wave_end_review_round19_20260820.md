# ddm_rv17 — wave-end adversarial review, ROUND 19: **CLEAN PASS — counter 2/3**; my own instrument was wrong, my conclusion was not

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · nineteenth sibling of `ddm_rv17_wave_end_review_round1-18_20260820.md`.

## THE ANSWER, FIRST

**Clean pass. Counter 2 / 3.** One more seals #1157.

You were right to send my four round-18 adjudications back as claims. One of them was **measured with
a broken instrument** — and re-measuring it correctly is the substance of this round.

**The `{1,12}` sweep, redone properly.** My round-18 grep matched `\.[^:]{13,}:\d` starting from the
*first* dot, so it caught the path segment `.github/workflows/eval.yml` and reported two "hits" that
were artifacts of my own pattern. The correct instrument extracts the extension as the characters
after the **last** dot before the colon:

```
json  len=4  ×28      md  len=2  ×123      sh  len=2  ×14
yml   len=3  ×4       py  len=2  ×8        txt len=3  ×4
MAX EXTENSION LENGTH IN THE ENTIRE CORPUS: 4
```

`{1,12}` is unreachable by a factor of three, and every extension is word-class. **Right conclusion,
wrong instrument, now correctly measured** — which is a distinction worth making explicitly, because
a conclusion that happens to be right does not retroactively validate the method that reached it.

**The other three adjudications hold, re-derived:**

| claim | re-verified how | result |
|---|---|---|
| doc-suffix emptiness | clean instrument, **text files only** (`file … \| grep text`), excluding the `archive.zip` binary artifact that muddied round 18 | **none** |
| `._` exclusions structural | `find` both trees at depth 1 | **0** entries — no real file is affected |
| rank-3 is fixed history | read the chain's own declarations, not the constant | unsuffixed receipt carries **no** `supplements`; R4 declares it supplements the unsuffixed one — the chain asserts its own origin |

---

## THE TWO FRESH ANGLES

**Rows-vs-lines — CLEAN, and better than clean.** No packet document states a line count for
`MANIFEST.sha256` at all, so none can state the wrong one of 36 / 52 / 68. And a second measurement
strengthens my round-18 note materially: **no document cites `MANIFEST.sha256` by line number
anywhere** — zero `` `MANIFEST.sha256:N` `` tokens in either tree. The resolution asymmetry I filed
therefore has not merely an unmet flip condition but **zero live exposure**: there is currently no
citation of any kind that could exercise the prep-vs-frozen resolution path.

**Receipt prose as load-bearing documentation — CLEAN.** I treated the `reason` fields as claims and
checked their falsifiable content against my own independent measurements. R15's reason states the
four formerly-invisible tokens as *GPU_ROUTING_VARIANTS :17+:30, README :30, README_PUBLIC :30* —
which matches my round-17 measurement exactly, including the distribution (`:30` three times, `:17`
once). R14's eight untracked docs matched my round-16 list name-for-name; R12's twelve-doc universe
and R14's twenty-two shas I have re-derived directly. **No factual error in any receipt's prose.**

## THE SEQUENCING CALL — **I endorse it, and more strongly than when I wrote the note**

You asked me to dispute it if I disagreed. I do not, for three reasons and one new measurement:

1. **It matches my own adjudication.** I ruled the direction conservative for all three pairs: a
   loud false failure, never a silent pass. Reversing that judgment now because it is inconvenient
   to leave open would be the inconsistency, not the discipline.
2. **The new measurement makes it stronger.** Zero citations by line number to any prep-published
   file exist today. The path is not merely safe in direction — it is currently unexercised.
3. **This wave's own evidence argues against it.** Every finding from round 12 onward was in
   *newly landed guard code*. Changing the resolution path mid-seal-chain would inject new code into
   the very thing being sealed, trading a loud, unexercised false-failure class for a fresh
   unreviewed surface. That is a bad exchange at 2/3.

Filed as #1172 with my exact flip condition as the fire trigger is the right disposition.

## STANDING SUBSTANCE — **CLEAN**

```
archive df7fd266e1b7488c… / 180,456 B · S 0.14827847122030852 · pointer match · 36 OK
chain rc=0 (22 shas / 3 derived pairs, R15) · citations rc=0 (27 / 9 / 0 / 133)
gen6 frozen · #1111 operator-HELD · packet and receipts unchanged · frontier arms on separate surfaces
```

---

## COUNTER

**2 / 3.** One clean round from sealing #1157.

The useful output of this round is methodological rather than a defect. Sending my adjudications
back as claims caught a real flaw — not in the answer, but in how the answer was reached — and that
is the third distinct way this wave has demonstrated the same principle: a review arm's output is a
claim. My round-2 prescription was wrong on execution, my round-11 rule would have breached
disclosure hygiene, my round-14 assessment mis-called an active gap as latent, and now my round-18
sweep reached a correct verdict through an instrument that was measuring the wrong substring. Four
different failure modes, one lesson: **re-derive, including from me.**

What has not changed in nineteen rounds is the thing that matters. `S = 0.14827847122030852`,
recomputed once more today from the frozen archive's own bytes; 36/36 rows; both guards green; and
no round has found a wrong score, a wrong pin, a wrong digest, a mis-scoped receipt, or an
unverifiable archive claim.

Round 20, if it is the sealing round, should re-derive rather than inherit — including this memo,
and including the four adjudications I have now verified twice.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round19_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
