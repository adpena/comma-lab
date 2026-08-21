# ddm_rv17 — WAVE 2, ROUND 10: **CLEAN — counter 2/3**; my own clean rounds audited under the new one-check law

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · retroactive instrument audit + FEED consistency.

## THE ANSWER, FIRST

**Clean pass. Counter 2/3.** I chose the lens that turns my round-9 law back on my own prior clean
rounds, because a law derived from my failure is worth nothing until it is applied retroactively.

**The audit found one truncation in a clean round — and it did not compound.**

```
round 6  em1 leg-3 read (cut -c1-170)   L129-133 lengths 98/99/99/90/0   none truncated
round 6  sweep-definition read (-c1-160)                                 none truncated
round 5  jg2 denominator read (-c1-175) L86 len=181                      *** TRUNCATED ***
         what I lost:  "elled** |"      — the tail of the word "modelled"
         what carried the conclusion: L18, len=83, UNTRUNCATED, full text
```

**That is the structural difference from round 8, and it is the whole point of the new law.** In
round 5 the two citations did **not** share a compromised instrument: one read was truncated and
immaterial, the other was complete and load-bearing. In round 8 all three checks shared *both*
faults — the wrong line target and the truncating display — so three reports were one observation.

**Round 6's instruments were genuinely diverse**: `grep -rl` (filename count), `shasum`, `find | wc -l`,
and a `sed` read — four instrument *types* answering four different questions. Independence there was
real, not asserted.

---

## THE FEED — accurate; incomplete, not false

**The `928-char` figure is right, and I nearly called it stale.** I measured line 375 at **1,274**
characters this round and the FEED says 928 — so I checked the state it describes rather than
assuming:

```
at b5f1f0efbe (the round-8 state my instruments failed on) : len =   928
now, post-cure 2d96f65393                                   : len = 1,274
```

The FEED is a correct historical snapshot. Had I flagged it on today's measurement I would have
committed the wave's own genus — a constant compared against the wrong regime — inside the round
auditing that genus.

**On your question — false-as-written, or merely incomplete? Merely incomplete.** One sentence pair
is worth naming for the successor entry:

> *"Round-8 SEAL REFUSED on a claim the artifact refutes … **Cure anyway** (2d96f65393)"*

Read cold, that says the refusal was unfounded and the cure a courtesy. What round 9 established is
that the refusal was **half** right: the annotation clause was false, the substantive clause was
true, and **the cure was owed, not granted**. Nothing in the FEED is false — it predates the ruling
that produced the split — but a reader arriving later would misweigh who was right about what. That
is a one-line successor entry when the wave seals, not a finding: an append-only record that lacks a
ruling written after it is complete for its moment.

**The law statement is consistent with my sharper form, and the two are complementary rather than
competing:**

| FEED | *"a TERMINATION needs verifying against its artifact exactly as a cure does — and so does the VERIFIER"* | verification **depth** |
| mine (round 9) | *"three checks that share an instrument are one check"* | verification **independence** |

The FEED's trailing clause — *"and so does the VERIFIER"* — anticipates precisely what round 9 did to
me. It is the more general statement; mine is the mechanism by which a verifier fails. Both belong.

## LEDGER — unchanged and complete

All fourteen rows still terminate: nine cure-verified (F3, F7, F8, F9, F10, F11, F12, F13, F14),
two adjudicated-and-cited (F5's 61-bit corrector gap, F6's manifest authority), one carried by name
(F1), F2 carried by name, and F4 now cured at source with its annotation intact. No termination was
disturbed by `2d96f65393` — the annotation is retained, and the ERRATUM + ADDENDA remain the
derivation record.

## COUNTER

**2 / 3.** One clean round closes wave 2.

The useful result is the negative one: **applying the new law retroactively did not overturn a
single clean round.** That is not luck. Rounds 5 and 6 verified different claims with different
instrument types, so a fault in one could not silently propagate into the others — which is exactly
the property round 8 lacked. The law is now load-bearing in both directions: it convicted round 8 and
it acquitted rounds 5 and 6, on the same evidence standard.

Eleven corrected outputs stands unchanged; nothing this round added to it.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round10_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
