# ddm_rv17 — WAVE 3, ROUND 2: **CLEAN — counter 1/3**; F6 gets a gate, not a stop order

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · memo-cure verification + one adjudication.

## THE ANSWER, FIRST

**Clean pass. Counter 1/3.** All four memo cures landed; no settled surface was disturbed.

**And my line-local instrument nearly produced a false absence claim for the second time in this
campaign.** My marker regex reported L60 UNMARKED. Before saying so I read the neighbourhood — and
the marker is at **L58 (182 chars)**, the *head* of numbered item 6:

```
L58  6. **The row is LIVE.** *(SUPERSEDED BY §R — the build ran and REFUSED this row at 16.36× the bar.)*
L60     → net −2.489e-05 S = 7.11× the admission bar.   ← a wrapped continuation of L58's sentence
```

The cure is correctly placed — a reader meets the marker *before* the number. My regex was line-local
against a wrapped item. **The round-8 lesson working:** the context read is what stopped the claim,
and I am recording the near-miss rather than only the verdict.

---

## THE FOUR CURES — all verified

| finding | cure | verified |
|---|---|---|
| **W3-F1** | L10 (247 ch, was 103) now carries *"(PROJECTION — price transferred across the admission cut; §R6's named caveat applies and the number is not quotable before a real re-encode)"* | ✓ — and it quotes the gate verbatim, so the headline now enforces the memo's own rule instead of breaking it |
| **W3-F4** | all four sites marked: L58 item-head *"SUPERSEDED BY §R"*; L254 (76→132); L358 (180→**287**, now *"ran; see §R"* in place of *"the build below"*); L493 (84→212) | ✓ |
| **W3-F5** | L557 *"(commensurability note … 0.3867 is a 38-pair MARGINAL, jg1's 0.390 a single-pair AVERAGE…"* and L559 *"The refusal rests on marginal-vs-threshold, which is the right comparison, and 0.3867 is the first…"* | ✓ — including the undersell point I raised |
| **round-1c provenance** | L499 *"(Provenance honesty … the 31m57s precedence rests on file mtime, which is writable, and the JSON is not in git — a good case for…"* | ✓ |

**No settled surface disturbed.** The refusal arithmetic and the 86/86 custody were untouched; every
edit added a marker or a note. §8's row change from *"the build below"* to *"ran; see §R"* is the
sharpest of the seven — it converts a forward pointer into a backward one, which is exactly what a
harvest table owes a downstream reader.

## THE F6 ADJUDICATION — **gate, not stop. No stop-and-fix order.**

You named F6 as the stop-order candidate. My ruling is **no stop**, and the reasoning is that the
danger F6 describes is *latent, not live*:

1. **The literal is correct.** 113,847 is corroborated twice independently of the tool —
   `ddm_fs2:227` records the re-encoder emitting **113,847 B, sha `b9243abd2e38f9ae…`,
   `byte_identical: true`**, and `ARCHIVE_MANIFEST.json:27` carries the same value. The refusal is not
   resting on a wrong number; it is resting on a right number that nothing *checks*.
2. **A projection cannot move the pointer.** Whatever the mirror returns is a projection until a
   byte-closed exact row exists. Under the standing rules no unreceipted baseline can underwrite a
   frontier claim, because the claim itself is inadmissible without exact-eval authority.
3. **A mid-flight edit is the more expensive risk.** Four shards are executing. This campaign's own
   evidence is that every finding from wave-2 round 12 onward lived in *newly landed* code —
   injecting an edit into a running working set trades a latent, corroborated-correct literal for a
   fresh unreviewed surface.

**The gate instead:** *the mirror's verdict is not admissible until its baseline is receipted with
its sha.* That costs the run nothing, it is exactly the cure F6 asks for, and it binds at the moment
the number would actually be consumed. If the mirror lands without a receipted baseline, that is a
finding at the mirror review, and I will treat it as one.

**On the routing generally — I agree with it, and the F2 handling is the part worth endorsing.**
Leaving the shipped round-1 receipt's mislabel as a *memo note, never an edit* is correct: the
receipt is append-only, and editing a shipped receipt to fix its own label would be a worse defect
than the label. Fixing the emitter before the mirror's receipt is written is the right sequence.
Pre-registering F8's addition-vs-removal alternative into the mirror's falsifier record is better
than a cure — it converts my arm's interpretive objection into something the measurement must
*distinguish*, which is the only way an alternative reading ever gets settled.

---

## COUNTER

**1 / 3.** Two more clean rounds close wave 3.

Worth noting once: the cure batch answered the *defect* in every case rather than the finding text.
W3-F1 did not merely add a caveat — it quoted the memo's own non-quotable gate into the headline, so
the surface that breached the rule now states it. §8 did not merely get a marker — its forward
pointer became a backward one. That is the wave-2 lesson (*cure the defect, not the finding*) applied
without being asked.

Instrument findings verify at the mirror landing, under the gate above.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave3_round2_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
