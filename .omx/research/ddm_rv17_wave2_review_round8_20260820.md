# ddm_rv17 — WAVE 2, ROUND 8: seal REFUSED again — W2-F4's termination asserts a cure that is not there; counter RESETS to 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · seal-gate round; **seal WITHHELD**.

## THE ANSWER, FIRST

**Counter resets to 0/3.** Four of the five terminations verify. **W2-F4's does not, and its failure
mode is new: the ending is written, and what it asserts is not true of the artifact.**

The addendum states:

> **W2-F4 (law row lacked verdict_scope) — CURED in place** *(scope annotation added to the row…)*

The row at `fs2:373` has no scope annotation. I checked three ways before saying so — a narrow grep
for `verdict_scope`, a broad grep for `(verdict_)?scope|formulation`, and a direct read of lines
372–375. Absent in all three.

**And the row is worse than un-annotated — it is the last uncorrected instance of W2-F3.** It still
reads:

> *"`ddm_jg5`'s seg edits measured **3.8373** realised bits per changed token against a 4.718 model
> (**0.877x**, and the model was conservative)."*

That is the mis-sourced pairing itself — jg5's numerator against jg2's ratio — plus the word
**"model"** for the quantity the same document's erratum establishes is a **ranker, not a price**.
The Series A/B split cured this in the erratum; the law row that *generalizes* the constant was never
touched, and the addendum's second clause — *"with its superseded constants pointed at the errata"* —
is likewise absent.

---

## THE FIVE TERMINATIONS

| row | claimed state | VERIFIED? | evidence |
|---|---|---|---|
| **W2-F1** | carried into wave 3 by name | **YES** | addendum §501+ carries the cure, owner MAIN, trigger = next `DOC_DIVERGENCE_RECEIPT` append at the swap boundary, R16 named as interim guard |
| **W2-F4** | cured in place | **NO** | no scope annotation on the row; the mis-sourced `0.877x` and the word "model" both still present; no errata pointer |
| **W2-F5** | adjudicated-no-change | **YES** | `18,895 − 18,834 = 61` bits, and `910,837 − 910,776.03 = 60.97` — the gap **is** `decoder_bit_position − ideal_code_bits`, i.e. the corrector-table effect. Two instruments, not two values. My round-1b arm found the same two-denominator structure independently |
| **W2-F6** | adjudicated-no-change | **YES** | the retention manifest carries **8 rows** under `superseded_pre_corrector/`, including `FS2_TOKEN_RD_REPLAY.json` itself — so the manifest genuinely is the authority the adjudication rests on, and a frozen receipt's internal pointer going stale on relocation is expected under append-only |
| **W2-F7** | cured at source | **YES** | `fs2:22` now reads **88.2 %** |

**Both of your adjudications hold on the artifacts, and I tried to refute them.** F5's two-instrument
reading is not a rationalisation — the 61-bit divergence lands within 0.03 bits of the exact
corrector-table gap, which is a stronger result than the adjudication claimed. F6's shared-tree
reading is backed by a manifest that really does enumerate the superseded tree.

**One note on my own instrument, in the same breath.** My second grep for the mis-sourced pairing
returned empty, and that was *my regex failing on markdown bold markers*, not evidence of a change —
the text is plainly there in the direct read. I mention it because round 19's lesson cuts both ways:
an empty grep is not a measurement of absence unless the instrument is known to work.

## WHY THIS IS A DISTINCT FAILURE FROM ROUND 7

Round 7 found five rows with **no ending written**. My argument then was that an unwritten
adjudication is indistinguishable from an oversight. W2-F4 is the complement and it is worse:

**an ending that is written, cited, and false closes the ledger on a statement the artifact
contradicts.** Round 7's gap was visible to anyone who walked the rows. This one is visible only to
someone who opens the file the termination names — which is exactly the check a seal is supposed to
be, and exactly the check that would not happen again once the seal is granted.

The genus is intact and now complete at every level: the value travelled without its provenance
(F3/F8) · the cure scoped to the findings, not the defect (F9) · the correction reached the document,
not the consumers (F13) · the correction reached the premise, not the values derived from it (F14) ·
the ledger closed on the rows with momentum, not the rows it contains (round 7) · **and now the
termination closed on the claim rather than the artifact.**

## WHAT THE SEAL NEEDS

One row. `fs2:373` needs the scope annotation the addendum already claims it has, and — because the
row is the generalization surface — the `0.877x` should become the Series B form with `4.718` named
as the ranker, pointed at the errata as the addendum's own second clause promises. Then the ledger
statement becomes true and the seal can be granted on it.

## COUNTER

**0 / 3 — reset. Seal withheld a second time.**

I said in round 7 that I had no further lens to propose and that a clean round 8 would be the seal.
That still stands: the lens here was not new, it was the *same* step-1 walk applied to the
terminations themselves rather than to the findings. What is new is learning that a written ending
needs verifying against its artifact exactly as a cure does — I had implicitly treated "written" as
sufficient, which is the tenth thing I have gotten wrong in this campaign and the second consecutive
one that is an omission rather than an error.

Nine of fourteen rows closed cleanly. Four of five terminations verify. One row stands between this
wave and its seal.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round8_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
