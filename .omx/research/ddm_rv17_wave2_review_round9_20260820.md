# ddm_rv17 — WAVE 2, ROUND 9: my round-8 headline was FALSE; the reset stands on its true clause; counter 1/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · self-refutation + cure verification.

## THE ANSWER, FIRST

**You are right, and I verified it at source rather than accepting it.** My round-8 claim that
`fs2:373` carried no scope annotation is **FALSE**, and the annotation was in committed HEAD at my
read time:

```
L373  len=  47   | row | why it survives | reactivation number |      ← the table HEADER
L375  len=1274   | **The recapture law itself** | …                   ← the law ROW
grep -n verdict_scope        → matches 375
git show b5f1f0efbe:… | sed -n '375p' | grep -c 'verdict_scope: formulation'  → 1
```

**I can explain exactly how I got it wrong, and the explanation is worse than the error.** Two
instrument faults compounded:

1. **Wrong line target.** I read and grepped **373** — the header. The row is **375**.
2. **Truncation.** My range read of 372–375 piped through `cut -c1-400` on a line that is **1,274
   characters**. The annotation lives at the end. So even where the row *was* in my output, the
   annotation was chopped off.

My round-8 memo said *"I checked three ways before saying so."* **All three shared both faults** —
the narrow grep and the broad grep both targeted 373, and the direct read was truncated. That was
not three checks. It was **one check run three times with the same blind spot**, reported as
independent confirmation.

And the memo that did this **named the genus in its own text**: *"an empty grep is not a measurement
of absence unless the instrument is known to work."* I wrote that about a different regex on the
same line, then treated three compromised instruments as corroboration.

> **New law: three checks that share an instrument are one check.** Independence is a property of
> the instruments, not of the count.

---

## THE RULING ON MY OWN COUNTER

You took no position; here is mine, and I have split it rather than resolved it in either direction.

**The round-8 HEADLINE is WITHDRAWN.** *"A written, cited, false ending"* was itself false as to F4.
The termination was written, cited, and **true**. That sentence should not stand in the record, and
I am not softening it — it was the load-bearing claim of the memo and it was wrong.

**The round-8 RESET STANDS, on the true clause only.** The row's main text handed a reader
`against a 4.718 model (0.877x)` and its reactivation cell handed out `~0.88x` bare — a mis-sourced
constant, on the surface that *generalizes* it, with no error signal. Under the bar I have applied
since round 4 of wave 1 — *a silent wrong value is a finding; a loud failure is a note* — that is a
finding, and the same class I have called MED five times in this wave.

I will not vacate a legitimate finding to compensate for having been wrong about the other half.
That would trade integrity for symmetry, and the count is worth less than either.

## THE CURE — verified at source

`2d96f65393` restates the generalization surface directly:

```
3.8373 = 0.927× of the ACTUAL flat price 4.1379   (Series B, MEASURED, 8.50% overcharge)
4.718  = jg3's LogitPrice RANKER — ordering-only, not a price, per its own docstring
0.877× survives only as Series A (instrument ordering, never a trust factor)
reactivation cell: overcharges ≤~12% away (real/price ~0.93x), overstates ~11x toward (~0.09x)
ERRATUM + ADDENDA retained as the derivation record; the annotation retained
```

This is the last uncorrected instance of W2-F3 closed, and closed on the surface where it mattered
most — the row whose entire function is to hand the constant to the next arm.

## COUNTER

**1 / 3.**

The reviewed set is clean this round. The defect was mine, and a reviewer's error does not count
against the artifacts — but it does go on my ledger. **Eleven corrected outputs**, and this is the
first that was *confidently asserted against a correct artifact* rather than an over-reach or an
omission. That is a worse category and I am recording it as such.

What makes it worth the round: the failure mode is now named and cheap to avoid. `awk 'NR==n'` with
a length check costs nothing, and *"how long is the line I am reading?"* is a question I will not
skip again on a table row. The instrument that failed me was the one I had just finished warning
about.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round9_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
