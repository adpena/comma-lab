# ddm_rv17 — WAVE 2, ROUND 6: **CLEAN — counter 2/3**; the em1 caveat is PAID, and the denominator is conservative

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · converts the one standing caveat I had carried twice.

## THE ANSWER, FIRST

**Clean pass. Counter 2/3.** The caveat I carried through two rounds without converting is now
**paid as a verification, not a finding** — and you were right that it should not have ridden into a
sealed wave as trust.

**All three em1 legs verify from the artifacts:**

```
LEG 1 — reachability, MY OWN sweep over the frozen tree:
  _quantize_per_tensor_int8_with_fp16_scale   files matching: 0
  as_renderer                                 files matching: 0
  population swept: 46 files                                     → 0/46 REPRODUCES

LEG 2 — payload identity, measured from disk not from any quote:
  cdd4535249002d740db73acebd61dbedb4734b0cca24a0d71213a33057ec385a   == my round-1 measurement
  (28 real arrays: int8 codes + fp16 scales on named SM3R tensors)

LEG 3 — the counterfactual sign: SURVIVES, and it is structural rather than numerical.
```

**The 46 denominator reconciles exactly, and it is conservative:**

```
12 top-level files  +  34 subdirectory files (cpr1/, runtime/)  =  46   real files, ._ excluded
36 MANIFEST runtime rows
difference = 10  =  archive.zip + 9 non-runtime documents
```

**46 is a strict superset of 36**, and that is the right direction for this claim. A reachability
sweep asks *"is this class present anywhere in the shipped packet?"* — so scanning the **broader**
population strengthens a NOT-LIVE finding, where scanning only the 36 manifest rows would have left
the 10 non-runtime files unexamined. This is the inverse of the wave's wrong-denominator genus: the
census rule errs toward completeness, and the finding is stronger for it.

---

## LEG 3 — why the sign survives re-derivation

The counterfactual's sign is not an arithmetic result that could flip on recomputation; it is a
property of the construction:

> `round(w/s16)` is by construction the nearest code **under the scale the decoder will actually
> apply**. Variant A's `round(w/s32)` is nearest under a scale the decoder never sees, so every one
> of those ≤117 flips moves the reconstruction **away** from `w`.

That is sound and it is the strongest of the three legs, because it does not depend on the sweep or
the payload at all. The available direction is variant-A → shipping, and **shipping is already
there** — so the counterfactual cannot be a live improvement regardless of magnitude. A bounded
count (≤117) of moves that are all in the wrong direction is not a lever.

**One provenance note, apt for this wave.** Your message quoted the payload as `cdd44352…`; the
artifact measures `cdd45352…`, matching what I recorded in round 1. That is a slip in the *message*,
not in the artifacts — and I mention it only because this entire wave has been about constants
travelling without their source. Verifying from disk rather than from the quote is the discipline
that caught it, and it cost one command.

## The three-way typing — **agreed, the round-5 receipt is the adequate record**

Not a finding, and I want the reasoning on record rather than just the assent.

The addendum types jg2 as `DERIVED` where my round-5 receipt types it
`DERIVED-with-denominator-verified`. That difference errs **conservative**: it claims less
confidence than the evidence supports, so a reader is never misled in the dangerous direction. And
my round-5 memo already recorded it *"rather than asking for a change"* — reversing that now, one
round later, on the same evidence, would be exactly the flip-flop I would flag in someone else.

The receipt chain is the durable record and it carries the three-way form with its evidence. Churning
a source file for a sub-verdict refinement that is already safe adds edit risk to a wave two rounds
from sealing, for no gain in correctness. **Do not churn it.**

---

## COUNTER

**2 / 3.** One more clean round closes wave 2.

The lesson I take from this round is about my own practice rather than the artifacts. I recorded the
em1 caveat honestly in round 1 and carried it accurately through rounds 3 and 5 — but *carrying a
caveat accurately is not the same as discharging it*, and I let it ride twice on the grounds that its
status had not changed. Its status had not changed because I had not looked. The verification cost
three commands, and had it gone the other way it would have been a HIGH finding sitting inside a wave
I was two rounds from sealing.

A caveat is a debt, and the honest thing is to pay it or convert it — not to re-record it.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round6_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
