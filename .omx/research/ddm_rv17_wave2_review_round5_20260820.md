# ddm_rv17 — WAVE 2, ROUND 5: **CLEAN — counter 1/3**; your divergence was right, and the toward-leg re-basing must NOT happen

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · cure-verification + two adjudications.

## THE ANSWER, FIRST

**Clean pass. Counter 1/3.** The landed cure is correct on all three surfaces. My two substantive
outputs are a concession and a refusal.

**1. Your divergence was right, and my round-4 headline over-reached.** You asked whether I had
verified the shared-4.718 denominator at jg2's and jg3's own sources. **Checked: I verified jg2's,
never jg3's.**

```
jg2  VERIFIED AT SOURCE   — its memo, twice: "That is 0.877x jg1's modelled 4.718"
                             and "+30 B = 4.1379 bits/token = 0.877x modelled"
jg3  NEVER VERIFIED       — its memo states only `bits per changed token 3.6471`.
                             No ratio. No denominator. The 0.773 was my arm's reconstruction.
```

So my `~0.88–1.00` headline published a **lower bound resting on an assumption I never checked** —
immediately after finding a range whose values rested on a denominator nobody had checked. That is
the genus one level out, exactly as you said, and the measured/derived split is the honest form.

**One refinement in your favour and one against.** The cure types jg2 and jg3 identically as DERIVED;
the honest typing is **three-way** — jg5 `0.927` MEASURED · jg2 `1.000` DERIVED-with-denominator-**verified**
· jg3 `0.881` DERIVED-with-denominator-**unverified**. Your two-way split under-states jg2, which errs
conservative, so I am recording it rather than asking for a change.

**2. Do NOT re-base the toward leg. The proposed ×1.1402 would introduce the error it means to fix.**

The two legs do not divide by the same *kind* of thing, so putting them on "the same denominator"
is not available:

```
AWAY   = realised bits/token ÷ modelled bits/token      ← per-token price, seg-edit family
         this is where 4.718-vs-4.1379 lives

TOWARD = 1,022 measured bytes ÷ 11,716.7 modelled bytes = 0.0872
         ← whole-stream byte totals over the DROP ladder. There is no 4.718 in it.
```

fs2 states it plainly: *"The measured saving is 1,022 archive bytes, not 11,716.7 — 8.72% of the
first-order model."* Multiplying that by `4.718/4.1379` would import a **per-token price correction
for the seg-edit family** into a **whole-stream byte ratio for the drop ladder** — a different
granularity over a different family. That is the cross-regime constant transfer this campaign names,
and it would be a fresh instance of it created by a cure.

The ~10× direction ratio needs no rescue: `0.93 away` versus `0.087 toward` is **10.7×**, and it is
sound precisely *because* each leg is priced against its own model rather than forced onto a shared
constant.

---

## THE LANDED CURE — verified on all three surfaces

```
MEMORY.md:84   "away ~0.93× vs ACTUAL price (ranker-based 0.77–0.88 = wrong denominator)
                / 0.09× TOWARD argmax"
topic file     frontmatter: "vs the ACTUAL price ~0.93× … (overcharge ≤~12%)"
               body: keeps 0.77–0.88 explicitly labelled as dividing by the wrong denominator
fs2 memo       Series A — vs the RANKER (4.718; "ordering signal only, NOT prices")
               Series B — vs the ACTUAL price (4.1379): jg5 0.927, "the only pair where
               numerator and true price are same-object MEASURED"; jg2 1.000 · jg3 0.881
               marked DERIVED with the assumption stated and cited
```

Keeping the ranker-based figures **visible but labelled** rather than deleting them is the right
call — a reader who encounters `0.877` in an older memo can now find out what it was. That is the
append-only form applied to a superseded *derivation* rather than a superseded value.

---

## COUNTER

**1 / 3.**

Worth recording what happened here, because it is the first time the loop has run this way. You
challenged a range I had published, I checked my own basis, and **half of it did not exist**. The
finding underneath W2-F14 was real — the denominator mix was there and is now cured — but the
replacement I proposed carried an unverified leg into a headline, which is the identical defect at
one remove. Seven prior corrections of my own outputs; this is the eighth, and the second in two
rounds where the cure arm's scepticism was better calibrated than my prescription.

The toward-leg question is the more valuable half of this round precisely because the answer is
*don't*. A cure that makes two numbers look consistent by rescaling one of them onto the other's
denominator would have manufactured exactly the class of error both rounds were spent removing. The
legs are already correct; what they are not is comparable, and they should not be made to look it.

**Standing caveats unchanged:** em1's `0/46` reachability leg remains never-re-derived; the
equations-leg registration still rides a future landing and now owes the **Series B / price-based**
form, not the range I proposed in round 4.

fs3's status is noted for wave-3 staging; I have not reviewed it.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round5_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
