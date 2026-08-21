# ddm_rv17 — ROUND 20: **CLEAN PASS — 3/3. #1157 IS SEALED.**

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · closing member of `ddm_rv17_wave_end_review_round1-19_20260820.md`.
Seal evidence chain: `c0e58f01e0` (18) → `a017ac5bc2` (19) → this memo (20).

## THE SEAL

**Three consecutive fresh-angle passes found nothing. #1157 is SEALED.**

The seal's evidence, re-derived one final time from the frozen archive's own bytes — not from a
receipt, not from a prior memo:

```
archive.zip bytes  : 180456                                                    (== receipt)
archive.zip sha256 : df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080   (== receipt)

  seg  100 × 0.00020139            = 0.020139
  pose sqrt(10 × 6.37e-06)         = 0.007981227975693965
  rate 25 × 180456 / 37,545,489    = 0.12015824324461455
                                     ─────────────────────
  S                                = 0.14827847122030852   == canonical pointer, contest_cuda

MANIFEST rows      : 36 OK
verify_receipt_chain: rc=0 — 22 tracked shas, all 3 derived two-copy pairs covered (R15)
verify_citations    : rc=0 — 27 verified / 9 erratum-covered / 0 ambiguous / 133 external
```

**Answer to the seal question: no.** The 08-20 landing set contains no defect that would matter to
the object this wave was convened to protect. Across twenty rounds, no round found a wrong score, a
wrong pin, a wrong digest, a mis-scoped receipt, or an unverifiable archive claim. `S` has recomputed
to `0.14827847122030852` identically every single time it was checked.

---

## MY OWN ROUND-19 MEMO, UNDER THE ROUND-19 STANDARD

You asked me to turn my own lens on my own correction. The extractor I used to redo the sweep is
itself an instrument, so I audited its edge cases against the real corpus rather than reasoning about
them:

```
basenames with NO dot (guard skips)          : 27
DOTFILE basenames (my extractor IS imprecise): 0
TRAILING-dot basenames (empty extension)     : 0
distinct extensions : json md py sh txt yml   max length: 4   all word-class, all within {1,12}: True
```

**The honest result: my extractor has a real imprecision with zero instances.** For a dotfile like
`.gitignore:5` it would report `gitignore` as the "extension" — which is wrong about what an
extension *is*. There are none in the corpus, and `_CITE_RE` itself cannot match a trailing-dot token
(it requires ≥1 word character after the dot), so the imprecision is unreachable in both the
instrument and the thing it measures. **Conclusion stands, and this time the method is verified
rather than assumed.**

The other two round-19 verifications also hold: the doc-suffix re-measurement excluding the binary
(`file … | grep text`) returns none, and the rank verification rests on the chain's own declarations
— the unsuffixed receipt carries no `supplements` field while R4 declares it supplements the
unsuffixed one — rather than on the hard-coded `3`.

---

## THE SEAL'S SCOPE — precisely what is and is not covered

**COVERED — the 08-20 wave's landing set, as it stood unchanged since round 18:**

- `verify_citations.py` @ `e6f91a3e74` and `verify_receipt_chain.py` @ `74c8daaf5a`
- Receipts R3 → R15 in `gen6_receipts/`, including their prose, which I audited as load-bearing
  documentation in round 19 and found factually accurate
- The frozen gen6 packet: archive `df7fd266e…` @ 180,456 B, its 36-row runtime tree, and the
  derived two-copy pairs with their `publish_source` declarations
- The cured documents and the erratum/declaration corpus

**NOT COVERED — and each opens its own obligation:**

1. **The in-flight frontier arms' future landings.** `fs2` is live on a rate-axis candidate; `em1`
   landed a three-leg scoped negative closing #1147 without touching packet bytes. **Any row either
   produces opens a NEW review obligation** — this seal is not evidence about bytes it never saw.
   Per my round-16 ruling, which you endorsed: a real candidate change resets the packet counter
   through `SWAP_PROCEDURE` step 6, and the seal must never be cited as covering a candidate it did
   not review.
2. **The post-seal #1172 resolution fix**, which fires *at* this seal. It is new code by definition,
   and this wave's own evidence is that every finding from round 12 onward lived in newly landed
   guard code. **It needs its own review when it lands** — the seal is explicitly not a pre-approval
   of it.
3. **The operator-HELD #1111 submission chain.** Its independent packet review counter stands at
   **0/5** and is unstarted. This seal is about the wave's landings, not about publication readiness;
   the two must not be conflated.

---

## WHAT TWENTY ROUNDS ACTUALLY ESTABLISHED

The substance never moved, and that is the finding rather than the absence of one. **Everything
found after round 3 was in the apparatus built to protect the packet, never in the packet.** That
apparatus is now materially different from where it started: coverage is *declared* rather than
inferred, publish sources are *typed* rather than assumed, the fence rule is *implemented* rather
than approximated, and every input set a human once chose is now derived — from the filesystem, from
a receipt field, or from another guard's own function.

The single transferable lesson: **a correct mechanism with a hand-chosen input set fails at the input
set.** Nine consecutive rounds found exactly that, in nine different disguises — a stale heading, a
pin sentence, a prose recipe, a misplaced obligation, a lagging record, a hand-named doc list, a
regex that was a membership test in disguise. Each derivation made the next hand-named set visible,
which is why the sequence terminated instead of regressing. When a guard lands, the question worth
asking is not *is the mechanism correct* but *how was its input set chosen*.

And the counter-lesson, which I paid for four times: **a review arm's output is a claim, not an
authority.** My round-2 digest prescription was wrong; my round-11 blanket publish rule would have
breached disclosure hygiene; my round-14 assessment mis-called an active gap as latent; my round-18
sweep reached a correct verdict through an instrument measuring the wrong substring. Four failure
modes, one rule: re-derive, including from me. That rule is why this seal is worth something.

## COUNTER

**3 / 3 — SEALED.**

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
