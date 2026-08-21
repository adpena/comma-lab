# ddm_rv17 — WAVE 2, ROUND 1b: fs2 verified — the refusal holds, the LAW extracted from it does not; counter 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · addendum to `ddm_rv17_wave2_review_round1_20260820.md` (items 3–6).
Item 2 of the wave-2 scope. Item 1 (fs1) remains in flight.

## THE ANSWER, FIRST

**The fs2 refusal is the strongest measurement in this landing and it verifies exactly. The two real
findings are both in the generalization drawn from it.**

Verified clean: two candidates genuinely **built and `stat`-able** (179,434 B and 180,493 B against
base 180,456), a byte-identical control, two independent instruments agreeing to **0.1164 bits**,
retention **86/86 flawless** (0 missing, 0 sha mismatch, 991,339,036 B summed exactly), the
instrument bug caught by the arm's own control and — verified — **not propagated** to any sibling
claim, and the REFUSED verdict correctly scoped to FORMULATION with an explicit
*"no claim here is a family kill."*

---

## RV17-W2-F3 — MED — the `0.88× away` constant is mis-sourced, and it is now a standing rule

I re-derived this myself rather than adopting the arm's report:

```
jg5   3.8373 / 4.718 = 0.81333      <- jg5's ACTUAL ratio
jg2   4.1379 / 4.718 = 0.87705      <- where 0.877 actually comes from
jg3   3.6471 / 4.718 = 0.77302
```

`ddm_jg2_sub015_chain_20260819.md:17-18` states it in its own words: *"+30 archive bytes = 4.1379
bits per changed token. That is **0.877x** jg1's modelled 4.718."* The fs2 memo pairs **jg5's
numerator with jg2's ratio**.

Three things compound it. The `0.877` belongs to the **thinnest** of the three measurements — jg2's
3-pair, ~58-token, +30 B edit set — while being attributed to the **fattest**, jg5's 8,654 tokens.
The real measured range is **0.773–0.877**, and both the memo and the standing `MEMORY.md` line
quote the **top** of it. And jg3's 0.773× is `delta_trustworthy: false` by its own memo.

**The verdict survives; the derivation does not.** The asymmetry is 0.81–0.88 against 0.087 —
**9.3× to 10.1×** — so direction-dependence is robust whichever "away" point you take. But a
constant that is now a standing pricing rule in `MEMORY.md` is mis-attributed and is the most
favourable single point of a range. This is the campaign's own named genus twice over:
**constants-are-poison** and **cross-regime constant transfer** — the latter being a discipline the
fs2 memo itself invokes elsewhere.

**CURE:** quote the range and its provenance — *"0.773–0.877× away, measured across three arms of
very different weight; jg5 (8,654 tokens) = 0.813×"* — and correct the `MEMORY.md` standing line to
the range, not its maximum.

## RV17-W2-F4 — LOW-MED — the extracted law escaped the scope discipline its parent obeyed

The §7 OPEN row *"The recapture law itself"* carries **no `verdict_scope`** while every sibling
claim in the §8 table does, and its text prescribes generally — *"the model is trustworthy to ~0.88x
moving away from the argmax and to ~0.09x moving toward it"* for **"any future token-field lever."**
That sentence is promoted verbatim into `MEMORY.md` as standing discipline.

Its evidence base is **two thresholds, one edit family, one body** on the "toward" side. Calling that
a law is INFERRED. The refusal was scoped with real care; the law extracted from it was not — and
the law is the part that outlives the memo.

## Three LOW findings, recorded

1. **Two values for the corrector's worth:** §3 says *18,895 bits = 2,362 B*, `WHY_SUPERSEDED.txt`
   says *18,834 bits (2,354 B)* — each correct against its own denominator, 8 B apart, same named
   quantity in two receipts.
2. **Stale paths in the superseded receipt:** `superseded_pre_corrector/FS2_TOKEN_RD_REPLAY.json`
   records payload paths at the **live** `retained/token_rd/…` locations, which now hold different
   arrays. The files were relocated and are correctly manifest-listed under their superseded shas, so
   a reader following the manifest is fine — one following the receipt's own pointers gets a sha
   mismatch.
3. **ANSWER-FIRST attributes an hv1 constant to the live body:** *"Measured on the live body … 86.6%"*
   — but 86.6% is labelled hv1 in §2, and the live body's Path B / Path A at u = 7.0 re-derives to
   **88.2%**. The error runs conservative (reality is better than quoted) and changes no verdict, but
   it is the same constant-transfer genus as W2-F3.

## What verified clean, and deserves saying

The **retention is textbook** — 86/86 rows re-hashed against disk, zero mismatches, both refused
candidates *built and kept* rather than merely priced, and the superseded run retained with a
`WHY_SUPERSEDED.txt` that states mechanism, measured gap, consequence-had-it-shipped, and the
structural cure. Keeping the bug's own receipt as evidence is exactly what the payload law is for.

The **instrument law verifies and did not propagate**: the 2.07% overstatement re-derives
(929,670.66 / 910,776 = 1.020746), its cause is correctly diagnosed as pricing the pre-corrector
table, and every downstream input in `FS2_DROP_LADDER.json` carries the **live** shas while the
superseded ones appear only under `superseded_pre_corrector/`. An overstatement found and fully
propagated is rarer than one found.

The **tool audit came back clean on every class**, including the one the arm expected to fail: the
shim's rebinding of `expect_raw_sha256` does not disable decode pinning, because all four jg5 call
sites pass it explicitly. Two honest notes carried forward — the shim's own repin is unasserted, and
it has never run against a real candidate.

---

## COUNTER

**0 / 3.** Five findings this round (2 MED, 3 LOW), all in fs2's generalization layer, none in its
measurements.

The pattern from round 1 holds and sharpens: **the measurements in this wave are sound; what is not
sound is where their results were allowed to rest.** Round 1 found a receipt-writer with no shared
home and an equations leg with no owner. Round 1b finds a constant promoted past its provenance and
a law promoted past its scope. Same genus, four surfaces.

Item 1 (fs1) remains with its arm in flight.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round1b_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — UNMOVED by fs2, gen6 frozen,
#1111 operator-HELD.
