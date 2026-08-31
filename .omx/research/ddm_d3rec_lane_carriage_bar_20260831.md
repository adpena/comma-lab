# D3 RECONCILED AT SOURCE — the receipt exists, the 1.65× is a units artifact, and the Lane carriage bar is 21,699 B against two measured carriers at 1.66× and 2.42×

Date: 2026-08-31 · Author: MAIN · Cost: **$0** (primary-source read + 6 lines of arithmetic)
Axis: no new measurement. Every input CITED from a landed memo; the arithmetic is exact integers.
`score_claim=false` · `promotable=false` · `verdict_scope`: **INSTANCE** — the D3 rate leg and the two
measured Lane carriers, on this lineage.

## Why this exists

`ddm_rt3_route_rederivation_20260831.md` §8a downgraded its own headline and left two explicit,
ordered obligations:

> **Receipt path + sha for the −63,928 B: ABSENT.** I never held it. … A realization that exceeds its
> own predicted ceiling by 1.65× is not a strong re-open. It is an unreconciled pair of numbers. …
> **Before that: reconcile 38,649.8 vs 63,928. Routing the larger number first would repeat the error
> I just made.**

and

> D3 and gf1 are **two prices for one object** … **Nobody has put those two prices in one table.**

rt3 was explicit that it had **not read `d3`/`d3b` at source** — the figures reached it through a
delegated read. This is that read. It is the [[charter-provenance-claims-need-primary-implementation-trace]]
discipline applied to a coordinator's own carried number.

## 1. The receipt is NOT absent — it is at `ddm_d3:62`

| claim | rt3 status | AT SOURCE |
|---|---|---|
| −63,928 B receipt path + sha | **ABSENT** | `ddm_d3_alphabet_merge_20260826.md:62` — `retained/encode/token_stream_alphabet4_n600.bin`, **49,696 B**, sha `84fa2f499fb6c052cf6a43f8cae98c227ac32412ce1495cc715aa5af94b8692d` |
| what "receiver-closed" verified against | **ABSENT** | `:7-8` — *"The receiver reproduced all **117,964,800** merged symbols byte-identically"* |

**ABSENT-TO-RT3, not absent from the repo.** rt3 was right to refuse to assert what it had not held,
and right that the carry did not survive questioning — but the artifact exists, and the honest label
is "not read," not "not there."

## 2. THE 1.65× IS A UNITS ARTIFACT — rt3's own first hypothesis was correct

rt3 offered two: *"Either tba1's ceiling was scoped to the token stream while −63,928 B is a
whole-archive delta … or they are different quantities wearing one label."* **The first one is right,
and it closes exactly.**

`ddm_d3:20` gives the D3 rate-only archive composition:

```
49,696 (four-symbol RC64 stream) + 13,515 (four-class model) = 63,211
                                 + 53,076 (other archive bytes)
                                 = 116,287 B   ← "receiver-closed rate leg"
```

and therefore, against the GB1 pointer of **180,215 B**:

```
180,215 − 116,287 = 63,928 B      ← EXACT, verified
```

So **−63,928 B is a WHOLE-ARCHIVE delta** (stream **and** model together, plus framing), while
`af1` §7's **38,649.8 B** ceiling is a **token-stream-scoped** quantity. Two different denominators
under one label — the [[m99]] genus, which rt3 correctly smelled and correctly refused to route on.
**The pair is now reconciled; neither number was wrong, and neither refutes the other.**

## 3. THE TWO PRICES, IN ONE TABLE — and the bar they are measured against

`ddm_d3:8` is emphatic that the credit is *"**before Lane carriage**"*, and `:21` prices the only
carriage D3 itself measured. Setting the sub-0.12 byte target **137,986 B** against the D3 rate-only
archive gives the number rt3 asked for:

> ### LANE CARRIAGE BAR = 137,986 − 116,287 = **21,699 B**

| Lane carrier | coded B | vs the 21,699 B bar | composed archive | over target | S penalty @ 6.658589531221714e-7 |
|---|---:|---:|---:|---:|---:|
| `gf1` lane stream | **36,044** | **1.661×** | 152,331 B | +14,345 B | +0.009552 |
| D3 `block_s3_t3` | **52,539** | **2.421×** | **168,826 B** | +30,840 B | +0.020535 |

**Both carriers exceed the bar, and both were independently REFUSED on their own axis:**
- `block_s3_t3` — `ddm_d3:21` labels it verbatim an **"n600 scorer-refused instance."**
- `gf1`'s lane stream — the whole HG1 family was REFUSED at **5.09×**; its 36,044 B buys Lane at
  **318,406 mismatches (24.03% of the capacity gap)**, i.e. it does not hold Lane correctly either.

`gf1` is the cheaper of the two by **16,495 B**, which is the real content of "two prices for one
object" — but cheaper is still 1.661× over a bar that has to be cleared with distortion to spare.

## 4. Verdict

**The D3 rate leg is REAL, EXACT, and receiver-closed at 116,287 B — and it is NOT a route, because
the object it produces has no Lane.** rt3's separability finding stands and is sharpened: the credit
is *produced by* deleting Lane, so it cannot be banked apart from Lane's re-carriage, and every
measured re-carriage overshoots the remaining 21,699 B of room.

**verdict_scope: INSTANCE** — these two carriers, this lineage. It does **not** close "a cheaper Lane
carrier exists": [[m131]] records Lane at **0.59% of area but 33.56% of model bits / 33.97% of token
bits**, and nothing here bounds a carrier below 21,699 B. What it does is give that hypothetical
carrier a **hard, exact bar to beat**, which did not exist before this memo — and put both measured
prices on the same axis for the first time.

**The smallest measurement that would move this** (rt3 §8a's own, unchanged and still owed, now
UNBLOCKED by §2): on lb1's retained field
`ddm_dc1_20260816/retained/redecoded_tokens_n600.u8` (sha `9ba2e52b…`), fold Lane into its dominant
neighbour and run the same `count_nonzero` `gf1` uses
(`experiments/ddm_gf1_generator_form_on_lb1_field.py`) — pricing what a Lane-only carrier must
restore, in gf1's own currency, against the 21,699 B bar. Scorer-free, existing instrument, no
retrain. **Its ordering precondition (reconcile first) is now DISCHARGED.**

## 5. Denominator

Carried figures traced to source: **4** (−63,928 · 49,696 B stream · 71,549 B bar · the
scorer-refusal). Reconciled: **1** (the 38,649.8 vs 63,928 pair, exact). Confirmed-present that rt3
marked absent: **2** (receipt path+sha · receiver-closure basis). New exact quantities derived:
**2** (the 21,699 B Lane carriage bar · the two-price table). Measurements run: **0**. Arms spawned:
**0**. Dollars: **0**.

The exact pointer did not move. This unit did **not** achieve the goal — it discharged an ordering
precondition its source arm named, and replaced a 1.65× anomaly with an exact bar.

`[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9.`
