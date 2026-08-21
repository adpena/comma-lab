# ddm_rv17 — REVIEW WAVE 2, ROUND 1: two MED findings on the apparatus debt; counter 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · first round of the wave my seal `d2e1941446` explicitly scoped out.

## THE ANSWER, FIRST

**Counter 0/3 — two MED findings, both in the apparatus debt rather than in any measurement.**

**Scope honesty for this round:** I completed items **3–6** myself. Items **1–2** (the fs1 and fs2
arithmetic) are delegated to two supervised verification arms still in flight; their results are not
in this memo and this round therefore does not certify them. The counter is 0/3 on the findings
below regardless of what they return.

**W2-F1 — the R15 note cure has nowhere to live.** The post-write schema check is not a class cure,
because **there is no shared receipt writer**: `grep -rl DOC_DIVERGENCE_RECEIPT tools experiments
src` returns **nothing**. Every receipt is written by an ad-hoc one-shot script, so a check added to
one writer must be *re-remembered* by the next. That is wave 1's terminal lesson — named the case,
not the class — on a new surface.

**W2-F2 — the equations leg is unpaid with no owner and no trigger.** No repo-ledger task exists for
the fs2 direction-dependent factor; the memory leg exists (2,533 B) but all three wave-2 commits
carry `[no-triality]`. "Rides the next canonical-equations-touching landing" names no owner and no
fire condition, which is the deferral-scatter pattern CLAUDE.md explicitly extincts.

---

## ITEM 4 — the #1172 cure — **CLEAN, verified both directions**

My own round-18 probe, the one that produced the note, now passes — and the negative control still
fails, so the cure did not simply make everything green:

```
prep MANIFEST = 68 lines · frozen = 52 · publish_source = prep

MANIFEST.sha256:60    (valid in prep-68, absent in frozen-52)  → rc=0  PASS, 1 verified
MANIFEST.sha256:9999  (absent in BOTH)                          → rc=1  FAIL
organic citations                                               → rc=0  27 / 9 / 0 / 133  (byte-identical)
verify_receipt_chain                                            → rc=0  22 shas, R16
```

Resolution now consumes `publish_source` for two-copy names, exactly as the note specified, with no
collateral change to the organic counts. The fix landed at the right scope.

## ITEM 5 — the R15 note defect — real, cured in the instance, **not cured in the class**

Verified at source: R15 carries exactly one non-string note —
`/repo_only_docs/verify_citations.py/note`, a single-element **list** of 1,448 characters — and R16
is clean (zero non-string notes) and names the defect in `known_defect_in_predecessor`. The
instance handling is correct and append-only, and no machine consumer was harmed: the chain checker
reads sha fields, not notes.

### RV17-W2-F1 — MED — the post-write check has no home, so it must be re-remembered

MEASURED: `grep -rl "DOC_DIVERGENCE_RECEIPT" tools experiments src` → **no matches**. The receipt
writers are not in the repo at all; they are one-shot scripts (heredocs, ad-hoc), which is precisely
the condition the coordinator named in the class question.

A post-write assertion added to *the script that wrote R16* binds nothing about the script that will
write R17. The check is therefore in exactly the position `DEFAULT_DOCS` was in before wave 1 round
15: correct, and dependent on a human remembering it next time.

**Severity is MED, not higher, and I want the reasoning explicit:** notes are documentation-only to
machine consumers, so the blast radius of a mistyped note is presentational. But my wave-1 round-19
finding was that **receipt prose is load-bearing documentation** — I audited it as such and the
coordinator adopted that framing. A schema that can silently change the *type* of load-bearing
documentation is worth one shared function.

**CURE:** a single writer/validator the receipts are emitted through — even a small
`tools/write_divergence_receipt.py` with a typed schema validated **before** the write. That makes
the discipline structural rather than remembered, and it is the same share-don't-duplicate move that
ended the `_default_docs` drift.

## ITEM 6 — the equations-leg debt — **owed, and currently orphaned**

MEASURED:

```
repo-ledger task for the fs2 factor : none
memory leg                          : price_token_field_levers_by_real_reencode_20260820.md, 2,533 B
fa1c61ac64 / 671f4e3734 / 3aa19e9712 : all carry [no-triality]
```

### RV17-W2-F2 — MED — an unpaid triality leg with no owner and no fire condition

The fs2 result is a **measured law** (a direction-dependent price), not merely a negative — and laws
are the equations leg's content. CLAUDE.md is explicit that a finding is known only when it is
expressible in all three legs and they agree, and the `[no-triality]` tags are an honest
acknowledgment that this one is not.

**My adjudication of the question you posed:** registration does **not** have to land this wave. But
*"rides the next canonical-equations-touching landing"* is not an acceptable disposition, because it
names no owner and no trigger — it is an orphan by construction, and CLAUDE.md names deferral-scatter
as the disease. The acceptable dispositions are exactly two: register it now, **or** file it as an
owned task with a named fire condition, which is precisely the disposition that worked for #1172 and
which I endorsed at the time. Choose either; do not choose neither.

## ITEM 3 — em1 — **verified as far as I took it**

```
payload  /Volumes/APDataStore/pact/ddm_em1/retained/rc2_real_sm3r_codes_and_scales.npz
sha256   cdd4535249002d740db73acebd61dbedb4734b0cca24a0d71213a33057ec385a   == claimed cdd45352…
content  int8 codes + fp16 scales on named SM3R tensors — token_embed (480/96),
         frame_embed (4800/8), coord_mix (9600/96), blocks.0.dw — real ranges, real dtypes
```

This is genuine measured data on real bytes, not a synthetic fixture, which satisfies the NO-FAKE
class-3 rule directly. **Stated honestly: I verified the payload's identity and reality, not the
0/46 reachability arithmetic itself** — that leg would need the reachability computation re-run, and
I am flagging it as verified-in-custody but not re-derived. It should be re-derived before em1's
`#1147` closure is cited as settled.

---

## COUNTER

**0 / 3.** Two MED findings, both structural, neither in a measurement.

The shape is worth noting on the first round of a new wave: wave 1 ended by deriving every input set
in the guard pair, and wave 2 opens with the same genus one layer further out — the **writers** of
the receipts those guards police, and the **legs** a measured law is owed. Neither is a defect in
what was measured; both are defects in where the result was allowed to rest.

Items 1–2 remain outstanding with two arms in flight.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round1_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
