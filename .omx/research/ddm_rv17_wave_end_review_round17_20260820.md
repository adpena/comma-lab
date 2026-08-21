# ddm_rv17 — wave-end adversarial review, ROUND 17: yes, one hand-chosen input set remains — `_CITE_RE`'s extension list; counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · seventeenth sibling of `ddm_rv17_wave_end_review_round1-16_20260820.md`.

## THE ANSWER, FIRST

**Counter stays 0/3 — one LOW finding, and it is the answer you asked for.**

You said: *"If you can still name a hand-chosen input set, that's the finding."* I can, and it sits
**upstream of every set you derived**: `_CITE_RE`'s extension list, `(py|json|md|sh|c|txt|yaml)`.
That list decides what *counts* as a citation, and therefore decides universe membership, which
citations get checked, and — since the shared derivation — which documents are owed a receipt. Every
derived set in both guards is downstream of it.

It already leaks, with the most ordinary failure mode a hand-written list has:

```
`yaml` is listed.  `yml` is not.

upstream/.github/workflows/eval.yml:30   ×3   invisible to the checker
upstream/.github/workflows/eval.yml:17   ×1   invisible to the checker
```

Four live tokens in the universe docs are neither verified, nor counted, nor classified external —
they are simply not seen.

**The cure itself is excellent and I could not fault it.** Sharing `_default_docs` rather than
duplicating it is the right move, and it is the first structural guarantee in this campaign: the
12-vs-11 drift I measured was possible *only* because the two sets had independent origins, and now
they cannot. **R14: 22/22 shas re-derive.**

---

## RV17-R17-F1 — LOW — the extension list is hand-chosen and upstream of every derived set

`verify_citations.py:39-41` — `_CITE_RE = ...\.(?:py|json|md|sh|c|txt|yaml):(\d+)`.

**Measured today:** 4 tokens fall through (`.yml`, above). **No active failure is masked** — those
targets are `upstream/…`, so they would classify as *external* (unresolvable-by-design breadcrumbs)
if recognized. And **no document is excluded from the universe** by the list: I swept every
prep/frozen top-level `.md`/`.txt` outside the 12 and found none whose citations are all
unlisted-extension. So the current blast radius is zero.

**Why it is still the finding you asked for.** The list is chosen by hand, it is the *root* input of
the derivation chain, and it has already demonstrated the classic list failure — a long form present,
its common short form absent. Two consequences are reachable rather than theoretical:

1. **A stale citation to an unlisted extension is silently unchecked.** The packet's own
   `MANIFEST.sha256` has extension `sha256` — unlisted. A `` `MANIFEST.sha256:31` `` citation, which
   is a natural thing to write and which I wrote myself in my round-2 memo, would be invisible.
   Others in reach: `.toml`, `.cfg`, `.rs`, `.lock`, `.zip`.
2. **The gate composes forward.** A doc whose only citations use an unlisted extension never enters
   the universe, so under the R16-F1 cure it is never owed a receipt either — the extension list
   silently sets the scope of *both* guards.

**CURE:** stop enumerating extensions. Match a filename-shaped token followed by `:N` and let the
existing three-way classification do the work — packet-resolvable, erratum-covered, or external.
That is the same move that fixed `_default_docs`: derive the set, then classify, rather than
pre-filtering by a hand-written membership test. If an allow-list must stay, add `yml` and `sha256`
now and state in the docstring that the list is the derivation's root input.

## ITEM — is the `frozen_only_docs` bucket the finding? — **No**

Your counter-argument holds and I verified its mechanism: the derived coverage check refuses any
universe doc missing from **all three** buckets, so a forgotten assignment is loud rather than
silent. The bucket is also a data-shape necessity, not a convenience — `README.md` exists only on the
frozen side, so a prep-side bucket structurally cannot carry it — and tracking its sha doubles as a
freeze-integrity row on an append-only tree. Principled extension, closed failure mode. Not a finding.

## ITEM — the shared-import scope notes — **endorsed**

**(i) `sys.path` co-location binding.** Correct call, and the alternative is worse: duplicating
`_default_docs` is precisely the independent-origin condition that produced R16-F1. Sharing is what
makes the drift structurally impossible, and both instruments live and run in the same directory.
Reviewed decision, properly disclosed.

**(ii) Higher receipt cadence from tracking 5 review memos + runbooks.** Real cost, correctly named.
It is the discipline working — append-only review history should owe a receipt when it changes — and
the cadence is the price of the guarantee, not a defect.

## ITEM — R14 and standing substance — **CLEAN**

```
R14: 22/22 shas verify · frozen_only_docs = [README.md]
chain rc=0     PASS 22 tracked shas / 3 derived pairs / all 12 citation-universe docs tracked
citations rc=0 PASS 27 verified / 9 erratum-covered / 0 ambiguous / 129 external
archive df7fd266e1b7488c… / 180,456 B · S 0.14827847122030852 · pointer match · 36 OK
```

Verified under the operator hold: gen6 frozen, #1111 HELD, packet and receipts unchanged.

---

## COUNTER

**0 / 3.** One LOW finding (R17-F1).

Your "nearly exhausted" test was a good one, and the honest result is that the prediction was *almost*
right: every set you named is now derived, and the one that remains is the one nobody looks at because
it does not look like a set — it looks like a regex. That is worth stating as the generalized lesson
of this cycle, because it is the seventh and cleanest instance of the pattern:

> **A hand-chosen input set does not stop being one because it is spelled as a pattern.** The
> extension list is a membership test written in regex syntax; it behaves exactly like
> `DEFAULT_DOCS = (...)` did, and it failed the same way — by omitting a member (`yml`) that nobody
> enumerated against reality.

That is also why I think this genuinely is close to the end: after the extension list, the
derivation chain bottoms out at the filesystem and at `evaluate.py`, both of which are external
givens rather than choices.

Seventeen rounds have still not found a wrong score, a wrong pin, a wrong digest, a mis-scoped
receipt, or an unverifiable archive claim. `S = 0.14827847122030852`, re-derived once more from the
frozen archive's own bytes.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round17_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
