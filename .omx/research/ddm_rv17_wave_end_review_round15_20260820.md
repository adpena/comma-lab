# ddm_rv17 — wave-end adversarial review, ROUND 15: **CLEAN PASS — counter 1/3**; and my round-14 assessment was wrong

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · fifteenth sibling of `ddm_rv17_wave_end_review_round1-14_20260820.md`.

## THE ANSWER, FIRST

**Clean pass. Counter 1 / 3.** Earned, and arrived at by correcting my own error rather than by
finding nothing.

**My round-14 call of "latent, not active" was WRONG, and I re-derived why at source.** The derived
set immediately surfaced two live R10-F1-genus citations:

```
REVIEW_PASS9_FRESH_EYES.md:134  cites archive_manifest.json:36   shipped manifest has 20 lines  → stale
REVIEW_PASS6_FRESH_EYES.md:33   cites ARCHIVE_MANIFEST.json:50-51  case trap + 20-line shipped copy → stale
```

The instructive part is *why* I got it wrong. I enumerated only the **published** documents
(`README.md`, `PR_BODY_DRAFT.md`, `README_PUBLIC.md`), found their citations resolved, and concluded
latent. The real defects were in **internal prep memos I never enumerated** — which is precisely the
error I was diagnosing in `DEFAULT_DOCS` one paragraph earlier. I hand-named a narrower universe
than the class while writing the finding that hand-naming a universe is the class. Your scope note
(a) — including internal memos deliberately — is not just defensible, it is the decision that caught
what my enumeration missed.

**Both remaining hand-named boundaries measure EMPTY.** You asked me to say if either is the finding.
Neither is, and here are the numbers rather than an opinion:

```
frozen subdirectory files (cpr1/, runtime/)            : 34 files, 0 carry a _CITE_RE match
non-.md/.txt top-level files carrying citations        : 0
```

---

## PER-ITEM VERDICT ROWS

| # | checked | method EXECUTED | MEASURED | verdict |
|---|---|---|---|---|
| 1 | R12 receipt | re-hashed every tracked file | **14/14** verify; both review memos now tracked | **CLEAN** |
| 2 | the two stale citations | line-counted both manifests at source | 36 > 20 lines; 50-51 unreachable in the shipped copy | **CONFIRMED — my call corrected** |
| 3 | derived doc set | my own organic run | rc=0, `27 verified / 9 erratum-covered / 0 ambiguous / 129 external` | **CLEAN** |
| 4 | publish_source resolution | **my own** control on the frozen BORROWED copy | **rc=1** — `ARCHIVE_MANIFEST.json:21` *"resolves only case-inse…"* | **CLEAN — selector proven** |
| 5 | top-level-only boundary | grep over all 34 subdirectory files | **0** citations | **CLEAN** |
| 6 | .md/.txt suffix boundary | grep over non-suffix top-level files | **0** citations | **CLEAN** |
| 7 | standing substance | recompute from frozen bytes | `df7fd266…` / 180,456 B / `S 0.14827847122030852` / 36 OK / chain rc=0 | **CLEAN** |

**Control 4 is the one worth naming.** I did not take the publish-source claim on your word: running
the checker explicitly against the *frozen* BORROWED copy fails on the citation whose covering
declaration exists only in prep, while the organic run passes. That is direct proof the derived
selector consumed `publish_source=prep` rather than landing on the right copy by accident — and it
makes the citations checker a genuine second consumer of the R11-F2 field. The two guards now
compose: the chain checker enforces that the declaration exists, the citations checker consumes it
to decide which copy to judge.

## THE SCOPE NOTES, ADJUDICATED

**(a) Internal memos included — CORRECT, and vindicated by measurement.** Both real defects lived
there. Endorsed without reservation.

**(b) Undeclared two-copy pairs default to frozen — sound, by composition.** An undeclared *diverged*
pair is refused by the chain checker (R11-F2), so the default only governs *identical* pairs, where
the copy chosen is immaterial by definition. The one ordering caveat — running the citations checker
standalone, without the chain checker, would use frozen for an undeclared diverged pair — is
defense-in-depth sequencing rather than a hole, and you disclosed it.

**(c) Crash on malformed receipt JSON — fail-closed, consistent.** Same posture I accepted for the
manifest `ValueError` in round 6 and you accepted from me in round 12. Consistent in both directions.

**(d) The derived-set print names the docs — correct, and load-bearing.** A silently narrowed
universe is the failure mode this whole class produces; printing the set makes narrowing visible in
output rather than discoverable only by audit.

## ONE NOTE, DELIBERATELY NOT RAISED AS A FINDING

The *justification* offered for top-level-only does not transfer, even though the boundary is sound.
`packet_census_guard`'s flatness invariant bounds an **intersection** — a two-copy document must live
in both trees, so prep flatness caps it. This checker's universe is a **union**, and the frozen tree
contains 34 files in `cpr1/` and `runtime/` that flatness says nothing about. The boundary is safe
for a different reason than the one given: those 34 files carry **zero** citations, measured, not
because prep is flat.

I am recording this as a note rather than a finding under the threshold I have applied since round 4
and applied *against* raising in rounds 5 and 12: there is no wrong result, current or demonstrated —
both boundaries measure empty, and note (d) makes any future narrowing visible. It belongs in the
docstring beside the flatness note so the next maintainer inherits the real reason, not the
inherited one.

---

## COUNTER

**1 / 3 — the first clean pass of this cycle.**

I said in round 12 that a cure improving on the review would be worth naming. This round the review
itself was the thing corrected: I called a gap latent, the derived cure proved it active twice, and
the mechanism that found it was the one I had prescribed being applied *wider than I scoped it*.
That is the healthiest possible outcome for a fifteen-round cycle — the apparatus outrunning the
reviewer who specified it.

The input-set question that has driven the last six rounds now has a real answer here: the doc
universe is **derived**, its two residual boundaries are **measured empty**, and the derivation
**consumes another guard's declared field** instead of duplicating a judgment. That is the first
input set in this campaign I could not narrow.

Fifteen rounds have still not found a wrong score, a wrong pin, a wrong digest, a mis-scoped receipt,
or an unverifiable archive claim. `S = 0.14827847122030852`, re-derived once more from the frozen
archive's own bytes.

Round 16 should re-derive rather than inherit — including this memo. My round-2 and round-11
prescriptions and my round-14 assessment were each wrong on execution, which is three separate
proofs that a review arm's output is a claim, not an authority.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round15_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
