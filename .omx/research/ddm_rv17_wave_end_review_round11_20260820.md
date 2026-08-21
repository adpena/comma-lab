# ddm_rv17 — wave-end adversarial review, ROUND 11: two MED findings — the citation guard is launderable, and 4A names two of three prep overrides; counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · eleventh sibling of `ddm_rv17_wave_end_review_round1-10_20260820.md`.

## THE ANSWER, FIRST

**Counter stays 0/3 — two MED findings, both in the new material.**

The erratum itself is **sound**: R8's **12/12 shas re-derive**, and the pass-8 backing claim
**checks out at source** — `REVIEW_PASS8_FRESH_EYES.md` does carry `model e35d1237…/70,453 B` among
the per-section shas for the gen2/rr4 candidate whose archive is `181,161 B`. The values were backed
at their own generation; the citation dangled when the manifest was regenerated per candidate. The
erratum's five-way account is accurate, and "erratum, not value-change" was the right call.

**R11-F1 — the new citation guard can be laundered.** Coverage is `token in erratum_section_text`.
Nothing requires the erratum to *address* the citation. I proved a real FAIL is silenced by an
unrelated mention — and, worse, by a header that explicitly *denies* being an erratum:

```
CTRL     (no erratum)                             rc=1  FAIL: cites line 99999 but inflate.py has 69 lines
LAUNDER  ("## Erratum on an unrelated matter")    rc=0  note: erratum-covered, not failed
NEGATED  ("## This is NOT an erratum")            rc=0  note: erratum-covered, not failed
```

**R11-F2 — 4A names two prep overrides; there are three diverged pairs.** `MANIFEST.sha256` also
diverges prep-vs-frozen, and its prep copy is where **five rounds of cures live**. No
executor-facing document says which copy publishes.

---

## ITEM 1 — R8 and the erratum — **CLEAN**

**12/12 shas re-derived**, including all three two-copy pairs and the two new tools:

```
MANIFEST.sha256          fea2dc4709b2 / ba6bbb45d499      BORROWED  5a76143ce08f / e49c14bf90b7  ← now diverged, receipted
archive_manifest.json    5b948c9032ec / 9349837fd9f6      verify_citations.py    1a32c3149170
SWAP_PROCEDURE.md        775f2a2b6b7f (4A extended)       verify_receipt_chain.py 1360a6926ada (flatness note)
PR_BODY 284d619d · verify_files_digest 52108a66 · FREEZE_CHECKLIST f1e3639a
```

**The backing claim checks out.** `REVIEW_PASS8_FRESH_EYES.md` lists the per-section shas including
`model e35d1237…/70,453 B`, and names `gen2 181,161 B` — the rr4 archive. So the erratum's core
assertion (values backed at their own generation; the citation, not the figures, went stale) is
true at source. Read as a first-time contest reviewer, §10.6 is accurate and sufficient: it names
all five ways, names the case trap, and states that the live §9.2→§10.2 chain never cited the
manifest — which is the fact that keeps this an erratum rather than a correction.

**The chain machinery firing organically twice this unit** — refusing the erratum edits until R8
landed, then catching the post-receipt `verify_citations.py` edit — is the strongest evidence yet
that the machine cures work. That is the guard behaving as designed against its own authors.

## ITEM 4 — standing substance — **CLEAN**

```
archive df7fd266e1b7488c… / 180,456 B · S 0.14827847122030852 · pointer match · 36 OK
verify_receipt_chain rc=0 · verify_citations rc=0 (1 verified, 2 erratum-covered, 0 ambiguous, 11 external)
```

---

## RV17-R11-F1 — MED — erratum coverage is defeatable by mere token presence

`verify_citations.py:49-56` + `:102-104`.

```python
def _erratum_text(doc_text):
    chunks, keep = [], False
    for line in doc_text.splitlines():
        if line.startswith("#"):
            keep = "erratum" in line.casefold()     # substring, anywhere in the header
        if keep:
            chunks.append(line)
    ...
if token in covered:                                # mere presence ⇒ downgrade to note
```

Coverage accumulates every line under any header whose text *contains* "erratum", and a citation is
excused if its token appears anywhere in that blob. Two consequences, both MEASURED above:

1. **Unrelated-mention laundering.** An erratum about a *different* matter that happens to quote the
   token silences a genuine failure. This is realistic, not contrived — errata quote citations by
   nature, and the packet's own §10.6 quotes the token it covers.
2. **Negated-header laundering.** `"## This is NOT an erratum"` satisfies the substring test and
   opens coverage for its whole section.

**Not currently exploited:** the organic sweep is `1 verified, 2 erratum-covered, 0 ambiguous,
11 external`, and both covered citations are the real R10-F1 pair genuinely addressed by §10.6. The
defect is that the guard's binding is optional at the author's discretion, which is the property it
was built to remove. It fails in the direction that produces a clean-looking PASS with no signal —
the class I have treated as a finding since round 4.

**CURE:** make coverage *specific and declared* rather than *incidental and inferred* — an explicit
machine-readable block (e.g. `citations_errata: [{doc, token, reason}]`) that names the citing
document as well as the token, so covering citation X in document A cannot excuse citation X in
document B; and match the header on a leading `Erratum` token rather than a substring, which closes
the negated-header vector.

## RV17-R11-F2 — MED — 4A names two prep overrides, but three pairs diverge

`SWAP_PROCEDURE.md` step 4A now reads: the published dir *"MUST include `verify_files_digest.py`
from the prep tree, and MUST publish the PREP-TREE `BORROWED_SUBSTRATE_ACCOUNTING.md` … in place of
the frozen copy."* Two explicit prep overrides.

MEASURED — three documents diverge prep-vs-frozen, all three receipted in R8:

| pair | prep | frozen | publish source stated? |
|---|---|---|---|
| `BORROWED_SUBSTRATE_ACCOUNTING.md` | `5a76143c…` | `e49c14bf…` | **yes** — 4A |
| `MANIFEST.sha256` | `fea2dc47…` | `ba6bbb45…` | **NO** |
| `archive_manifest.json` | `5b948c90…` | `9349837f…` | **NO** |

No executor-facing document names the publish source for the latter two; grep of
`SWAP_PROCEDURE.md` + `FREEZE_CHECKLIST.md` returns only the incidental phrase "two published
surfaces (the MANIFEST.sha256 header …)", which is not an instruction. The successor receipts also
dropped r3's `authoritative_source` field (`<absent>` in R8).

**Why the omission is load-bearing rather than pedantic.** 4A's phrasing "in place of the frozen
copy" implies that, absent such a clause, the frozen copy is what publishes. Applied to
`MANIFEST.sha256`, that inference ships the **pre-fix** header — dropping, in one step, the R2-F2
pin-sentence correction, the R3-F1/R4-F1 digest naming, the RV17-F7 working-directory line, and the
R6-F2 enumeration entry. Five rounds of cures live only in the prep copy, and the 36 data rows are
byte-identical between them, so nothing else would flag the substitution.

**CURE:** state the rule generally in 4A — *publish the prep-tree copy of every diverged
non-runtime document, frozen retained as history* — and list the three pairs, rather than naming
individual files as they come up. Restoring `authoritative_source` to the receipts would make the
record agree with the checklist.

---

## COUNTER

**0 / 3.** Two MED findings.

Both are the eleventh and twelfth instances of the same genus, now in its two mature forms: a guard
whose binding is decided by an author (R11-F1) and an obligation stated for the instance in front of
us rather than the class (R11-F2). Every cure in this campaign that named *the case at hand* came
back; the three that derived their own scope — `verify_files_digest.py`, the chain checker's
prep∩frozen coverage, and the flatness invariant — have not. R11-F2 is precisely 4A being written
per-file for the third time; R11-F1 is coverage being decided per-prose rather than per-declaration.

Eleven rounds still have not found a wrong score, a wrong pin, a wrong digest, a mis-scoped receipt,
or an unverifiable archive claim. `S = 0.14827847122030852` recomputed again from the frozen
archive's own bytes; 36/36 rows verify; both machine guards pass on the live tree.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round11_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
