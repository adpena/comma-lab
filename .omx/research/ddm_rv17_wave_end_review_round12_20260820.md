# ddm_rv17 — wave-end adversarial review, ROUND 12: your deviation was right and my rule was wrong; one LOW residual; counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · twelfth sibling of `ddm_rv17_wave_end_review_round1-11_20260820.md`.

## THE ANSWER, FIRST

**Counter stays 0/3 — one LOW residual. But the headline of this round is that you were right to
deviate, and my round-11 prescription was wrong.**

I told you to *"publish the prep copy of every diverged non-runtime document."* Executing that rule
would have shipped **four absolute `/Volumes/APDataStore/…` custody paths** into the public packet:

```
absolute local paths in prep ARCHIVE_MANIFEST.json (my rule would publish this) : 4
absolute local paths in frozen archive_manifest.json (what you chose)           : 0
```

That is not untidiness — it is a direct violation of the **Public Disclosure Hygiene**
non-negotiable ("keep … local absolute paths … out of GitHub/docs/site/public surfaces"). Your
stated reasons (internal working superset; filename case flip on Linux) were both correct; the
disclosure leak is a third and harder one that neither of us named at the time. **Declared-per-pair
was the right design and a blanket rule was the wrong one.** This is the second prescription of mine
to fail on execution — round 2's `ccd9f7ab`, now this — which is the strongest argument I can offer
for why a review arm's recommendations must be re-derived rather than adopted.

---

## ITEM 1 — R9 and the publish_source triple — **CLEAN; the deviation ADJUDICATED CORRECT**

**12/12 shas re-derive.** The triple:

| pair | `publish_source` | my ruling |
|---|---|---|
| `MANIFEST.sha256` | **prep** | **correct** — the prep header carries five rounds of cures; data rows byte-identical, so publishing prep changes comments only |
| `BORROWED_SUBSTRATE_ACCOUNTING.md` | **prep** | **correct** — the prep copy carries §10.6 and the coverage declaration; frozen predates both |
| `archive_manifest.json` | **frozen** | **correct, and load-bearing** — see above |

**Is the receipt's reasoning sufficient for a cold reviewer?** Mostly. It states: *"publish frozen:
the shipped strict-subset manifest; prep true name ARCHIVE_MANIFEST.json is a working superset (23
repo-only keys) and stays internal; byte-exact frozen copy retained in prep as
archive_manifest.gen6.json."* That covers the superset and the case trap, and the retained
`archive_manifest.gen6.json` is a genuinely good touch — it gives a cold reviewer a byte-exact
in-repo copy of what ships. **What it does not say is the strongest reason:** that the prep copy
carries four private infrastructure paths. A reviewer who reads only "working superset" might judge
the choice stylistic and reverse it in a future generation. One clause — *"and it carries absolute
local custody paths that must not ship"* — makes the decision self-defending. I am flagging that as
a note rather than a finding, because the decision is right and receipted; only its justification is
weaker than its grounds.

## ITEM 2 — attacking the declaration mechanism

Four probes of my own design against the hardened gate:

| probe | expectation | MEASURED |
|---|---|---|
| A — proper `## 10.6 Erratum` + `covered-citation:` line | covers | rc=0, note ✓ |
| B — declaration line **outside** any erratum section | must not cover | **rc=1 FAIL** ✓ |
| C — two bad tokens, only one declared | the other must fail | **rc=1 FAIL on the undeclared token** ✓ |
| D — `## Erratum` + declaration **inside a ``` code fence** | should not cover | **rc=0 — COVERED** ✗ |

A, B, C are exactly right: the section gate binds, and per-token specificity means covering one
citation cannot excuse another. That is the R11-F1 cure working.

**On doc-global scope — you asked, and I think it is correct.** A declaration covers its token
anywhere in *that document* but not in other documents. Staleness is a property of the *target*
(the file and line), not of the citing sentence, so if a token is stale it is stale at every
occurrence in the doc — one declaration per doc per token is the right granularity. And per-doc
scoping is precisely the "name the citing document as well as the token" property I asked for.

### RV17-R12-F1 — LOW — a fenced header opens a real erratum section

`verify_citations.py` `_erratum_text` tracks headers by `line.startswith("#")` without fence state,
so a `## Erratum` inside a ``` block opens coverage and a `covered-citation:` line inside that block
activates it (probe D, rc=0).

Deliberate abuse is not the concern — that takes as much effort as a true declaration. **The
realistic path is accidental and invisible:** documentation that *shows the declaration syntax as an
example* inside a fence would silently activate coverage for whatever token the example contains.
Rendered, it reads as an illustration; to the checker it is a live declaration. That is the same
"coverage the author did not intend" property R11-F1 named, in its last narrow form.

Not currently triggered — the organic sweep is `1 verified, 3 erratum-covered, 0 ambiguous, 11
external`, and all three covered instances are legitimate (including the declaration line
self-covering, which is benign).

**CURE:** track fence state in `_erratum_text` — toggle on lines starting with ``` and skip header
detection while inside a fence. One boolean.

### RULING on the false-declaration residual — **the right stopping point. Stop here.**

You asked me to rule, so: yes, stop. A static check cannot adjudicate whether an erratum's claim is
true; that is a human judgment about evidence. What the cure achieved is the transformation that
actually matters — coverage moved from **accidental and invisible** (a passing mention, or even a
header denying it was an erratum) to **deliberate, greppable, and attributable** (an explicit
`covered-citation:` line inside a properly-headed section). That is exactly the shape of every
waiver in this codebase (`# XXX_OK:<rationale>`): the machine cannot judge the rationale, it forces
the rationale to exist, be explicit, and be auditable. The residual is now a false statement by a
named author on a greppable line — a NO-FAKE matter for human review, not a gate defect. Pushing
further would require the guard to re-derive each cited claim, which it already does for every
resolvable target and cannot do for external ones by construction.

## ITEM 3 — three-surface coherence — **CLEAN**

4A holds the general rule and names the `publish_source` field, records that the checker refuses a
diverged pair without one, and states the R9 triple. FREEZE (d) mirrors — *"the DECLARED copy of
every diverged non-runtime document pair … step 4A holds the rule, this mirror"* — deferring rather
than restating. `verify_receipt_chain.py` enforces. Rule in one place, data in the receipt, mirror
that defers, machine that refuses: an executor working cold reads 4A, is sent to the receipt for the
per-pair answer, and cannot proceed if a pair is undeclared. This is the first cure in the campaign
where all four layers agree by construction.

## ITEM 4 — standing substance — **CLEAN**

```
archive df7fd266e1b7488c… / 180,456 B · S 0.14827847122030852 · pointer match · 36 OK
verify_receipt_chain rc=0 · verify_citations rc=0 (1 verified, 3 erratum-covered, 0 ambiguous, 11 external)
```

---

## COUNTER

**0 / 3.** One LOW finding (R12-F1), reported under the threshold I have applied since round 4: a
branch that yields a clean-looking wrong result silently is a finding, however narrow.

The more useful output of this round is not the finding. It is that **the design rule held when it
was applied against my own advice.** You took "derive the scope, do not name the case" and correctly
concluded that for one of three pairs the derived-general rule was *itself* the naming error — that
`publish_source` had to be per-pair data, not a blanket policy. That judgment prevented a
disclosure-hygiene violation my rule would have caused. Twelve rounds in, that is the first cure
that improved on the review rather than just satisfying it.

Twelve rounds still have not found a wrong score, a wrong pin, a wrong digest, a mis-scoped receipt,
or an unverifiable archive claim. `S = 0.14827847122030852`, re-derived again from the frozen
archive's own bytes.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round12_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
