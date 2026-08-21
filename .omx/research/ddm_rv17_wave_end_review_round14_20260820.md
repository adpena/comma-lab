# ddm_rv17 — wave-end adversarial review, ROUND 14: the fence cure IS at the fixed point; one LOW finding one level out; counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · fourteenth sibling of `ddm_rv17_wave_end_review_round1-13_20260820.md`.

## THE ANSWER, FIRST

**You asked for a plain answer on the fixed point, so: YES. The fence cure is at the fixed point.
The rule is implemented, not approximated.** I wrote six CommonMark cases outside your nine
controls and every one behaves correctly:

```
A_trailclose  close attempt with trailing content, inside a fence   rc=1  does not close  ✓
B_crosschar   backtick close attempt inside a ~~~ fence             rc=1  does not close  ✓
C_close2      closing fence indented 2, opened at 0                 rc=0  DOES close      ✓
D_decl3       declaration at exactly 3 spaces                       rc=0  covers          ✓
E_decl4       declaration at 4 spaces (indented code block)         rc=1  does not cover  ✓
F_long        3-backtick line inside a 5-backtick fence             rc=1  does not close  ✓
```

I could not construct a fail-open. `(char, count)` with a `^ {0,3}` bound and a whitespace-only
tail is the rule, and it is what the code now does. **R11's 12/12 shas re-derive.**

**But one LOW finding, and it is one level out from the parser:** `DEFAULT_DOCS` is
`("BORROWED_SUBSTRATE_ACCOUNTING.md",)` — the single document where round 10's defect happened. Three
published documents carry `FILE:LINE` citations; the guard scans one.

---

## ITEM 2 — the three residuals, adjudicated

**(i) Closing-fence indent bound — CONFIRMED CORRECT, no direction missed.** CommonMark allows a
closing fence 0–3 spaces of indent independent of the opening indent, so reusing `^ {0,3}` is right.
`C_close2` measures it: a fence opened at column 0 and closed at 2 spaces closes, and the post-fence
declaration covers (rc=0). The complementary direction is `I_indent` from round 13, now rc=1 — a
4-space marker inside an open fence is content. Both directions land where CommonMark puts them.

**(ii) Backtick info strings may not contain backticks — omission is safe, and safe in *both*
directions.** Your note said the granting direction is unaffected; the stronger statement is true.
If such a line were wrongly treated as an *opening* fence, the checker starts skipping — less
coverage granted, fail-closed. It cannot be wrongly treated as a *closing* fence, because a close
requires a whitespace-only tail and a backtick-bearing info string fails that test (`A_trailclose`
confirms the tail rule binds). So the unimplemented distinction has no fail-open expression. Endorsed
as-is; implementing it would add surface for no safety.

**(iii) Deep-nested list fences — VERIFIED against the corpus, not assumed.** I measured rather than
accepted the scope note:

```
4+-indented fence markers in the prep corpus     : 0
4+-indented covered-citation lines anywhere       : 0
```

The scope note is honest and now has a number behind it. The analogy to `packet_census_guard`'s
flatness invariant holds with one difference worth stating: flatness is *enforced* by a tool, while
this is *observed*. Fine at present size; if the corpus ever gains list-nested fences the note stops
being protective. That belongs in the docstring beside the flatness note, not as a finding.

## ITEM 1 — R11 — **CLEAN**

12/12 shas re-derive. The sibling you found while implementing the real rule — `_DECLARE_RE` now
requiring ≤3 spaces, since a 4-indented `covered-citation:` line is an indented code block by the
same principle as a fence — is the right catch, and `E_decl4` confirms it binds. That is the rule
being applied consistently rather than patched per-symptom, which is what "implement the rule" means.

## RV17-R14-F1 — LOW — the guard's document scope names the case, not the class

`verify_citations.py:37` — `DEFAULT_DOCS = ("BORROWED_SUBSTRATE_ACCOUNTING.md",)`.

MEASURED — documents carrying `FILE:LINE` citations under the checker's own `_CITE_RE`:

| document | citations | scanned by default |
|---|---|---|
| `BORROWED_SUBSTRATE_ACCOUNTING.md` | many | **yes** |
| `README.md` (ships in the packet) | 1 | **no** |
| `PR_BODY_DRAFT.md` (becomes the PR body) | 1 | **no** |
| `README_PUBLIC.md` | 1 | **no** |

**No active defect:** run explicitly over all three, the checker returns `rc=0 — 2 verified, 0
erratum-covered, 0 ambiguous, 1 external`. The unscanned citations are `upstream/evaluate.py:92` and
`upstream/README.md:114`, and they resolve.

**Why it is still worth a line.** The guard exists because a shipped document carried a stale
citation. Its default scope is the one file where that happened. A future stale citation in the
shipped `README.md` — the same defect, one file over — is not caught. This is the exact shape the
last six rounds have been converging on: `DEFAULT_DOCS` is a named case where a derived set belongs.

**CURE:** derive it — scan every published document that contains a `_CITE_RE` match (the packet's
shipped `*.md` / `*.txt` plus the repo-side published surfaces), the way the chain checker derives
its two-copy pairs from `prep ∩ frozen` rather than from a list.

## ITEM 4 — standing substance — **CLEAN**

```
archive df7fd266e1b7488c… / 180,456 B · S 0.14827847122030852 · pointer match · 36 OK
verify_receipt_chain rc=0 · verify_citations rc=0 (1 verified, 3 erratum-covered, 0 ambiguous, 11 external)
```

---

## COUNTER

**0 / 3.** One LOW finding (R14-F1).

I want to be precise about what this round did and did not find, because you asked a direct question
and it deserves a direct answer separated from the counter. **The fence implementation is done.** It
is the first artifact in this campaign I attacked with novel cases and could not move — six probes,
zero fail-opens, both disclosed residuals adjudicated sound, the third verified against the corpus
with a number. If the question is "did taking the rule all the way work," the answer is yes, and the
sibling `_DECLARE_RE` catch shows the rule generalizing on its own rather than needing another round.

The finding is not in that work. It is in the guard's *scope*, which is the same lesson at the next
level out — and I note without irony that this is the fifth consecutive round where the residual has
been "the mechanism is right, the set it applies to was hand-named." That pattern is now legible
enough to be predictive: whenever a cure lands, the next question worth asking is not *is the
mechanism correct* but *how was its input set chosen*.

Fourteen rounds have still not found a wrong score, a wrong pin, a wrong digest, a mis-scoped
receipt, or an unverifiable archive claim. `S = 0.14827847122030852`, re-derived once more from the
frozen archive's own bytes.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round14_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
