# ddm_rv17 — wave-end adversarial review, ROUND 16: the answer to your own question is yes — `repo_only_docs` is still hand-named; counter RESETS to 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · sixteenth sibling of `ddm_rv17_wave_end_review_round1-15_20260820.md`.

## THE ANSWER, FIRST

**Counter RESETS to 0/3.** One MED finding, and it is the direct answer to the question you posed.

You asked: *"If any input set here is still hand-named, that's the finding."* One is. The citations
checker's universe is now **derived**; the chain checker's `repo_only_docs` is not — and the two are
now measurably out of step:

```
citations checker derives : 12 docs   (prep ∪ frozen, suffix-filtered, _CITE_RE-gated)
chain checker tracks      : 11 docs   (2 derived pairs + a hand-curated repo_only set)
in the citation universe but NOT chain-tracked : 8 of 12
  COMPLIANCE_RUNBOOK.md · GPU_ROUTING_VARIANTS.md · README.md · README_PUBLIC.md
  REVIEW_PASS5 · REVIEW_PASS7 · REVIEW_PASS8 · REVIEW_PASS10
```

**Your docstring landing is exactly right** — I re-derived it rather than inheriting it, and it
carries the real justification verbatim: *"both MEASURED empty at rv17 R15 (not justified by the prep
flatness invariant, which bounds an INTERSECTION while this universe is a UNION containing frozen
subdirectory files)."* That is the note landed as a reason, not as a citation of my memo. **R13:
14/14 shas re-derive.**

---

## RV17-R16-F1 — MED — the chain checker's `repo_only_docs` is hand-curated, and now narrower than its sibling's derived universe

**Why it matters, concretely.** The R11-F1 cure made coverage *"deliberate, greppable, and
reviewable."* The *reviewable* half depends on someone noticing a declaration was added — and for
tracked documents the chain checker enforces exactly that: a moved sha owes a successor receipt. For
the 8 untracked docs it does not. A stale citation **and** its covering `covered-citation:`
declaration can be added to `REVIEW_PASS7` in a single edit: the citations checker passes (covered),
no tracked sha moves, no receipt is owed, and nothing records that coverage was granted.

**Latent, not active — measured, not assumed:**

```
declarations / erratum-headers in the 7 untracked prep docs : 0 / 0 across all seven
organic citations run                                       : rc=0, 27 verified / 9 covered / 0 ambiguous / 129 external
```

**The tell that the rule is reactive rather than derived:** `REVIEW_PASS6` and `REVIEW_PASS9` are
tracked, and they are tracked *because they received errata in R12*. Their four sibling memos, which
sit in the same derived universe and carry the same citation exposure, are not. The operative rule in
practice is "track a doc once it gets an erratum" — applied by hand, after the fact.

**CURE:** derive `repo_only_docs` the way `_default_docs()` derives its set — at minimum, track every
document in the citations universe, so the guard that *grants* coverage and the guard that *records*
edits share one universe. That closes the composition seam rather than widening the list by one
document each time a memo earns an erratum.

## ITEM 1 — the docstring landing — **CLEAN**

Both boundary measurements reproduce independently (34 frozen subdirectory files / 0 `_CITE_RE`
matches; non-`.md`/`.txt` top-level / 0), the docstring states measurement as the ground and
explicitly disowns the flatness justification, and it names the derived-set print as the visibility
mechanism if either measurement changes. Organic run identical: `27 / 9 / 0 / 129`. Docstring-only
change, tracked sha moved, R13 filed — the discipline held.

## ITEM 2 — the broken-edit disclosure — **NOT the finding**

You asked me to say if the mis-anchored Edit is the round-16 finding. It is not, for three reasons:
the verification existed and fired (`ast.parse`, immediately); nothing downstream consumed the broken
state — no run, no receipt, no commit; and consistency demands it, because I made analogous
self-caught errors in round 9 (misreading `rc` through a pipe) and round 11 (a heredoc
`SyntaxError`), both disclosed, neither counted against you. Holding you to a standard I was not held
to would be the wrong kind of rigor. The disclosure is what good practice looks like.

## ITEM 3 — seal scope under the state change — **your position is correct; endorsed**

The seal binds the wave it was convened for. New frontier work opens a new review obligation. Four
reasons, the last of which I think is decisive:

1. **A counter over a growing set never converges.** If in-flight landings are absorbed, every
   landing resets the count and the seal can never complete while the campaign is active — which
   defeats the purpose of a seal.
2. **The reviewed object is stable.** gen6 is frozen, #1111 is operator-HELD, the receipts are
   unchanged. The thing under seal will not drift beneath it — I re-verified the archive, the 36
   rows, and the chain this round under the hold.
3. **Different objects, different questions.** The seal asks *is this packet correct*; frontier arms
   against the live rc2 body ask *is the next candidate better*. Merging them conflates a
   correctness gate with an exploration loop.
4. **The supersession case is already encoded.** If a frontier landing actually replaces the
   candidate, `SWAP_PROCEDURE` step 6 resets the counter, and the scaffold's own history shows
   rounds 1–13 did not carry across generations. So absorption is unnecessary: a real candidate
   change resets the seal by existing rule.

The one safeguard worth restating: the seal must never be cited as covering a candidate it did not
review. That is already the scaffold's behaviour, and it is what makes (4) safe.

## ITEM 4 — standing substance under the hold — **CLEAN**

```
archive df7fd266e1b7488c… / 180,456 B · S 0.14827847122030852 · pointer match · 36 OK · chain rc=0
```

---

## COUNTER

**0 / 3 — reset.** Round 15's clean pass was real for its scope and does not carry; that is the rule
working, not a re-litigation.

The finding is the sixth instance of the pattern that has been predictive since round 9, and it has
now closed a loop worth naming: the citations checker's universe was hand-named until round 15
derived it, and *deriving it is what made the chain checker's hand-named universe visible.* One guard
becoming principled exposed the seam in its neighbour. That is a good property of this apparatus —
each derivation makes the next hand-named set measurable — and it is also the reason I expect the
list of such seams to be finite and nearly exhausted: `diverged_files` is derived, `_default_docs` is
derived, the fence rule is implemented, coverage is declared, publish sources are typed.
`repo_only_docs` is the last input set in either guard that a human still chooses.

Sixteen rounds have still not found a wrong score, a wrong pin, a wrong digest, a mis-scoped receipt,
or an unverifiable archive claim. `S = 0.14827847122030852`, re-derived once more from the frozen
archive's own bytes, under the operator hold.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round16_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
