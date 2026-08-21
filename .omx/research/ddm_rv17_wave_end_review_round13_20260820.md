# ddm_rv17 — wave-end adversarial review, ROUND 13: one LOW finding that refutes a disclosed safety claim; counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · thirteenth sibling of `ddm_rv17_wave_end_review_round1-12_20260820.md`.

## THE ANSWER, FIRST

**Counter stays 0/3 — one LOW finding, and its value is that it refutes one of the two residuals
you disclosed rather than that the vector itself is wide.**

R10's **12/12 shas re-derive**. The fence boolean closes my R12-F1 exploit path, and your
opposite-direction catch (a fenced `# comment` silently closing a live Erratum section) was a real
latent defect I had not named.

But a **single boolean cannot track fences**, and two of my four probes are **fail-OPEN**:

```
M_mixed    ``` fence containing one ~~~ line, then fake Erratum + declaration   rc=0  COVERED  ✗
I_indent   ``` fence containing a 4-space-indented ``` line, then fake Erratum  rc=0  COVERED  ✗
U_unclosed unclosed fence at EOF, real declaration after it                     rc=1  FAIL     ✓
T_info     ~~~ fence with an info string wrapping a fake declaration            rc=1  FAIL     ✓
```

`I_indent` is the one that matters: your disclosure argued the any-indent looseness is *"fail-closed
in the direction that matters."* **That holds only in the toggle-ON state.** Inside an open fence,
a loosely-indented marker toggles OFF and resumes real parsing — granting coverage from inside an
illustration, which is the exact direction the cure exists to prevent.

---

## ITEM 1 — R10 and the fence boolean

**12/12 shas verify.** The write-time sha assertion is a good hardening — a receipt that cannot
record a stale sha removes the failure mode that produced R8-F1.

### RV17-R13-F1 — LOW — a one-boolean fence tracker toggles on the wrong markers

`verify_citations.py::_declared_coverage:73-76`

```python
stripped = line.lstrip()
if stripped.startswith("```") or stripped.startswith("~~~"):
    in_fence = not in_fence
```

Two CommonMark divergences, both measured fail-open:

1. **Mixed markers.** A fenced block closes only with the *same* character. `~~~` inside a ```
   block is content; here it toggles the boolean off. Realistic carrier: documentation that
   illustrates one fence style inside another — which is precisely what a doc explaining this
   declaration syntax would contain.
2. **Indentation inside an open fence.** CommonMark treats a 4-space-indented line inside a fence as
   content, never as a close. `lstrip()` erases that distinction.

**CURE — a tuple instead of a boolean.** On open, record `(char, count)`; close only on an lstripped
line of ≥ `count` of the *same* `char`; bound the *opening* indent to ≤3 spaces. That is the minimal
correct fence tracker and it closes both vectors at once.

**Severity LOW, honestly calibrated:** narrower carriers than R12-F1's plain fenced example, not
currently triggered (organic sweep `1 verified, 3 erratum-covered, 0 ambiguous, 11 external`, all
three covered instances legitimate), and no score, digest, or row impact. I am reporting it because
you asked me to adjudicate the residual and the measurement contradicts its stated safety property.

## ITEM 2 — adjudicating your two disclosed residuals

**(i) Citation scanning stays ON inside fences — CORRECT. Endorsed.** The asymmetry is principled,
not a compromise: *granting* coverage is the fail-open direction and must be conservative, while
*scanning* is the fail-closed direction and should be liberal. Keeping the scan on means a
stale-looking fenced citation fails visibly instead of being silently ignored, and the design
composes cleanly — a fenced citation that is *legitimately* stale is handled by declaring it,
which is exactly what the declaration mechanism is for. Keep it.

**(ii) The any-indent toggle — REFUTED as stated.** The disclosure's reasoning is sound for one of
two states and the code has both. Toggling ON from outside a fence does skip more and grant less
(fail-closed, as argued); toggling OFF from inside a fence resumes parsing and grants coverage
(fail-open). `I_indent` measures the second. This residual should be closed by the tuple cure above
rather than accepted.

## ITEM 3 — is R9→R10 append-only strengthening the right placement? — **YES, and for a specific reason**

Append-only is right for the *chain*: R9 stays untouched, R10 supplements, and the checker always
adjudicates against the latest by rank — so the machine never sees the weaker ground.

The cold-reader question turns entirely on something other than the receipts, and you got it right:
**the decisive ground also landed inline in SWAP 4A**, which is what an executor reads *first*:

> *"…decisively, it carries 4 absolute `/Volumes/…` local custody paths vs 0 in the frozen copy, so
> publishing the prep copy would breach Public Disclosure Hygiene."*

I re-derived that measurement myself rather than inheriting it: **prep 4, frozen 0.** With the
ground at 4A, a reader who never opens a receipt still gets the reason; the receipt chain then
carries the durable record. Had the strengthening lived *only* in R10, I would have called it buried
— the R6-F1 genus again. It does not, so the placement is correct.

## ITEM 4 — standing substance — **CLEAN**

```
archive df7fd266e1b7488c… / 180,456 B · S 0.14827847122030852 · pointer match · 36 OK
verify_receipt_chain rc=0 · verify_citations rc=0 (1 verified, 3 erratum-covered, 0 ambiguous, 11 external)
```

---

## COUNTER

**0 / 3.** One LOW finding (R13-F1).

Worth naming plainly at round thirteen: the substance has not moved since round 3 and every finding
since has been in the *apparatus built to protect it*. That is not a criticism of the apparatus —
each guard has caught something real, twice catching its own authors mid-edit — but it is the honest
shape of where we are. The remaining defects are in a citation-coverage parser, and the parser now
needs the same thing every earlier cure needed: to stop approximating the rule and implement it.
`startswith("```") or startswith("~~~")` is an approximation of fence syntax the way
`"erratum" in header` was an approximation of section membership, and it fails the same way.

Thirteen rounds have still not found a wrong score, a wrong pin, a wrong digest, a mis-scoped
receipt, or an unverifiable archive claim. `S = 0.14827847122030852`, re-derived again from the
frozen archive's own bytes.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round13_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**

## Verdict scope (appended by MAIN per the verdict-scope gate)

verdict_scope: formulation — the REFUTED verdict above applies to the ONE-BOOLEAN
lstrip/startswith fence-tracking formulation in verify_citations.py (and specifically its
"any-indent toggle is fail-closed" disclosed-residual argument, which holds only in the
toggle-ON state). Measured by two executed probes (M_mixed, I_indent) against that
implementation. NOT refuted: the declaration-coverage family itself, which was cured in the
same wave by the (char, count) CommonMark-rule implementation (commit 1d15043423, receipt
DOC_DIVERGENCE_RECEIPT_R11.json) — all nine post-cure controls pass.
