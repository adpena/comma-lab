# ddm_rv17 — wave-end adversarial review, ROUND 7: **CLEAN PASS — counter 1/3**

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [source + receipt review, scorer-free]` ·
`score_claim: false` · cost $0 · seventh sibling of `ddm_rv17_wave_end_review_round1-6_20260820.md`.

## THE ANSWER, FIRST

**Clean pass. Counter 1 / 3.** The first of this cycle, and it is earned: I found nothing, having
looked at the four things I was asked to look at plus four more I went hunting for on my own.

The decisive result is that **step 4A is executable exactly as written.** I built the real contest
layout — `evaluate.py` at the repository root, the packet staged at `submissions/<name>/` — and ran
the bare command the step prescribes, with no extra argument:

```
$ python3 verify_files_digest.py
evaluate.py: ../../evaluate.py (7da71a84ce24286b...)
runtime_files_sha256: e8dcbc6542d6f4752559726a6b88bd645f5974a2d941a0bbaef6f9932dc8cb8f
PASS: matches the packet's pinned runtime_files_sha256 (36 rows verified)
rc=0
```

An executor following only that block ships the script **and** validates it, and the validation
command runs as printed. The record-vs-behavior genus that generated findings in six consecutive
rounds is closed at the surface that drives behavior.

---

## PER-ITEM VERDICT ROWS

| # | checked | method EXECUTED | MEASURED result | verdict |
|---|---|---|---|---|
| 1a | step 4A ships the script | read the `REPUBLISH_AND_REPIN_HOSTED_ARCHIVE` block | binding **SHIP verify_files_digest.py** clause with the reason inline | **CLEAN** |
| 1b | copy source unambiguous | grep the prep-tree path in the same document | named **2×** in `SWAP_PROCEDURE.md` — "from the prep tree" resolves | **CLEAN** |
| 1c | run-and-require-PASS executable | **simulated the real contest layout** and ran the bare command | `PASS`, **rc=0**, `e8dcbc65…`, 36 rows | **CLEAN** |
| 2 | FREEZE (d) vs step 4A | read both | same obligation, same reason; (d) **defers to 4A** as authority | **CLEAN** |
| 3a | enumeration edit | read the non-runtime document list | `verify_files_digest.py` present, first after "this file" | **CLEAN** |
| 3b | edit disturbed nothing | `shasum -a 256 -c` from the frozen root | **36 OK** | **CLEAN** |
| 4 | standing substance | recompute S; compare pointer, archive, pin, rows | all identical to round 1 | **CLEAN** |

### 1c — why this is the load-bearing test

The prior rounds' failures were never in the arithmetic; they were in whether a stranger following
our text lands where we say. So I did not read the step and judge it plausible — I reconstructed the
executor's world. `upstream/evaluate.py` sits at the contest snapshot root and `upstream/submissions/`
sits beside it, so a published packet at `submissions/<name>/` resolves the script's first candidate
`../../evaluate.py`. It does, and the digest lands. The candidate ordering in the script was written
for exactly this geometry and it holds under test rather than under argument.

### 2 — the two surfaces do not compete

`FREEZE_CHECKLIST.md` (d) reads *"Step 4A ALSO ships `verify_files_digest.py` into the published
submission directory … Run it from the published tree root and require PASS before publication."*
It carries the same obligation and the same reason, and it **points at 4A** rather than restating
the copy source. That is single-source-of-truth done correctly: one authoritative instruction, one
pointer to it, no second version to drift out of sync. Had (d) re-specified the source, I would have
flagged it as a future divergence.

### 4 — substance, unchanged across seven rounds

```
S recomputed from components : 0.14827847122030852
archive sha / bytes          : df7fd266e1b7488c… / 180,456
runtime tree pin             : fdd5774921319a31…
pointer effective_frontier   : 0.14827847122030852 | df7fd266e1b7488c… | contest_cuda
pointer == recomputed S      : True
manifest data rows identical : True (36)
```

---

## WHAT I HUNTED FOR AND DID NOT FIND

A clean pass is only worth something if it says what it looked for. Four candidate defects I raised
against this cure myself and cleared on evidence:

1. **Is the script's own integrity protected?** It is not a manifest row, so no row check covers it.
   But the realistic failure — a stale or wrong script copied at step 4A — is caught by the required
   PASS, loudly: a script carrying the wrong `EXPECTED_DIGEST` fails against a correct tree. The
   script's sha is additionally pinned in `DOC_DIVERGENCE_RECEIPT_R5.json` (`52108a66…`), and the
   digest itself appears on four independent surfaces. Not a gap.
2. **Cross-generation staleness.** If a future generation 7 ships a different tree, the copied
   script still carries gen6's `EXPECTED_DIGEST`/`EXPECTED_ROWS` and will **FAIL** — blocking
   publication until the stager regenerates, which the script's own docstring instructs. This is
   fail-closed by construction, and is a strength of the design rather than a defect.
3. **Does FREEZE (d) omit the copy source?** Yes, deliberately — see item 2. Correct, not missing.
4. **Could the bare command fail in the executor's actual context?** Tested, not assumed. It does
   not.

Also carried forward and still correctly disposed: the untyped `ValueError` on a malformed manifest
row remains loud and unreachable with the shipped manifest, filed as a note.

---

## COUNTER

**1 / 3 — the first clean pass of this cycle.**

I said in round 5 that a clean round would be given if earned, and in round 6 that two lines in the
executor checklists would close this. Both lines landed, and I tested them rather than taking them.
Giving the pass is the same discipline as withholding it was.

**The arc, for the record.** Seven rounds found: no wrong number, no wrong pin, no unverifiable
claim, no mis-scoped receipt, and no score error — the sixteenth-move row has recomputed to
`0.14827847122030852` identically in every round it was checked. What they did find, six times
running, was one genus: **a correction landing in the record while the surface that drives behavior
went unchanged** — a stale heading over a corrected body, a pin sentence describing rows it did not
hash, a digest named in prose that prose could not specify, an obligation filed in frozen custody.
Each cure moved one step closer to the behavioral surface, and the last one moved it onto the
executor's own checklist and into an executable script. That is why this round is clean, and why I
expect the remaining two to be.

Round 8 should re-derive rather than inherit: run `verify_files_digest.py` itself, recompute S from
components, and re-read step 4A against the executor's real layout. Nothing in this memo is
authority for round 8 — my own round-2 recommendation was wrong once already.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round7_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
