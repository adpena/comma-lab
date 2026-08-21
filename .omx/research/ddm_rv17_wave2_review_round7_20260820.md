# ddm_rv17 — WAVE 2, ROUND 7: the seal is REFUSED — five ledger rows have no terminal state; counter RESETS to 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · seal-gate round; **seal WITHHELD**.

## THE ANSWER, FIRST

**Counter resets to 0/3. The seal waits.** Step 1 of your own checklist fired, and it fired on
exactly what it was designed to catch: **five of the fourteen findings terminate nowhere.**

```
ROW      SEV      TERMINAL STATE
W2-F1    MED      NONE — no shared receipt writer exists (grep tools/experiments/src: still blank)
W2-F4    LOW-MED  NONE — the law row at fs2:373 still carries NO verdict_scope
W2-F5    LOW      NONE — 2,362 B (memo, ×2) vs 2,354 B (WHY_SUPERSEDED, ×1), both live, unreconciled
W2-F6    LOW      NONE — superseded receipt still points at live `retained/token_rd/`, not at
                         `superseded_pre_corrector/` (measured: True / False)
W2-F7    LOW      NONE — fs2:22 still reads "Measured on the live body … 86.6%"; the live body is 88.2%
```

Only **W2-F2** is named in your carry list (the equations-leg RANGE registration). The other five
are neither cured, nor adjudicated-no-change with a citation, nor carried into wave 3 by name.

**Nine rows do terminate cleanly**, and I want that credited before the refusal: W2-F3 (four consumer
surfaces, verified round 4) · W2-F8 (erratum §E + fs3's 0.0991, verified rounds 2/4) · W2-F9 (cured at
the defect, `aff229623b`, verified round 4) · W2-F10 (family gone from every heading, verified round 2)
· W2-F11 (E4 with its positive control, verified round 2) · W2-F12 (terminates via F9's cure) ·
W2-F13 (four surfaces, verified round 4) · W2-F14 (Series A/B split, verified round 5) · and W2-F2
carried by name.

---

## WHY THIS IS THE WAVE'S OWN GENUS, ONE LAST TIME

Look at which five failed to terminate. Every one was raised in **round 1b** and never became the
subject of a cure round. The nine that closed are exactly the nine that drove a round — the HIGH,
and the MEDs that generated their own cure-verification cycles.

**A ledger closed on the rows that got attention, not on the rows it contains.**

That is the same shape as every other finding in this wave, at the last available level:

- fs1 mis-consumed a constant whose source sentence it had truncated → *the value travelled, the
  provenance did not*.
- The fs1 erratum cured the surfaces W2-F10 named and left W2-F9's number standing → *the cure
  scoped to the findings it was handed, not the defect*.
- W2-F3's correction reached the memo and not the memory hook → *the correction scoped to the
  document, not the consumers*.
- W2-F14's cure re-typed the range and carried three values computed under the denominator it had
  just disqualified → *the correction reached the premise, not the values derived from it*.
- And now: **the ledger scoped to the rows that had momentum, not the rows on it.**

**W2-F7 is the sharpest instance**, because it is the same defect as W2-F3 sitting in the same
memo. Both are "a constant attributed to the wrong object" — 0.88 attributed to jg5, and hv1's 86.6%
attributed to the live body. **The one with a finding chain behind it was cured across four surfaces
over three rounds; the one without is still there, in ANSWER-FIRST, at line 22.** The difference
between them is not severity or difficulty. It is whether a cure round happened to be convened.

## STEP 2 — the fresh lens I applied

The unapplied lens was **terminal-state completeness across the ledger itself** — not "is each
finding correct" (rounds 1–1c) or "is each cure correct" (rounds 2, 4, 5, 6), but *"does every row
have an ending?"* That question had never been asked, and it is the only one that could have caught
this: each of the five is individually small, and none of them would surface under a
finding-verification or a cure-verification pass, because **there is no cure to verify.**

I have no further lens to propose that I have not now used. If round 8 is clean, that is the seal.

## WHAT THE SEAL WOULD NEED

Each of the five needs exactly one of the three terminal states — and for four of them,
*adjudicated-no-change* is a perfectly honest ending:

- **W2-F1** — the class cure (a shared receipt writer with a typed schema validated before write) is
  real work. Either land it or carry it into wave 3 **by name**, as W2-F2 is.
- **W2-F4** — one clause: give the law row the `verdict_scope` its siblings carry.
- **W2-F5 / W2-F6 / W2-F7** — each is a one-line correction, and any of them may honestly be
  adjudicated *"immaterial, recorded, no change"* — but that adjudication has to be **written and
  cited**, not assumed. An unwritten adjudication is indistinguishable from an oversight, which is
  the whole reason step 1 exists.

## COUNTER

**0 / 3 — reset. Seal withheld.**

I want to be plain that refusing this seal is the checklist working rather than the wave failing.
Nine rows closed properly, the substance never moved, and the two adjudications I was asked for both
held. What step 1 caught is that a fourteen-row ledger had been reconciled by memory rather than by
walk — including by me, since I raised those five rows and then spent four rounds on the ones that
generated replies.

Eight of my own outputs corrected in this campaign; this is the ninth thing I got wrong, and it is
an omission rather than an error: **I never re-read my own round-1b findings after round 1b.** The
lesson from round 6 applies to me again, one level up — *a caveat is a debt*, and so is a finding.
I converted the em1 caveat when you told me to. Nobody had to tell me to re-walk my own ledger; I
just did not.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round7_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
