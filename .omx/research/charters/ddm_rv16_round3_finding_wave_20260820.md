# ddm_rv16 — recursive adversarial review, ROUND 3 (finding round) over the 08-20 wave

## MANDATE

The clean-pass counter is **0**. `ddm_rv15` round 2 (`f3c517c853`) reset it with 19 findings;
`ddm_rvf1` executed them (5 cured, 3 refuted). This is **round 3, a FINDING round**.

The surface has GROWN since rv15. Four landings are **MAIN's own work from this session**, and MAIN
is RECUSED as their reviewer (CLAUDE.md council-conduct: authorship + same-session sister conflict).
**You are the fresh eyes on those four.** They are the highest-risk unreviewed surface in the wave
because one of them is a decisive claim on the submission critical path.

## PRIMARY TARGETS — MAIN's own landings (you are the only reviewer)

1. **`fa6863305c` — the rr8 T4 wall-clock verdict.** Claims the decode wall is CLEARED: inflate
   464.558564563 s vs jg5 1,419.9042126240001 s, score bit-identical, "the port SHIPS."
   **MAIN has already self-found one defect here and is correcting it in parallel** (the
   `[822, 1302]` bar is labeled a PROJECTION with estimated step seconds at `wc2:273`, and the memo
   used it as a measured budget). Do NOT re-report that one. Look for what MAIN did *not* find:
   - Is 464.6 s reproducible, or is a **single T4 row** being read as a stable wall? No repeat ran.
   - The host-variance factor **1.2855** was derived from ONE stage (`neural_render`) on ONE run.
     Is that a sound estimator, or is it itself a one-sample quantity carrying a decision?
   - The score-identity control is FORCED (identical archive bytes). Does the memo anywhere let a
     forced control read as evidence *for the port*?
   - `free_corrector=NativeFreeCorrector` proves the port ran. Does it prove the port ran **for
     every pair**, or only that it was selected once?
2. **`5047803cf0` — the rc1 composed candidate.** Claims the port × rider merge is clean and
   "decode-proven." Specifically pressure-test the **scope of the identity proof**: the parse-layer
   comparison was run on the composed tree, but the PORT leg's bit-identity is *inherited* from
   rr6/rr8 measurements on a **different tree**. Is the memo's title over-scoped? Is
   "37 files, decode-proven" a claim the evidence supports at composed-object scope?
3. **`edf4fc0608` — the rr5 jg5 re-measure.** −169 B, ΔS −1.125302e-04. The three controls cited
   (C1/C2/C3) were run on the **standalone rider**, not the composed object. Does the memo make
   that boundary explicit, or does it let standalone controls read as composed-object proof?
4. **`5c60d32af3` — the rvf1+gt2 harvest.** Both arms' memos landed; check whether MAIN's summary
   of their verdicts matches what the memos actually say.

## SECOND TARGET — the 25 uncommitted `.py` files

`rvf1` cured F7/F11/F14/F15/F19 in the working tree; the fixes are LIVE ON DISK and UNCOMMITTED
(`git status --porcelain | grep '\.py$'`). They will be committed. Review them as code that is
about to land, with particular attention to **F19's serializer self-inference**: the cure lets a
fresh, file-covering, same-session checkpoint be inferred as *self*. Prove a lone sister still
cannot be inferred as self, and that the inference cannot be spoofed by a stale same-session row.

## THE MANDATORY AXES (a round without these does NOT advance the counter)

- **Item 8 — assumption-challenge.** Name the shared assumption this wave operates within, and say
  whether violating it would change any verdict. Do not skip; a round missing this is incomplete.
- **Item 9 — measured-runnability + measured-scored-quantity.** For every claim that a thing WORKS
  or SHIPS, ask what was actually executed at real scale versus what was modeled or inherited.
- **verdict_scope on every negative**, at the narrowest supported level.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. Do NOT touch `submissions/robust_current/jg5_sub015_runtime/` (seal-pinned).
- **Do NOT fire any Modal dispatch.** MAIN owns dispatch. Do not touch
  `/Volumes/APDataStore/pact/ddm_rr8/` or `/Volumes/APDataStore/pact/ddm_rc1/`.
- `.py` edits: 2 genuine review-tracker passes; commit via `tools/commit_autosha.sh`.
- Detached launches ONLY via `tools/launch_detached_process.py`.
- ALWAYS KEEP THE PAYLOAD: any run that materializes bytes persists them with sha256 + count.

## OPTIMAL FORM

- **Family reference:** the canonical recursive adversarial review at its landed form —
  CLAUDE.md "Recursive adversarial review protocol", rotating adversarial perspectives, findings
  fixed-or-refused in the same round, counter advancing only on a genuinely clean pass. Reference
  instance: `ddm_rv15` (`f3c517c853`, 19 findings, counter reset). SCOPE reductions permitted and
  must be stated per row (e.g. reviewing a subset of the 106 commits — say which and why).
  MECHANISM reductions FORBIDDEN: no finding may rest on reading a headline instead of the source;
  no cure may be a doc edit where the defect is in code; no gate without an EXECUTED red run.
- **Provenance pins:** rv15 memo `.omx/research/ddm_rv15_wave_end_round2_review_20260820.md`
  (`f3c517c853`, `1d3acc9b7e`, `1ac9c945b6`) · rvf1 execution
  `.omx/research/ddm_rvf1_rv15_findings_execution_20260820.md` (`5c60d32af3`) · MAIN's four
  landings `edf4fc0608`, `fa6863305c`, `5047803cf0`, `5c60d32af3` · the rr8 T4 receipt at
  `/Volumes/APDataStore/pact/ddm_rr8/t4_wallclock_r1/contest_auth_eval.json` · wc2's band
  derivation `.omx/research/ddm_wc2_wall_clock_pass_20260820.md:273`.
- **PRIOR-LAW PREDICTION (derived, falsifiable):** the standing law from rv15→rvf1 is that findings
  split roughly evenly between real defects and reviewer over-reads. A SECOND law applies here:
  **self-authored work reviewed by its own author under-reports** — MAIN wrote four memos today and
  found exactly one defect in them, self-audited. Predict fresh eyes find **≥2 further real defects
  in MAIN's four landings** (over-scoped claim language and inherited-vs-measured boundaries being
  the likeliest classes), and **≥1 defect in the uncommitted `.py` set**. FALSIFIER: if MAIN's four
  landings are clean under fresh eyes, the self-audit was adequate and the recusal cost nothing —
  say so plainly, and the counter advances toward its first clean pass on that evidence.

## DELIVERABLE

`.omx/research/ddm_rv16_round3_finding_wave_20260820.md` — per-finding rows
{finding · re-derived-at-source verdict · severity · cure landed (commit) OR refused-with-reason ·
control executed}, the two mandatory axes answered explicitly, a counter verdict
(CLEAN → advance, or FINDINGS → reset), and a MAIN-adjudication queue for anything you cannot close.
End with the own-vehicle frontier line.
