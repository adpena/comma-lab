# ddm_rvf1 — execute the rv15 round-2 findings debt (F1 already landed by MAIN)

## MANDATE

ddm_rv15's wave-end round-2 adversarial review landed 19 findings, one HIGH, and reset the
3-clean-pass counter to 0. MAIN has already fixed and committed **F1** (the packet published the
superseded 3,422.7 s contest-CPU wall on 9 live surfaces; the MEASURED 4,369.6 s now appears on
all of them, commit `d9874dbf9b`). **Do not redo F1.** Execute the remaining findings, each to a
real cure or an honest refusal-with-reason. Nothing may exit as prose.

## SOURCE (read first, re-derive rather than trust the summary)

`.omx/research/ddm_rv15_wave_end_round2_review_20260820.md` — the full findings table with
per-finding receipts. Its own §3 assumption-challenge and scope-declaration block bind you.

## THE REMAINING FINDINGS (rv15's numbering; adjudicate every one)

- **F2/F3 — the port break-even bar is published without its band.** rv15 measured the `2.03×/2.77×`
  bar carries neither cd1's own ±61 s band (re-anchored on jg5 it is `2.22×/3.08×`) nor the k=2
  corner where the port MISSES frame B by −7.6 s. Removing `d2h_sync` alone moves break-even to
  1.75×. CURE: republish the bar WITH its band at every live citation, and state the k=2 corner.
  NOTE: ddm_rr8 has since measured the port at **6.007× scope-isolated / 4.450× conservative**, so
  the bar is cleared with margin at every corner — the defect is the PUBLISHED FORM, not the verdict.
- **F4 — the inversion's two halves were measured on unmatched instruments** (local torch=6 threads,
  BLAS unpinned vs T4 torch=1/OMP=MKL=1), against our own pin-`(code,weights,threads,batch)` law.
  rv15's read: the rr6 falsification survives a fortiori; the PER-CORE claim is an upper bound.
  CURE: re-state the per-core claim as a bound at source, or re-measure matched.
- **F7 — vr1 has a silent-OSError data-loss path.** Trace it, decide (fix / fail-closed / refuse
  with reachability argument), and pin whichever you choose with a test.
- **F10 a–e — vr1 detail findings, REFUTED individually.** Adjudicate each: real defect vs
  rv15 over-read. Honest "rv15 was wrong here" is a valid, valuable outcome — say it plainly.
- **F11 — ledger routings REFUTED.** The cross-cutting one. Find what is actually mis-routed and
  cure it structurally, not by editing rows.
- **F14 — `fire_local_advisory --dry-run` POISONS the attempt dir it previews** (writes at
  :168/:171/:227). A preview that mutates is a broken instrument; cure it and pin with a control
  that proves --dry-run leaves the dir byte-identical.
- **F15 — sd1's SessionStart monitor broadcasts a superseded `9` for 72 h.** Cure the staleness at
  the source, not the display.
- **F19 — the commit serializer refused rv15 on its OWN checkpoint.** Reproduce, diagnose, cure.
- **§2.5 — sd1 INDETERMINATE, unresolved in round 2.** rv15 named the resolving measurement.
  RUN IT, or state precisely why it cannot run.

## HARD CONSTRAINTS

- **Do NOT touch** `submissions/robust_current/jg5_sub015_runtime/` (seal-pinned custody) or
  `upstream/` (READ-ONLY). Do not re-fire any Modal row; MAIN owns dispatch.
- A live T4 row is in flight (`fc-01M0FZKTSY9ZRH2TEX27TZACKP`, ddm_rr8 decode wall-clock).
  Do not touch `/Volumes/APDataStore/pact/ddm_rr8/t4_wallclock_r1/`.
- `.py` edits need 2 genuine review-tracker passes. Commit via `tools/commit_autosha.sh`.
- Every negative verdict carries a `verdict_scope:` declaration at the narrowest supported level.
- Detached launches ONLY via `tools/launch_detached_process.py`.

## OPTIMAL FORM

- **Family reference:** the canonical two-landing cure (immediate fix + a STRICT-or-warn gate that
  refuses the re-introduction), CLAUDE.md "Bugs must be permanently fixed AND self-protected
  against", at its landed form. SCOPE reductions permitted: you may cure a subset of F10 a–e if you
  state per-row which and why. MECHANISM reductions FORBIDDEN: no cure may be a doc edit where the
  defect is in code, and no gate may be added without an EXECUTED positive control (rc≠0 proven).
- **Provenance pins:** rv15 memo `.omx/research/ddm_rv15_wave_end_round2_review_20260820.md`
  (commits `f3c517c853`, `1d3acc9b7e`, `1ac9c945b6`) · F1 cure commit `d9874dbf9b` ·
  cd1 memo `.omx/research/ddm_cd1_corrector_shipping_axis_decomposition_20260820.md` ·
  rr8 commits `083d351f95`, `fa4b93a3a3`, `296829a382`, `2e64c0b19b`.
- **PRIOR-LAW PREDICTION (derived, falsifiable):** the standing law is that review findings split
  roughly evenly between real defects and reviewer over-reads once re-derived at source. Predict
  ≥3 of the remaining findings are REAL code defects with executable cures, and ≥2 are honest
  rv15 over-reads that close as refuted-at-source. FALSIFIER: if ALL remaining findings are real
  defects, the review process is under-calling and that is itself the finding; if NONE are, the
  round-2 finding round was noise and the counter should not have reset — say either plainly.

## DELIVERABLE

`.omx/research/ddm_rvf1_rv15_findings_execution_20260820.md` — per-finding row
{finding · re-derived-at-source verdict · cure landed (commit) OR refused-with-reason · control
executed} + a MAIN-adjudication queue for anything you could not close. End with the own-vehicle
frontier line.
